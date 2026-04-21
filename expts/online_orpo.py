"""Online ORPO with self-revision.

Each step:
  1. Sample a batch of problems from the train set.
  2. Sample y1 from the current policy on the plain (system + problem) prompt.
  3. For y1-failures, sample y2 via EDITOR_TEMPLATE (shows y1 + ground-truth feedback).
  4. Grade y2. Keep pairs where y2 is correct: chosen = y2, rejected = y1.
  5. Compute ORPO loss on (problem -> chosen vs rejected) — we train the *base*
     policy P(y | problem), not the editor. The editor is just a sampler that
     produces a higher-quality counterfactual y2 to pair against y1.
  6. Every --sync-every steps, push the updated HF weights into the vLLM sampler.

Memory: vLLM + HF model + AdamW live in the same process. For Qwen3-4B on a
single 80GB card this is tight; use a smaller model, LoRA, or 8-bit AdamW if
it doesn't fit. Tune --vllm-mem to leave room for training activations.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

# IPC handles from reduce_tensor() carry a Callable; vLLM V1's engine-core IPC
# uses msgpack which rejects functions, so enable pickle fallback.
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

# Make the repo root importable regardless of whether the script is run as
# `python expts/online_orpo.py` or `python -m expts.online_orpo`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.multiprocessing.reductions import reduce_tensor
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferInitRequest,
    WeightTransferUpdateRequest,
)
from vllm.distributed.weight_transfer.ipc_engine import IPCWeightTransferUpdateInfo

from utils import (
    DATASET_REGISTRY_TRAIN,
    EDITOR_TEMPLATE,
    SYSTEM_PROMPT,
    build_edit_feedback,
    extract_boxed_answer,
    is_equiv,
)


def base_messages(problem: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]


def editor_messages(problem: str, r1: str, r1_answer: str, gold: str) -> list[dict]:
    fb = build_edit_feedback(r1_answer, gold)
    return [
        {"role": "user", "content": EDITOR_TEMPLATE.format(question=problem, r1=r1, feedback=fb)},
    ]


# --------------------------------------------------------------------------
# Rollout
# --------------------------------------------------------------------------


def rollout_pairs(llm: LLM, sampling: SamplingParams, batch: list[dict]) -> tuple[list[dict], dict]:
    """Sample y1 then y2 (editor) for every problem; always pair (chosen=y2, rejected=y1).

    No correctness-based filtering: when y1 is already right the editor template
    asks for a revision "if any", so y2 ~ y1 and the ORPO gradient is near-zero.
    Grading is kept only for diagnostics.
    """
    y1_prompts = [base_messages(ex["problem"]) for ex in batch]
    y1_outs = llm.chat(y1_prompts, sampling, use_tqdm=False)
    y1_texts = [o.outputs[0].text for o in y1_outs]
    y1_ans = [extract_boxed_answer(t) or "" for t in y1_texts]
    y1_correct = [bool(a) and is_equiv(a, ex["answer"]) for a, ex in zip(y1_ans, batch)]

    edit_prompts = [
        editor_messages(ex["problem"], y1_texts[i], y1_ans[i], ex["answer"])
        for i, ex in enumerate(batch)
    ]

    y2_outs = llm.chat(edit_prompts, sampling, use_tqdm=False)
    y2_texts = [o.outputs[0].text for o in y2_outs]
    y2_ans = [extract_boxed_answer(t) or "" for t in y2_texts]
    y2_correct = [bool(a) and is_equiv(a, ex["answer"]) for a, ex in zip(y2_ans, batch)]

    pairs = [
        {"problem": ex["problem"], "chosen": y2_texts[i], "rejected": y1_texts[i]}
        for i, ex in enumerate(batch)
    ]
    stats = {
        "n_batch": len(batch),
        "n_y1_correct": sum(y1_correct),
        "n_y2_correct": sum(y2_correct),
        "n_pairs": len(pairs),
    }
    return pairs, stats


# --------------------------------------------------------------------------
# Tokenization / batching
# --------------------------------------------------------------------------


def _encode(tokenizer, prompt_ids: list[int], resp_text: str, max_len: int):
    resp_ids = tokenizer(resp_text, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None and (not resp_ids or resp_ids[-1] != tokenizer.eos_token_id):
        resp_ids = resp_ids + [tokenizer.eos_token_id]
    full_ids = prompt_ids + resp_ids
    if len(full_ids) > max_len or len(resp_ids) == 0:
        return None
    labels = [-100] * len(prompt_ids) + resp_ids
    return full_ids, labels


def build_orpo_batch(pairs: list[dict], tokenizer, max_len: int) -> dict | None:
    """Tokenize pairs into (chosen, rejected) sequences, right-padded to batch max."""
    chosen, rejected = [], []
    for p in pairs:
        prompt_text = tokenizer.apply_chat_template(
            base_messages(p["problem"]), tokenize=False, add_generation_prompt=True,
        )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        c = _encode(tokenizer, prompt_ids, p["chosen"], max_len)
        r = _encode(tokenizer, prompt_ids, p["rejected"], max_len)
        if c is None or r is None:
            continue
        chosen.append(c)
        rejected.append(r)

    if not chosen:
        return None

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    longest = max(max(len(c[0]) for c in chosen), max(len(r[0]) for r in rejected))

    def pack(seqs, pad_val):
        return torch.tensor(
            [s + [pad_val] * (longest - len(s)) for s in seqs], dtype=torch.long
        )

    return {
        "chosen_input_ids":       pack([c[0] for c in chosen], pad_id),
        "chosen_labels":          pack([c[1] for c in chosen], -100),
        "chosen_attention_mask":  pack([[1] * len(c[0]) for c in chosen], 0),
        "rejected_input_ids":     pack([r[0] for r in rejected], pad_id),
        "rejected_labels":        pack([r[1] for r in rejected], -100),
        "rejected_attention_mask":pack([[1] * len(r[0]) for r in rejected], 0),
    }


# --------------------------------------------------------------------------
# ORPO forward / loss
# --------------------------------------------------------------------------


def _seq_logps(model, input_ids, labels, attention_mask):
    """Per-sequence (sum_logp, n_tokens) for response tokens (labels != -100)."""
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = out.logits[:, :-1, :]
    labels = labels[:, 1:]
    mask = (labels != -100)
    safe = labels.clamp_min(0)
    logp = F.log_softmax(logits.float(), dim=-1)
    tok_logp = logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1) * mask
    sum_logp = tok_logp.sum(-1)
    n_tokens = mask.sum(-1).clamp_min(1)
    return sum_logp, n_tokens


def orpo_loss(model, batch, beta: float):
    input_ids = torch.cat([batch["chosen_input_ids"], batch["rejected_input_ids"]], dim=0)
    attn     = torch.cat([batch["chosen_attention_mask"], batch["rejected_attention_mask"]], dim=0)
    labels   = torch.cat([batch["chosen_labels"], batch["rejected_labels"]], dim=0)

    sum_logp, n_tokens = _seq_logps(model, input_ids, labels, attn)
    B = batch["chosen_input_ids"].size(0)
    c_sum, r_sum = sum_logp[:B], sum_logp[B:]
    c_n,   r_n   = n_tokens[:B], n_tokens[B:]
    c_avg = c_sum / c_n
    r_avg = r_sum / r_n

    sft = -(c_avg).mean()

    # log1p(-exp(avg)) needs avg strictly < 0; clamp for numerical safety.
    eps = 1e-6
    c_clamp = c_avg.clamp(max=-eps)
    r_clamp = r_avg.clamp(max=-eps)
    log_odds = (c_avg - r_avg) - (torch.log1p(-torch.exp(c_clamp)) - torch.log1p(-torch.exp(r_clamp)))
    or_term = -F.logsigmoid(log_odds).mean()

    loss = sft + beta * or_term
    metrics = {
        "sft_loss": sft.item(),
        "or_loss": or_term.item(),
        "log_odds": log_odds.mean().item(),
        "reward_acc": (c_avg > r_avg).float().mean().item(),
        "chosen_logp": c_avg.mean().item(),
        "rejected_logp": r_avg.mean().item(),
    }
    return loss, metrics


# --------------------------------------------------------------------------
# Weight sync (HF -> vLLM, in-process)
# --------------------------------------------------------------------------


def _peft_to_hf_name(k: str) -> str | None:
    """Translate a PEFT state-dict key to the underlying HF name, or None to skip.

    PEFT wraps the model so keys look like 'base_model.model.<hf-name>' and
    LoRA-adapted linears get a '.base_layer.' infix. LoRA tensors are skipped —
    after merge_adapter() their contribution lives in base_layer.weight.
    """
    if ".lora_" in k:
        return None
    if k.startswith("base_model.model."):
        k = k[len("base_model.model."):]
    return k.replace(".base_layer.", ".")


def sync_weights_to_vllm(model, llm: LLM, gpu_uuid: str):
    """Push current (base + LoRA-merged) weights into vLLM via IPC.

    For each LoRA-wrapped linear we briefly merge the adapter into base_layer.weight,
    export the merged tensor, then unmerge to keep training state intact. Base
    params (embeddings, layernorms, unLoRA'd modules) are passed through as-is.
    """
    from dataclasses import asdict

    model.merge_adapter()
    try:
        names, dtype_names, shapes, ipc_handles = [], [], [], []
        for k, p in model.named_parameters():
            hf_name = _peft_to_hf_name(k)
            if hf_name is None:
                continue
            t = p.detach().contiguous()
            names.append(hf_name)
            dtype_names.append(str(t.dtype).split(".")[-1])
            shapes.append(list(t.shape))
            ipc_handles.append({gpu_uuid: reduce_tensor(t)})

        update_info = asdict(IPCWeightTransferUpdateInfo(
            names=names,
            dtype_names=dtype_names,
            shapes=shapes,
            ipc_handles=ipc_handles,
        ))
        llm.update_weights(WeightTransferUpdateRequest(update_info=update_info))
    finally:
        model.unmerge_adapter()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--dataset", default="deepmath", choices=list(DATASET_REGISTRY_TRAIN))
    ap.add_argument("--max-train-samples", type=int, default=10000)
    ap.add_argument("--num-steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Problems sampled per rollout. Actual training batch is <= this (only y1-wrong + y2-right survive).")
    ap.add_argument("--gen-temperature", type=float, default=0.9)
    ap.add_argument("--gen-top-p", type=float, default=0.95)
    ap.add_argument("--max-gen-tokens", type=int, default=2048)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=5e-7)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--sync-every", type=int, default=4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lora-target-modules", nargs="+", default=None,
                    help="Target module names for LoRA. Default: all linear modules.")
    ap.add_argument("--vllm-mem", type=float, default=0.4,
                    help="Fraction of GPU mem vLLM reserves. Lower this to leave room for training.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path, default=Path("orpo_log.jsonl"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    loader = DATASET_REGISTRY_TRAIN[args.dataset]
    ds = loader(max_samples=args.max_train_samples, seed=args.seed)
    data_list = list(ds)
    print(f"Loaded {len(data_list)} training problems from {args.dataset}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading HF model for training ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16,
    ).to("cuda")

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules or "all-linear",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    # LoRA + grad checkpointing: inputs need requires_grad so gradients flow
    # through the frozen base into the adapters.
    model.enable_input_require_grads()
    model.train()

    print("Loading vLLM for sampling ...")
    llm = LLM(
        model=args.model,
        seed=args.seed,
        dtype="bfloat16",
        gpu_memory_utilization=args.vllm_mem,
        weight_transfer_config=WeightTransferConfig(backend="ipc"),
    )
    llm.init_weight_transfer_engine(WeightTransferInitRequest(init_info={}))
    gpu_uuid = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)

    sampling = SamplingParams(
        temperature=args.gen_temperature,
        top_p=args.gen_top_p,
        max_tokens=args.max_gen_tokens,
        seed=args.seed,
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_file = args.output.open("w")

    step = 0
    while step < args.num_steps:
        batch = rng.sample(data_list, args.batch_size)
        batch = [{"problem": b["problem"], "answer": b["answer"]} for b in batch]

        pairs, roll_stats = rollout_pairs(llm, sampling, batch)
        if not pairs:
            print(f"[step {step}] rollout {roll_stats} -> skip")
            log_file.write(json.dumps({"step": step, "skipped": True, **roll_stats}) + "\n")
            log_file.flush()
            continue

        orpo_batch = build_orpo_batch(pairs, tokenizer, args.max_seq_len)
        if orpo_batch is None:
            print(f"[step {step}] all pairs exceeded max_seq_len, skipping")
            continue
        orpo_batch = {k: v.to("cuda") for k, v in orpo_batch.items()}

        loss, metrics = orpo_loss(model, orpo_batch, args.beta)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        log = {
            "step": step,
            "loss": loss.item(),
            "grad_norm": grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm),
            **roll_stats,
            **metrics,
        }
        print(
            f"[step {step:4d}] loss={log['loss']:.4f} sft={metrics['sft_loss']:.4f} "
            f"or={metrics['or_loss']:.4f} acc={metrics['reward_acc']:.2f} "
            f"pairs={roll_stats['n_pairs']}/{roll_stats['n_batch']}"
        )
        log_file.write(json.dumps(log) + "\n")
        log_file.flush()

        step += 1
        if step % args.sync_every == 0:
            sync_weights_to_vllm(model, llm, gpu_uuid)

    log_file.close()
    print(f"Done. Wrote log to {args.output}")


if __name__ == "__main__":
    main()
