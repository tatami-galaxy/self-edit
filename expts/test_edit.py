"""Compare editor-conditioned vs regenerator-conditioned responses on math failures.

Stage 1: sample R1 from the policy, keep the wrong ones.
Stage 2: for each failure, generate
    R2_regen via TEACHER_TEMPLATE (teacher sees feedback only)
    R2_edit  via EDITOR_TEMPLATE  (editor sees R1 + feedback)
Stage 3: correctness, token-level overlap with R1, length ratio.
"""

from __future__ import annotations

import argparse
import json
from difflib import SequenceMatcher
from pathlib import Path

from vllm import LLM, SamplingParams

from utils import (
    EDITOR_TEMPLATE,
    SYSTEM_PROMPT,
    TEACHER_TEMPLATE,
    DATASET_REGISTRY_EVAL,
    extract_boxed_answer,
    is_equiv,
)


def chat_messages(user_content: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_regen_feedback(mode: str, r1: str, wrong: str, gold: str) -> str:
    wrong_str = wrong if wrong else "[no boxed answer]"
    if mode == "minimal":
        return f"Your final answer was {wrong_str}, but the correct answer is {gold}."
    if mode == "with-r1":
        return (
            f"Your previous solution was:\n{r1}\n\n"
            f"Your final answer was {wrong_str}, but the correct answer is {gold}."
        )
    raise ValueError(f"unknown feedback mode: {mode}")


def build_edit_feedback(wrong: str, gold: str) -> str:
    # Editor prompt already shows R1 structurally, so feedback stays minimal.
    wrong_str = wrong if wrong else "[no boxed answer]"
    return f"Your final answer was {wrong_str}, but the correct answer is {gold}."


def stage1_r1(llm: LLM, sampling: SamplingParams, problems: list[dict]) -> list[dict]:
    prompts = [chat_messages(p["problem"]) for p in problems]
    outs = llm.chat(prompts, sampling)
    results = []
    for p, out in zip(problems, outs):
        r1 = out.outputs[0].text
        wrong = extract_boxed_answer(r1) or ""
        correct = bool(wrong) and is_equiv(wrong, p["answer"])
        results.append(
            {
                "problem": p["problem"],
                "answer": p["answer"],
                "level": p.get("level", 0),
                "subject": p.get("subject", ""),
                "unique_id": p.get("unique_id", ""),
                "r1": r1,
                "r1_answer": wrong,
                "r1_correct": correct,
            }
        )
    return results


def stage2_r2(
    llm: LLM,
    sampling: SamplingParams,
    failures: list[dict],
    feedback_mode: str,
) -> list[dict]:
    regen_prompts, edit_prompts = [], []
    for f in failures:
        regen_fb = build_regen_feedback(feedback_mode, f["r1"], f["r1_answer"], f["answer"])
        edit_fb = build_edit_feedback(f["r1_answer"], f["answer"])
        regen_prompts.append(
            chat_messages(TEACHER_TEMPLATE.format(question=f["problem"], feedback=regen_fb))
        )
        edit_prompts.append(
            chat_messages(
                EDITOR_TEMPLATE.format(question=f["problem"], r1=f["r1"], feedback=edit_fb)
            )
        )

    regen_outs = llm.chat(regen_prompts, sampling)
    edit_outs = llm.chat(edit_prompts, sampling)

    results = []
    for f, ro, eo in zip(failures, regen_outs, edit_outs):
        r2_regen = ro.outputs[0].text
        r2_edit = eo.outputs[0].text
        regen_ans = extract_boxed_answer(r2_regen) or ""
        edit_ans = extract_boxed_answer(r2_edit) or ""
        results.append(
            {
                **f,
                "r2_regen": r2_regen,
                "r2_edit": r2_edit,
                "r2_regen_answer": regen_ans,
                "r2_edit_answer": edit_ans,
                "r2_regen_correct": bool(regen_ans) and is_equiv(regen_ans, f["answer"]),
                "r2_edit_correct": bool(edit_ans) and is_equiv(edit_ans, f["answer"]),
            }
        )
    return results


def stage3_metrics(results: list[dict], tokenizer) -> list[dict]:
    def tok(s: str) -> list[int]:
        return tokenizer.encode(s, add_special_tokens=False)

    for r in results:
        r1_t = tok(r["r1"])
        regen_t = tok(r["r2_regen"])
        edit_t = tok(r["r2_edit"])

        r["len_r1"] = len(r1_t)
        r["len_regen"] = len(regen_t)
        r["len_edit"] = len(edit_t)
        r["len_ratio_regen"] = len(regen_t) / max(1, len(r1_t))
        r["len_ratio_edit"] = len(edit_t) / max(1, len(r1_t))

        # SequenceMatcher.ratio() in [0, 1]: 1 = identical, 0 = no overlap.
        r["overlap_regen"] = SequenceMatcher(a=r1_t, b=regen_t, autojunk=False).ratio()
        r["overlap_edit"] = SequenceMatcher(a=r1_t, b=edit_t, autojunk=False).ratio()

    return results


def summarize(results: list[dict], stage1_count: int) -> dict:
    def avg(key: str) -> float:
        vals = [r[key] for r in results]
        return sum(vals) / len(vals) if vals else 0.0

    n = len(results)
    summary = {
        "stage1_total": stage1_count,
        "stage1_failures": n,
        "stage1_failure_rate": n / max(1, stage1_count),
        "r2_regen_correct": avg("r2_regen_correct"),
        "r2_edit_correct": avg("r2_edit_correct"),
        "len_ratio_regen": avg("len_ratio_regen"),
        "len_ratio_edit": avg("len_ratio_edit"),
        "overlap_regen": avg("overlap_regen"),
        "overlap_edit": avg("overlap_edit"),
    }
    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"{k:24s} {v:.4f}" if isinstance(v, float) else f"{k:24s} {v}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--dataset", default="math500", choices=list(DATASET_REGISTRY_EVAL))
    ap.add_argument(
        "--levels",
        type=int,
        nargs="*",
        default=None,
        help="Filter by difficulty levels (e.g. --levels 3 4 5).",
    )
    ap.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Take the first N problems after level filtering.",
    )
    ap.add_argument(
        "--feedback-mode",
        choices=["minimal", "with-r1"],
        default="minimal",
        help="Feedback for R2_regen. Editor always uses minimal (R1 is already structural).",
    )
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--output", type=Path, default=Path("results.json"))
    args = ap.parse_args()

    loader = DATASET_REGISTRY_EVAL[args.dataset]
    problems = loader(levels=args.levels)
    if args.max_samples is not None:
        problems = problems[: args.max_samples]
    print(f"Loaded {len(problems)} problems from {args.dataset}")

    llm = LLM(
        model=args.model,
        seed=args.seed,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    stage1 = stage1_r1(llm, sampling, problems)
    failures = [r for r in stage1 if not r["r1_correct"]]
    print(f"[Stage 1] {len(failures)}/{len(stage1)} incorrect")

    stage2 = stage2_r2(llm, sampling, failures, args.feedback_mode)
    stage3 = stage3_metrics(stage2, tokenizer)
    summary = summarize(stage3, stage1_count=len(stage1))

    payload = {
        "config": {
            "model": args.model,
            "dataset": args.dataset,
            "levels": args.levels,
            "max_samples": args.max_samples,
            "feedback_mode": args.feedback_mode,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
        },
        "summary": summary,
        "results": stage3,
    }
    args.output.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nSaved {len(stage3)} results to {args.output}")


if __name__ == "__main__":
    main()
