"""Render edit visualizations from a results.json produced by test_edit.py.

Outputs to --out-dir:
  index.html           — table of examples with correctness + overlap stats
  example_<i>.html     — per-example: problem, answers, inline word-diff for
                         R1 → R2_edit and R1 → R2_regen
  edit_density.png     — aggregate edit density vs normalized R1 position,
                         one curve for R2_edit, one for R2_regen
"""

from __future__ import annotations

import argparse
import html
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np


WORD_RE = re.compile(r"\S+|\s+")


def tokenize(text: str) -> list[str]:
    """Split into whitespace-preserving word tokens."""
    return WORD_RE.findall(text)


def render_diff_html(a: str, b: str) -> str:
    """Word-level inline diff. Returns HTML safe to drop inside a <div class='diff'>."""
    a_tok = tokenize(a)
    b_tok = tokenize(b)
    matcher = SequenceMatcher(a=a_tok, b=b_tok, autojunk=False)
    parts: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        a_chunk = html.escape("".join(a_tok[i1:i2]))
        b_chunk = html.escape("".join(b_tok[j1:j2]))
        if tag == "equal":
            parts.append(f'<span class="equal">{a_chunk}</span>')
        elif tag == "delete":
            parts.append(f'<span class="delete">{a_chunk}</span>')
        elif tag == "insert":
            parts.append(f'<span class="insert">{b_chunk}</span>')
        elif tag == "replace":
            parts.append(f'<span class="replace-del">{a_chunk}</span>')
            parts.append(f'<span class="replace-ins">{b_chunk}</span>')
    return "".join(parts)


def edit_mask(a_tokens: list[str], b_tokens: list[str]) -> np.ndarray:
    """Binary mask over a_tokens: 1 where the token was deleted or replaced in b."""
    mask = np.zeros(len(a_tokens), dtype=float)
    matcher = SequenceMatcher(a=a_tokens, b=b_tokens, autojunk=False)
    for tag, i1, i2, _j1, _j2 in matcher.get_opcodes():
        if tag in ("delete", "replace"):
            mask[i1:i2] = 1.0
    return mask


def binned_density(mask: np.ndarray, n_bins: int) -> np.ndarray:
    """Project a per-token mask onto n_bins equal-width bins over normalized position."""
    n = len(mask)
    if n == 0:
        return np.zeros(n_bins)
    edges = np.linspace(0, n, n_bins + 1)
    out = np.zeros(n_bins)
    for i in range(n_bins):
        lo = int(edges[i])
        hi = max(lo + 1, int(edges[i + 1]))
        hi = min(hi, n)
        out[i] = mask[lo:hi].mean() if lo < hi else 0.0
    return out


# ---------------------------------------------------------------------------
# HTML scaffolding
# ---------------------------------------------------------------------------

PAGE_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, sans-serif;
       max-width: 1100px; margin: 2em auto; padding: 0 1.5em; color: #222; }
h1 { font-size: 1.4em; } h2 { margin-top: 1.8em; font-size: 1.1em; }
pre.problem { white-space: pre-wrap; background: #f6f8fa; padding: 0.8em 1em;
              border-radius: 6px; font-size: 14px; }
table { border-collapse: collapse; margin: 0.5em 0 1.5em; font-size: 14px; }
td, th { border: 1px solid #d0d7de; padding: 0.3em 0.7em; text-align: left; }
th { background: #f6f8fa; }
.good { background: #d4f7d4; } .bad { background: #fddede; }
.diff { white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 13px; line-height: 1.55; border: 1px solid #d0d7de;
        padding: 1em; background: #fcfcfc; border-radius: 6px; }
.equal       { color: #24292f; }
.delete      { background: #ffe0e0; color: #82071e; text-decoration: line-through; }
.insert      { background: #dcffe4; color: #0a6b27; }
.replace-del { background: #ffe0e0; color: #82071e; text-decoration: line-through; }
.replace-ins { background: #dcffe4; color: #0a6b27; }
a { color: #0969da; text-decoration: none; } a:hover { text-decoration: underline; }
"""


def correctness_cell(correct: bool) -> str:
    cls = "good" if correct else "bad"
    mark = "correct" if correct else "wrong"
    return f'<td class="{cls}">{mark}</td>'


def render_example_page(i: int, r: dict) -> str:
    gold = html.escape(str(r.get("answer", "")))
    problem = html.escape(r["problem"])
    r1_ans = html.escape(r.get("r1_answer", "") or "[none]")
    regen_ans = html.escape(r.get("r2_regen_answer", "") or "[none]")
    edit_ans = html.escape(r.get("r2_edit_answer", "") or "[none]")

    diff_edit = render_diff_html(r["r1"], r["r2_edit"])
    diff_regen = render_diff_html(r["r1"], r["r2_regen"])

    overlap_edit = r.get("overlap_edit", 0.0)
    overlap_regen = r.get("overlap_regen", 0.0)
    lr_edit = r.get("len_ratio_edit", 0.0)
    lr_regen = r.get("len_ratio_regen", 0.0)

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Example {i}</title>
<style>{PAGE_CSS}</style></head>
<body>
<p><a href="index.html">&larr; index</a></p>
<h1>Example {i}
 &middot; level {html.escape(str(r.get("level", "")))}
 &middot; {html.escape(r.get("subject", "") or "")}</h1>

<h2>Problem</h2>
<pre class="problem">{problem}</pre>

<h2>Answers</h2>
<table>
  <tr><th>Source</th><th>Boxed answer</th><th>Correct?</th></tr>
  <tr><td>Gold</td><td>{gold}</td><td>&ndash;</td></tr>
  <tr><td>R1</td><td>{r1_ans}</td>{correctness_cell(bool(r.get("r1_correct")))}</tr>
  <tr><td>R2_edit</td><td>{edit_ans}</td>{correctness_cell(bool(r.get("r2_edit_correct")))}</tr>
  <tr><td>R2_regen</td><td>{regen_ans}</td>{correctness_cell(bool(r.get("r2_regen_correct")))}</tr>
</table>

<h2>R1 &rarr; R2_edit
  <small>(overlap {overlap_edit:.3f} &middot; len ratio {lr_edit:.2f})</small></h2>
<div class="diff">{diff_edit}</div>

<h2>R1 &rarr; R2_regen
  <small>(overlap {overlap_regen:.3f} &middot; len ratio {lr_regen:.2f})</small></h2>
<div class="diff">{diff_regen}</div>

</body></html>"""


def render_index(results: list[dict], summary: dict, config: dict) -> str:
    rows = []
    for i, r in enumerate(results):
        rows.append(
            "<tr>"
            f'<td><a href="example_{i}.html">{i}</a></td>'
            f'<td>{html.escape(r.get("subject", "") or "")}</td>'
            f'<td>{html.escape(str(r.get("level", "")))}</td>'
            f"{correctness_cell(bool(r.get('r2_edit_correct')))}"
            f"{correctness_cell(bool(r.get('r2_regen_correct')))}"
            f'<td>{r.get("overlap_edit", 0.0):.3f}</td>'
            f'<td>{r.get("overlap_regen", 0.0):.3f}</td>'
            f'<td>{r.get("len_ratio_edit", 0.0):.2f}</td>'
            f'<td>{r.get("len_ratio_regen", 0.0):.2f}</td>'
            "</tr>"
        )

    summary_rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{v:.4f}</td></tr>"
        if isinstance(v, float)
        else f"<tr><td>{html.escape(k)}</td><td>{html.escape(str(v))}</td></tr>"
        for k, v in summary.items()
    )
    config_rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{html.escape(str(v))}</td></tr>"
        for k, v in config.items()
    )

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Edit viz</title>
<style>{PAGE_CSS}</style></head>
<body>
<h1>Edit visualization</h1>

<h2>Config</h2>
<table>{config_rows}</table>

<h2>Summary</h2>
<table>{summary_rows}</table>

<h2>Aggregate edit density</h2>
<p>Fraction of R1 tokens modified in R2, averaged across examples, binned by
normalized position in R1. Higher = more editing at that region.</p>
<p><img src="edit_density.png" style="max-width: 900px;"></p>

<h2>Examples ({len(results)})</h2>
<table>
<tr>
  <th>#</th><th>subject</th><th>lvl</th>
  <th>edit?</th><th>regen?</th>
  <th>overlap_edit</th><th>overlap_regen</th>
  <th>lr_edit</th><th>lr_regen</th>
</tr>
{''.join(rows)}
</table>
</body></html>"""


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def plot_edit_density(results: list[dict], out_path: Path, n_bins: int = 40) -> None:
    import matplotlib.pyplot as plt

    edit_curves, regen_curves = [], []
    for r in results:
        r1_tok = tokenize(r["r1"])
        if not r1_tok:
            continue
        edit_tok = tokenize(r["r2_edit"])
        regen_tok = tokenize(r["r2_regen"])
        edit_curves.append(binned_density(edit_mask(r1_tok, edit_tok), n_bins))
        regen_curves.append(binned_density(edit_mask(r1_tok, regen_tok), n_bins))

    if not edit_curves:
        print("no data to plot")
        return

    edit_arr = np.stack(edit_curves)
    regen_arr = np.stack(regen_curves)
    xs = (np.arange(n_bins) + 0.5) / n_bins

    fig, ax = plt.subplots(figsize=(9, 4.5))
    em = edit_arr.mean(axis=0)
    rm = regen_arr.mean(axis=0)
    es = edit_arr.std(axis=0) / np.sqrt(len(edit_arr))
    rs = regen_arr.std(axis=0) / np.sqrt(len(regen_arr))

    ax.plot(xs, em, label=f"R2_edit (n={len(edit_arr)})", color="#0a6b27", lw=2)
    ax.fill_between(xs, em - es, em + es, color="#0a6b27", alpha=0.15)
    ax.plot(xs, rm, label=f"R2_regen (n={len(regen_arr)})", color="#82071e", lw=2)
    ax.fill_between(xs, rm - rs, rm + rs, color="#82071e", alpha=0.15)

    ax.set_xlabel("normalized position in R1 (0 = start, 1 = end)")
    ax.set_ylabel("fraction of R1 tokens modified")
    ax.set_ylim(0, 1.02)
    ax.set_title("Edit density along R1")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("viz"))
    ap.add_argument("--max-examples", type=int, default=None,
                    help="Only render first N example pages (plot still uses all).")
    ap.add_argument("--n-bins", type=int, default=40)
    args = ap.parse_args()

    payload = json.loads(args.results.read_text())
    results = payload["results"]
    summary = payload.get("summary", {})
    config = payload.get("config", {})

    args.out_dir.mkdir(parents=True, exist_ok=True)

    plot_edit_density(results, args.out_dir / "edit_density.png", n_bins=args.n_bins)

    to_render = results if args.max_examples is None else results[: args.max_examples]
    for i, r in enumerate(to_render):
        (args.out_dir / f"example_{i}.html").write_text(render_example_page(i, r))

    (args.out_dir / "index.html").write_text(render_index(to_render, summary, config))
    print(f"wrote {len(to_render)} example pages + index to {args.out_dir}/")


if __name__ == "__main__":
    main()
