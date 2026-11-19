"""AIGIBench (998 images, 10 subsets): our two image detectors vs SPAI (CVPR 2025) and GAPL (CVPR 2026),
alone and combined. SPAI/GAPL scores come from their official inference code (run_spai_official.py,
run_gapl_official.py, in .venv-eval); CF/B-Free scores from eval_aigibench.py.
Protocol (same as make_scoreboard.py): 5 random half/half splits, threshold on one half for <=5% real flagged,
metrics on the other half; combinations = logistic regression fitted on the tuning half.
Run: .venv-win\\Scripts\\python.exe experiments\\eval_aigibench_all.py
"""
import os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
B = os.path.join(os.path.dirname(HERE), "results", "benchmarks")
from experiments.make_scoreboard import protocol  # noqa: E402  (importing also rebuilds the scoreboard)

norm = lambda p: p.replace("\\", "/")
a = pd.read_csv(os.path.join(B, "aigibench_scores.csv")); a["path"] = a.path.map(norm)
s = pd.read_csv(os.path.join(B, "spai", "aigibench_input.csv")).rename(columns={"image": "path"}); s["path"] = s.path.map(norm)
g = pd.read_csv(os.path.join(B, "gapl_aigibench_scores.csv")); g["path"] = g.path.map(norm)
df = a.merge(s[["path", "spai"]], on="path").merge(g, on="path")
print(f"{len(df)} images ({int((df.label == 0).sum())} real / {int(df.label.sum())} AI) with all four scores")

methods = {"Community Forensics": ["commfor"], "B-Free": ["bfree"], "SPAI (CVPR 2025)": ["spai"], "GAPL (CVPR 2026)": ["gapl"],
           "CF + B-Free (app now)": ["commfor", "bfree"], "CF + B-Free + SPAI": ["commfor", "bfree", "spai"],
           "CF + B-Free + GAPL": ["commfor", "bfree", "gapl"], "All four": ["commfor", "bfree", "spai", "gapl"]}
res = {k: protocol(df, c) for k, c in methods.items()}
print(f"{'method':24s} {'AUC':>6s} {'caught':>8s} {'flagged':>8s} {'acc':>6s} {'F1':>6s}")
for k, r in res.items():
    print(f"{k:24s} {r['auc']:6.3f} {r['caught']:7.1f}% {r['flagged']:7.1f}% {r['acc']:5.1f}% {r['f1']:5.1f}%")
per = {sub: {k: round(100 * float((d[d.label == 1][c[0]] > 0.5).mean()), 1) for k, c in list(methods.items())[:4]}
       for sub, d in df.groupby("subset")}

from experiments.bench_record import save
save({"benchmark": "AIGIBench: CF, B-Free, SPAI, GAPL alone and combined (998 images; 5 splits; <=5% real flagged)",
      **{k: f"AUC {r['auc']:.3f}, caught {r['caught']:.1f}%, real flagged {r['flagged']:.1f}%, acc {r['acc']:.1f}%, F1 {r['f1']:.1f}%"
         for k, r in res.items()},
      "per_subset_caught_at_0.5 (CF, BFree, SPAI, GAPL)": {k: list(v.values()) for k, v in per.items()},
      "notes": "SPAI official CLI with --resize-to 1024 (6 GB GPU); GAPL official checkpoint, all 541 tensors loaded"},
     B, "aigibench_all_summary.json")
with open(os.path.join(os.path.dirname(B), "RESULTS.md"), "a", encoding="utf8") as f:
    f.write(f"\n## Newer AI-image detectors on AIGIBench ({len(df)} images: {int((df.label == 0).sum())} real / {int(df.label.sum())} AI)\n\n"
            "Held-out half, 5 splits, threshold for <=5% real flagged. SPAI run with its official tool (--resize-to 1024);"
            " GAPL with the authors' checkpoint (all weights loaded).\n\n| Detectors | AUC | AI caught | Real flagged | Accuracy | F1 |\n|---|---|---|---|---|---|\n")
    for k, r in res.items():
        f.write(f"| {k} | {r['auc']:.3f} | {r['caught']:.1f}% | {r['flagged']:.1f}% | {r['acc']:.1f}% | {r['f1']:.1f}% |\n")
    f.write("\nPer subset, AI images caught at score > 0.5 (CF / B-Free / SPAI / GAPL):\n\n| Subset | CF | B-Free | SPAI | GAPL |\n|---|---|---|---|---|\n")
    for sub, v in per.items():
        f.write(f"| {sub} | " + " | ".join(f"{x:.0f}%" for x in v.values()) + " |\n")
