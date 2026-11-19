"""Evaluates AI image detectors on the independent AIGIBench benchmark (Li et al. 2025):
500 real + 500 fake across 10 modern generators & wild subsets:
Midjourney, FLUX1-dev, SD3, Imagen3, DALLE-3, SDXL, CommunityAI, SocialRF, InstantID, StyleGAN-XL.

Evaluates:
  1) Community Forensics alone
  2) B-Free alone
  3) Both combined (logistic regression on held-out half, 5 splits)

Run: .venv-win\\Scripts\\python.exe experiments\\eval_aigibench.py
Writes results/benchmarks/aigibench_scores.csv, aigibench_summary.json, BENCHMARKS.md, and updates results/RESULTS.md.
"""
import os, sys, glob
import numpy as np, pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
DATA_DIR = os.path.join(os.path.dirname(HERE), "data", "benchmarks", "aigibench")
OUT = os.path.join(os.path.dirname(HERE), "results", "benchmarks")
CACHE = os.path.join(OUT, "aigibench_scores.csv")
os.makedirs(OUT, exist_ok=True)

from detectors.ai_image_detector import AIImageDetector
from detectors.bfree_detector import BFreeDetector
from detectors.spai_detector import SPADetector
from detectors.gapl_detector import GAPLDetector
from experiments.bench_record import save

print("Loading detectors on GPU...", flush=True)
cf = AIImageDetector()
bf = BFreeDetector()

try:
    sp = SPADetector()
    print("  SPAI loaded", flush=True)
except Exception as e:
    sp = None
    print(f"  SPAI unavailable: {e}", flush=True)

try:
    gp = GAPLDetector()
    print("  GAPL loaded", flush=True)
except Exception as e:
    gp = None
    print(f"  GAPL unavailable: {e}", flush=True)

rows = []
if os.path.exists(CACHE):
    df_cached = pd.read_csv(CACHE)
    done_paths = set(df_cached["path"])
    rows = df_cached.to_dict("records")
    print(f"Resuming from cache: {len(rows)} images already scored.")
else:
    done_paths = set()

# Collect all files
file_list = []
for subset in sorted(os.listdir(DATA_DIR)):
    s_path = os.path.join(DATA_DIR, subset)
    if not os.path.isdir(s_path):
        continue
    for cls in ("0_real", "1_fake"):
        cls_path = os.path.join(s_path, cls)
        if not os.path.isdir(cls_path):
            continue
        label = 1 if cls == "1_fake" else 0
        for f in sorted(os.listdir(cls_path)):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                full = os.path.join(cls_path, f)
                rel = os.path.join(subset, cls, f)
                file_list.append((rel, full, subset, label))

print(f"Total dataset: {len(file_list)} images across {len(set(x[2] for x in file_list))} subsets.")

count = 0
for rel, full, subset, label in file_list:
    if rel in done_paths:
        continue
    try:
        img = Image.open(full).convert("RGB")
        p_cf = float(cf.score_image(img)["prob_ai"])
        p_bf = float(bf.score_image(img)["prob_ai"])
        p_sp = float(sp.score_image(img)["prob_ai"]) if sp is not None else None
        p_gp = float(gp.score_image(img)["prob_ai"]) if gp is not None else None
        rows.append({"path": rel, "subset": subset, "label": label,
                     "commfor": p_cf, "bfree": p_bf, "spai": p_sp, "gapl": p_gp})
        count += 1
        if count % 50 == 0:
            print(f"  Scored {len(rows)}/{len(file_list)} images...", flush=True)
            pd.DataFrame(rows).to_csv(CACHE, index=False)
    except Exception as e:
        print(f"  Error reading {rel}: {e}")

df = pd.DataFrame(rows)
df.to_csv(CACHE, index=False)
print(f"Scoring complete. Total: {len(df)} images.")

# Evaluation protocol: 5 random splits, tune threshold on train half for <= 5% FPR, evaluate on test half
def at_fpr(y, s, ty, ts, max_fpr=0.05):
    fpr, _, th = roc_curve(ty, ts)
    valid_idx = np.where(fpr <= max_fpr)[0]
    t = th[valid_idx[-1]] if len(valid_idx) > 0 else 0.5
    caught = ((s > t) & (y == 1)).sum() / max(1, (y == 1).sum())
    flagged = ((s > t) & (y == 0)).sum() / max(1, (y == 0).sum())
    return float(caught), float(flagged), float(t)

methods = {
    "Community Forensics alone": ["commfor"],
    "B-Free alone": ["bfree"],
    "SPAI alone": ["spai"],
    "GAPL alone": ["gapl"],
    "Both combined": ["commfor", "bfree"],
    "All four combined": ["commfor", "bfree", "spai", "gapl"]
}

res = {k: [] for k in methods}
per = {k: {} for k in methods}

for seed in range(5):
    d = df.sample(frac=1, random_state=seed)
    a, b = d.iloc[:len(d) // 2], d.iloc[len(d) // 2:]
    for k, cols in methods.items():
        if len(cols) == 1:
            sa, sb = a[cols[0]].values, b[cols[0]].values
        else:
            lr = LogisticRegression(max_iter=1000).fit(a[cols], a.label)
            sa, sb = lr.predict_proba(a[cols])[:, 1], lr.predict_proba(b[cols])[:, 1]
        c, f, t = at_fpr(b.label.values, sb, a.label.values, sa)
        res[k].append((roc_auc_score(b.label, sb), c, f))
        for g, gb in b[b.label == 1].assign(s=sb[b.label.values == 1]).groupby("subset"):
            per[k].setdefault(g, []).append((gb.s > t).mean())

title = f"AI images: AIGIBench independent benchmark ({len(df)} images, 10 subsets)"
print(f"\n{title}\n{'method':35s} {'AUC':>6s} {'caught':>8s} {'real flagged':>13s}")
summary = {"benchmark": title}
for k, v in res.items():
    v = np.array(v)
    mean_auc = float(v[:, 0].mean())
    mean_caught = float(v[:, 1].mean())
    mean_flagged = float(v[:, 2].mean())
    print(f"{k:35s} {mean_auc:6.3f} {mean_caught * 100:7.1f}% {mean_flagged * 100:12.1f}%")
    summary[k] = f"AUC {mean_auc:.3f}, caught {mean_caught * 100:.1f}%, real flagged {mean_flagged * 100:.1f}%"

ks = list(methods)
print(f"\ncaught per subset (AI images only):\n{'':20s}" + "".join(f"{k[:20]:>22s}" for k in ks))
for g in sorted(per[ks[0]]):
    print(f"{str(g)[:20]:20s}" + "".join(f"{np.mean(per[k][g]) * 100:21.1f}%" for k in ks))
summary["per_group_caught_pct"] = {str(g): [round(float(np.mean(per[k][g]) * 100), 1) for k in ks] for g in per[ks[0]]}

save(summary, OUT, "aigibench_summary.json")

# Append table to results/RESULTS.md
with open(os.path.join(os.path.dirname(OUT), "RESULTS.md"), "a", encoding="utf8") as f:
    f.write(f"\n## Independent benchmark: AIGIBench (1,000 images: 500 real vs 500 AI, 10 subsets)\n\n"
            "| Tools | AUC | AI images caught | Real flagged |\n|---|---|---|---|\n")
    for k in methods:
        v = np.array(res[k])
        bold = "**" if k == "Both combined" else ""
        f.write(f"| {bold}{k}{bold} | {bold}{v[:, 0].mean():.3f}{bold} | {bold}{v[:, 1].mean() * 100:.1f}%{bold} | {bold}{v[:, 2].mean() * 100:.1f}%{bold} |\n")
    f.write("\n### Per-generator / subset breakdown (share caught @ 5% real flagged)\n\n"
            "| Generator / Subset | Community Forensics | B-Free | SPAI | GAPL | All Combined |\n|---|---|---|---|---|---|\n")
    for g in sorted(per[ks[0]]):
        scores_g = summary["per_group_caught_pct"][g]
        f.write(f"| {g} | {scores_g[0]:.1f}% | {scores_g[1]:.1f}% | {scores_g[2]:.1f}% | {scores_g[3]:.1f}% | **{scores_g[4]:.1f}%** |\n")

print("\nSaved to results/benchmarks/aigibench_summary.json, BENCHMARKS.md, and RESULTS.md")


