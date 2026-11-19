"""Separate vs combined: TruFor + Community Forensics on CocoGlide (AI-inpainted edits).
Uses the TruFor scores already saved by eval_localization.py, adds the AI-detector score for
the same 1,024 images, then compares (on a held-out half, 5 random splits):
  TruFor alone | AI detector alone | max of both | logistic-regression combiner
Run: .venv-win\\Scripts\\python.exe experiments\\eval_combined.py
"""
import os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
OUT = os.path.join(os.path.dirname(HERE), "results", "benchmarks")
loc = pd.read_csv(os.path.join(OUT, "localization_eval.csv"))

cache = os.path.join(OUT, "cocoglide_ai_scores.csv")
if os.path.exists(cache):
    ai = pd.read_csv(cache)
else:
    from detectors.ai_image_detector import AIImageDetector
    det = AIImageDetector()
    rows = []
    for _, r in loc.iterrows():
        f = os.path.join(os.path.dirname(HERE), "data", "benchmarks", "fake" if r.label == 1 else "real", r.file)
        rows.append({"file": r.file, "prob_ai": det.score_image(f)["prob_ai"]})
    ai = pd.DataFrame(rows); ai.to_csv(cache, index=False)
df = loc.merge(ai, on="file")

def at_fpr(y, s, thr_fit_y, thr_fit_s, max_fpr=0.05):
    """threshold chosen on the TUNING half for <= max_fpr, applied to the TEST half"""
    fpr, _, th = roc_curve(thr_fit_y, thr_fit_s)
    t = th[np.where(fpr <= max_fpr)[0][-1]]
    caught = ((s > t) & (y == 1)).sum() / (y == 1).sum(); fa = ((s > t) & (y == 0)).sum() / (y == 0).sum()
    return caught, fa

res = {k: [] for k in ["TruFor alone", "AI detector alone", "max(TruFor, AI)", "Combined (logistic)"]}
for seed in range(5):
    d = df.sample(frac=1, random_state=seed)
    a, b = d.iloc[:len(d) // 2], d.iloc[len(d) // 2:]
    lr = LogisticRegression().fit(a[["score", "prob_ai"]], a.label)
    for name, sa, sb in [("TruFor alone", a.score, b.score), ("AI detector alone", a.prob_ai, b.prob_ai),
                         ("max(TruFor, AI)", np.maximum(a.score, a.prob_ai), np.maximum(b.score, b.prob_ai)),
                         ("Combined (logistic)", lr.predict_proba(a[["score", "prob_ai"]])[:, 1],
                          lr.predict_proba(b[["score", "prob_ai"]])[:, 1])]:
        c, f = at_fpr(b.label.values, np.asarray(sb), a.label.values, np.asarray(sa))
        res[name].append((roc_auc_score(b.label, sb), c, f))

print("CocoGlide, held-out half, mean of 5 random splits (threshold set on the other half for <=5% false alarms)")
print(f"{'method':24s} {'AUC':>6s} {'edits caught':>13s} {'real flagged':>13s}")
summary = {"benchmark": "Separate vs combined on CocoGlide (held-out half, 5 splits, threshold for <=5% FPR on tuning half)"}
for k, v in res.items():
    v = np.array(v)
    print(f"{k:24s} {v[:, 0].mean():6.3f} {v[:, 1].mean() * 100:12.1f}% {v[:, 2].mean() * 100:12.1f}%")
    summary[k] = f"AUC {v[:, 0].mean():.3f}, edits caught {v[:, 1].mean() * 100:.1f}%, real flagged {v[:, 2].mean() * 100:.1f}%"
lr = LogisticRegression().fit(df[["score", "prob_ai"]], df.label)
summary["combiner_weights (TruFor, AI, bias)"] = [round(float(x), 3) for x in list(lr.coef_[0]) + [lr.intercept_[0]]]
print("combiner weights (fit on all, for reference):", summary["combiner_weights (TruFor, AI, bias)"])
from experiments.bench_record import save
save(summary, OUT, "combined_cocoglide_summary.json")
