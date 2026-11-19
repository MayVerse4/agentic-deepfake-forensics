"""Separate vs combined on CocoGlide for every tool scored so far (TruFor, Mesorch, AI detector).
Same protocol as eval_combined.py: held-out half, 5 random splits, threshold picked on the other half
for <= 5% false alarms. Every single tool and every combination of 2+ tools (logistic regression).
Run: .venv-win\\Scripts\\python.exe experiments\\eval_combined_all.py
"""
import itertools, os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
OUT = os.path.join(os.path.dirname(HERE), "results", "benchmarks")
df = pd.read_csv(os.path.join(OUT, "localization_eval.csv"))[["file", "label", "score"]].rename(columns={"score": "TruFor"})
df = df.merge(pd.read_csv(os.path.join(OUT, "cocoglide_ai_scores.csv")).rename(columns={"prob_ai": "AI detector"}), on="file")
df = df.merge(pd.read_csv(os.path.join(OUT, "mesorch_cocoglide_eval.csv"))[["file", "score"]].rename(columns={"score": "Mesorch"}), on="file")
TOOLS = ["TruFor", "Mesorch", "AI detector"]

def at_fpr(y, s, ty, ts, max_fpr=0.05):
    fpr, _, th = roc_curve(ty, ts)
    t = th[np.where(fpr <= max_fpr)[0][-1]]
    return ((s > t) & (y == 1)).sum() / (y == 1).sum(), ((s > t) & (y == 0)).sum() / (y == 0).sum()

combos = [(t,) for t in TOOLS] + [c for n in (2, 3) for c in itertools.combinations(TOOLS, n)]
res = {c: [] for c in combos}
for seed in range(5):
    d = df.sample(frac=1, random_state=seed)
    a, b = d.iloc[:len(d) // 2], d.iloc[len(d) // 2:]
    for c in combos:
        cols = list(c)
        if len(c) == 1:
            sa, sb = a[cols[0]].values, b[cols[0]].values
        else:
            lr = LogisticRegression().fit(a[cols], a.label)
            sa, sb = lr.predict_proba(a[cols])[:, 1], lr.predict_proba(b[cols])[:, 1]
        caught, fa = at_fpr(b.label.values, sb, a.label.values, sa)
        res[c].append((roc_auc_score(b.label, sb), caught, fa))

print("CocoGlide, held-out half, mean of 5 splits (threshold for <=5% false alarms set on the other half)")
print(f"{'tools':34s} {'AUC':>6s} {'edits caught':>13s} {'real flagged':>13s}")
summary = {"benchmark": "All tools separate vs combined on CocoGlide (held-out half, 5 splits, <=5% FPR threshold)"}
for c, v in res.items():
    v = np.array(v); name = " + ".join(c) if len(c) > 1 else c[0] + " alone"
    print(f"{name:34s} {v[:, 0].mean():6.3f} {v[:, 1].mean() * 100:12.1f}% {v[:, 2].mean() * 100:12.1f}%")
    summary[name] = f"AUC {v[:, 0].mean():.3f}, edits caught {v[:, 1].mean() * 100:.1f}%, real flagged {v[:, 2].mean() * 100:.1f}%"
lr = LogisticRegression().fit(df[TOOLS], df.label)
summary["weights (TruFor, Mesorch, AI, bias)"] = [round(float(x), 3) for x in list(lr.coef_[0]) + [lr.intercept_[0]]]
print("3-tool combiner weights:", summary["weights (TruFor, Mesorch, AI, bias)"])
from experiments.bench_record import save
save(summary, OUT, "combined_all_cocoglide_summary.json")
