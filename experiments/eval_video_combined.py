"""Videos: NSG-VD (physics) vs the whole-image AI detector (Community Forensics, median over 8 frames)
vs both combined, on the same GenVideo test videos as eval_nsgvd.py.
Same protocol as the image combiner: held-out half, 5 random splits, threshold for <=5% false alarms
picked on the other half. Run after eval_nsgvd.py:
  .venv-win\\Scripts\\python.exe experiments\\eval_video_combined.py [nsgvd_csv_name]
"""
import os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
OUT = os.path.join(os.path.dirname(HERE), "results", "benchmarks")
VID = os.path.join(os.path.dirname(HERE), "data", "benchmarks", os.environ.get("GENVIDEO_DIR", "genvideo"))
TAG = "" if os.path.basename(VID) == "genvideo" else "_" + os.path.basename(VID)
NAME = sys.argv[1] if len(sys.argv) > 1 else f"nsgvd_standard-Pika-d{TAG}_eval.csv"
df = pd.read_csv(os.path.join(OUT, NAME))

cache = os.path.join(OUT, f"genvideo{TAG}_ai_scores.csv")
done = pd.read_csv(cache) if os.path.exists(cache) else pd.DataFrame(columns=["id", "prob_ai"])
todo = df[~df.id.isin(done.id)]
if len(todo):
    from detectors.ai_image_detector import AIImageDetector
    det = AIImageDetector(); rows = []
    for i, vid in enumerate(todo.id, 1):
        rows.append({"id": vid, "prob_ai": det.score_video(os.path.join(VID, vid + ".mp4"), n_frames=8)["prob_ai"]})
        if i % 100 == 0:
            print(f"  AI detector {i}/{len(todo)}", flush=True)
            pd.concat([done, pd.DataFrame(rows)]).to_csv(cache, index=False)
    done = pd.concat([done, pd.DataFrame(rows)]); done.to_csv(cache, index=False)
df = df.merge(done, on="id")
df["log_nsg"] = np.log1p(df.nsg_mmd.clip(lower=0))           # MMD is heavy-tailed; log keeps the combiner stable

def at_fpr(y, s, ty, ts, max_fpr=0.05):
    fpr, _, th = roc_curve(ty, ts); t = th[np.where(fpr <= max_fpr)[0][-1]]
    return ((s > t) & (y == 1)).sum() / (y == 1).sum(), ((s > t) & (y == 0)).sum() / (y == 0).sum()

methods = {"NSG-VD alone (physics)": ["log_nsg"], "AI detector alone": ["prob_ai"], "NSG-VD + AI detector": ["log_nsg", "prob_ai"]}
res = {k: [] for k in methods}; per = {k: {} for k in methods}
for seed in range(5):
    d = df.sample(frac=1, random_state=seed); a, b = d.iloc[:len(d) // 2], d.iloc[len(d) // 2:]
    for k, cols in methods.items():
        if len(cols) == 1:
            sa, sb = a[cols[0]].values, b[cols[0]].values
        else:
            lr = LogisticRegression().fit(a[cols], a.label)
            sa, sb = lr.predict_proba(a[cols])[:, 1], lr.predict_proba(b[cols])[:, 1]
        c, f = at_fpr(b.label.values, sb, a.label.values, sa)
        res[k].append((roc_auc_score(b.label, sb), c, f))
        fpr, _, th = roc_curve(a.label, sa); t = th[np.where(fpr <= 0.05)[0][-1]]
        for g, gb in b[b.label == 1].assign(s=sb[b.label.values == 1]).groupby("generator"):
            per[k].setdefault(g, []).append((gb.s > t).mean())

print(f"GenVideo shard 0: {int((df.label == 0).sum())} real vs {int(df.label.sum())} AI videos; held-out half, 5 splits, <=5% false alarms")
print(f"{'method':26s} {'AUC':>6s} {'AI caught':>10s} {'real flagged':>13s}")
summary = {"benchmark": f"Videos: NSG-VD vs AI detector vs combined on GenVideo shard 0 (held-out half, 5 splits, <=5% FPR){TAG}"}
for k, v in res.items():
    v = np.array(v)
    print(f"{k:26s} {v[:, 0].mean():6.3f} {v[:, 1].mean() * 100:9.1f}% {v[:, 2].mean() * 100:12.1f}%")
    summary[k] = f"AUC {v[:, 0].mean():.3f}, AI caught {v[:, 1].mean() * 100:.1f}%, real flagged {v[:, 2].mean() * 100:.1f}%"
gens = sorted(per["NSG-VD alone (physics)"])
print(f"\nAI videos caught per generator:\n{'generator':16s}" + "".join(f"{k.split(' (')[0]:>22s}" for k in methods))
for g in gens:
    print(f"{g:16s}" + "".join(f"{np.mean(per[k][g]) * 100:21.1f}%" for k in methods))
summary["per_generator_caught_pct (NSG, AI, combined)"] = {g: [round(float(np.mean(per[k][g]) * 100), 1) for k in methods] for g in gens}
lr = LogisticRegression().fit(df[["log_nsg", "prob_ai"]], df.label)
summary["combiner weights (log NSG, AI, bias)"] = [round(float(x), 3) for x in list(lr.coef_[0]) + [lr.intercept_[0]]]
from experiments.bench_record import save
save(summary, OUT, f"video_combined{TAG}_summary.json")
