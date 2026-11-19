"""Evaluates a newer localization model on CocoGlide with exactly the same metrics as
eval_localization.py (TruFor), and prints it side by side with TruFor's saved results.

Run: .venv-win\\Scripts\\python.exe experiments\\eval_localizer_cocoglide.py mesorch
Writes results/benchmarks/<name>_cocoglide_eval.csv and <name>_cocoglide_summary.json (+ BENCHMARKS.md)
"""
import glob, os, re, sys
import cv2, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
NAME = sys.argv[1] if len(sys.argv) > 1 else "mesorch"
D = os.path.join(os.path.dirname(HERE), "data", "benchmarks")
OUT = os.path.join(os.path.dirname(HERE), "results", "benchmarks")

if NAME == "mesorch":
    from detectors.mesorch_localizer import MesorchLocalizer as Model
elif NAME == "safire":
    from detectors.safire_localizer import SafireLocalizer as Model
else:
    sys.exit(f"unknown model {NAME}")

masks = {re.search(r"_(\d+)_mask", os.path.basename(m)).group(1): m for m in glob.glob(os.path.join(D, "mask", "*.png"))}

def f1_iou(pred, gt):
    tp = np.logical_and(pred, gt).sum(); fp = np.logical_and(pred, ~gt).sum(); fn = np.logical_and(~pred, gt).sum()
    return 2 * tp / max(2 * tp + fp + fn, 1), tp / max(tp + fp + fn, 1)

M = Model()
rows = []
for label, folder in [(0, "real"), (1, "fake")]:
    files = sorted(glob.glob(os.path.join(D, folder, "*.png")))
    for i, f in enumerate(files, 1):
        r = M.analyze(f)
        row = {"file": os.path.basename(f), "label": label, "score": r["score"]}
        if label == 1:
            cid = re.search(r"val2017_(\d+)_up", os.path.basename(f)).group(1)
            heat = r["heat"]
            gt = cv2.resize(cv2.imread(masks[cid], cv2.IMREAD_GRAYSCALE), (heat.shape[1], heat.shape[0]),
                            interpolation=cv2.INTER_NEAREST) > 127
            row["f1_05"], row["iou_05"] = f1_iou(heat > 0.5, gt)
            row["f1_best"] = max(f1_iou(heat > t, gt)[0] for t in np.linspace(0.05, 0.95, 19))
        rows.append(row)
        if i % 128 == 0:
            print(f"  {folder}: {i}/{len(files)}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, f"{NAME}_cocoglide_eval.csv"), index=False)

def metrics(d):
    y, s = d.label.values, d.score.values; fk = d[d.label == 1]
    return {"image_auc": round(float(roc_auc_score(y, s)), 4),
            "edited_caught_pct": round(float(((s > 0.5) & (y == 1)).sum() / (y == 1).sum() * 100), 1),
            "real_flagged_pct": round(float(((s > 0.5) & (y == 0)).sum() / (y == 0).sum() * 100), 1),
            "pixel_f1_at_0.5": round(float(fk.f1_05.mean()), 4), "pixel_f1_best_threshold": round(float(fk.f1_best.mean()), 4),
            "pixel_iou_at_0.5": round(float(fk.iou_05.mean()), 4)}

new = metrics(df)
tru = metrics(pd.read_csv(os.path.join(OUT, "localization_eval.csv")))
print(f"\nCocoGlide (512 real + 512 GLIDE-inpainted), same script for both models")
print(f"{'metric':26s} {'TruFor':>9s} {NAME:>9s}")
for k in new:
    print(f"{k:26s} {tru[k]:9.3f} {new[k]:9.3f}" if "pct" not in k else f"{k:26s} {tru[k]:8.1f}% {new[k]:8.1f}%")

from experiments.bench_record import save
save({"benchmark": f"{NAME} localization on CocoGlide (512 real + 512 GLIDE-inpainted), same protocol as TruFor",
      **new, "trufor_same_protocol": tru, "per_image_csv": f"{NAME}_cocoglide_eval.csv"},
     OUT, f"{NAME}_cocoglide_summary.json")
