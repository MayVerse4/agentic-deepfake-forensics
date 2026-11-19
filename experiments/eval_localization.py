"""Evaluates TruFor (detectors/localization.py) on CocoGlide (Guillaro et al., CVPR 2023):
512 real COCO images + 512 with a region inpainted by GLIDE (AI), with ground-truth masks.
Downloaded to data/benchmarks/ from https://www.grip.unina.it/download/prog/TruFor/CocoGlide.zip

Run: .venv-win\\Scripts\\python.exe experiments\\eval_localization.py
Reports image-level detection (AUC, accuracy at 0.5) and pixel-level localization
(F1 at 0.5 and best-threshold F1 per image, IoU), and writes results/benchmarks/localization_eval.csv
"""
import glob, os, re, sys
import cv2, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from detectors.localization import Localizer

D = os.path.join(os.path.dirname(HERE), "data", "benchmarks")
OUT = os.path.join(os.path.dirname(HERE), "results", "benchmarks"); os.makedirs(OUT, exist_ok=True)
masks = {re.search(r"_(\d+)_mask", os.path.basename(m)).group(1): m for m in glob.glob(os.path.join(D, "mask", "*.png"))}

def f1_iou(pred, gt):
    tp = np.logical_and(pred, gt).sum(); fp = np.logical_and(pred, ~gt).sum(); fn = np.logical_and(~pred, gt).sum()
    f1 = 2 * tp / max(2 * tp + fp + fn, 1); iou = tp / max(tp + fp + fn, 1)
    return f1, iou

L = Localizer()
rows = []
for label, folder in [(0, "real"), (1, "fake")]:
    files = sorted(glob.glob(os.path.join(D, folder, "*.png")))
    for i, f in enumerate(files, 1):
        r = L.analyze(f)
        row = {"file": os.path.basename(f), "label": label, "score": r["score"], "reliability": r["reliability"]}
        if label == 1:
            cid = re.search(r"val2017_(\d+)_up", os.path.basename(f)).group(1)
            gt = cv2.imread(masks[cid], cv2.IMREAD_GRAYSCALE)
            heat = r["_heat"]
            gt = cv2.resize(gt, (heat.shape[1], heat.shape[0]), interpolation=cv2.INTER_NEAREST) > 127
            row["f1_05"], row["iou_05"] = f1_iou(heat > 0.5, gt)
            row["f1_best"] = max(f1_iou(heat > t, gt)[0] for t in np.linspace(0.05, 0.95, 19))
        rows.append(row)
        if i % 128 == 0:
            print(f"  {folder}: {i}/{len(files)}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "localization_eval.csv"), index=False)
y, s = df.label.values, df.score.values
fk = df[df.label == 1]
print(f"\nCocoGlide: {int((y == 0).sum())} real + {int((y == 1).sum())} AI-inpainted images")
print(f"Image-level detection: AUC {roc_auc_score(y, s):.3f}   accuracy@0.5 {accuracy_score(y, s > 0.5) * 100:.1f}%   "
      f"edited caught {((s > 0.5) & (y == 1)).sum() / (y == 1).sum() * 100:.1f}%   real flagged {((s > 0.5) & (y == 0)).sum() / (y == 0).sum() * 100:.1f}%")
print(f"Localization (edited images): F1@0.5 {fk.f1_05.mean():.3f}   best-threshold F1 {fk.f1_best.mean():.3f}   IoU@0.5 {fk.iou_05.mean():.3f}")

SUMMARY_NAME = "localization_cocoglide_summary.json"
summary = {"benchmark": "TruFor localization on CocoGlide (512 real + 512 GLIDE-inpainted)",
           "n_real": int((y == 0).sum()), "n_edited": int((y == 1).sum()),
           "image_auc": round(float(roc_auc_score(y, s)), 4),
           "image_accuracy_at_0.5": round(float(accuracy_score(y, s > 0.5)), 4),
           "edited_caught_pct": round(float(((s > 0.5) & (y == 1)).sum() / (y == 1).sum() * 100), 1),
           "real_flagged_pct": round(float(((s > 0.5) & (y == 0)).sum() / (y == 0).sum() * 100), 1),
           "pixel_f1_at_0.5": round(float(fk.f1_05.mean()), 4), "pixel_f1_best_threshold": round(float(fk.f1_best.mean()), 4),
           "pixel_iou_at_0.5": round(float(fk.iou_05.mean()), 4), "per_image_csv": "localization_eval.csv"}


from experiments.bench_record import save
save(summary, OUT, SUMMARY_NAME)
