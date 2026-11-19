"""Evaluates the whole-image AI detector on a random sample of the Community Forensics
evaluation set (OwensLab/CommunityForensics-Eval, split "CompEval": 51,836 labelled images,
real photos + many generators). Rows are fetched through the Hugging Face datasets-server,
so the 5.8 GB set does not have to be downloaded.

Run: .venv-win\\Scripts\\python.exe experiments\\eval_ai_detector.py [n_chunks] [rows_per_chunk]
Writes results/benchmarks/ai_detector_eval.csv and prints overall + per-generator results.
"""
import base64, io, json, os, random, sys, time, urllib.request
import numpy as np, pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detectors.ai_image_detector import AIImageDetector

N_CHUNKS = int(sys.argv[1]) if len(sys.argv) > 1 else 70
PER_CHUNK = int(sys.argv[2]) if len(sys.argv) > 2 else 15
TOTAL = 51836
URL = "https://datasets-server.huggingface.co/rows?dataset=OwensLab/CommunityForensics-Eval&config=default&split=CompEval&offset={o}&length={n}"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "benchmarks")
os.makedirs(OUT, exist_ok=True)

def fetch(offset, n):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(URL.format(o=offset, n=n), timeout=60) as r:
                return json.load(r)["rows"]
        except Exception as e:
            time.sleep(3 * (attempt + 1))
    return []

def decode(v):
    if isinstance(v, dict) and "src" in v:                     # datasets-server may return an asset URL
        return Image.open(io.BytesIO(urllib.request.urlopen(v["src"], timeout=60).read()))
    s = v.strip('"') if isinstance(v, str) else v
    return Image.open(io.BytesIO(base64.b64decode(s)))

det = AIImageDetector()
random.seed(0)
offsets = sorted(random.sample(range(0, TOTAL - PER_CHUNK), N_CHUNKS))
rows = []
for i, off in enumerate(offsets, 1):
    for r in fetch(off, PER_CHUNK):
        row = r["row"]
        try:
            img = decode(row["image_data"]).convert("RGB")
        except Exception:
            continue
        p = det.score_image(img)["prob_ai"]
        rows.append({"model_name": row["model_name"], "architecture": row["architecture"],
                     "real_source": row.get("real_source"), "label": int(row["label"]), "prob_ai": p})
    if i % 10 == 0:
        print(f"  {i}/{N_CHUNKS} chunks, {len(rows)} images", flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "ai_detector_eval.csv"), index=False)
y, p = df.label.values, df.prob_ai.values
pred = (p > 0.5).astype(int)
tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
print(f"\nimages: {len(df)}  (real {int((y == 0).sum())}, AI {int((y == 1).sum())}); generators sampled: {df[df.label == 1].model_name.nunique()}")
print(f"accuracy {accuracy_score(y, pred) * 100:.1f}%   AUC {roc_auc_score(y, p):.3f}   "
      f"AI detected (recall) {tp / max(tp + fn, 1) * 100:.1f}%   real wrongly flagged (FPR) {fp / max(fp + tn, 1) * 100:.1f}%")
print("\nper architecture (AI images only, % detected):")
print(df[df.label == 1].groupby("architecture").prob_ai.agg(n="size", detected=lambda s: round((s > 0.5).mean() * 100, 1)).to_string())
print("\nper generator (AI images, % detected):")
print(df[df.label == 1].groupby("model_name").prob_ai.agg(n="size", detected=lambda s: round((s > 0.5).mean() * 100, 1))
      .sort_values("n", ascending=False).to_string())

SUMMARY_NAME = "ai_detector_commforeval_summary.json"
g = df[df.label == 1].groupby("model_name").prob_ai
summary = {"benchmark": f"Community Forensics AI detector on random sample of CommunityForensics-Eval/CompEval ({N_CHUNKS}x{PER_CHUNK} rows)",
           "n_images": int(len(df)), "n_real": int((y == 0).sum()), "n_ai": int((y == 1).sum()),
           "generators_sampled": int(df[df.label == 1].model_name.nunique()),
           "accuracy_at_0.5": round(float(accuracy_score(y, pred)), 4), "auc": round(float(roc_auc_score(y, p)), 4),
           "ai_detected_pct": round(tp / max(tp + fn, 1) * 100, 1), "real_flagged_pct": round(fp / max(fp + tn, 1) * 100, 1),
           "per_group": {k: (int(v.size), round(float((v > 0.5).mean() * 100), 1)) for k, v in g},
           "per_image_csv": "ai_detector_eval.csv"}


from experiments.bench_record import save
save(summary, OUT, SUMMARY_NAME)
