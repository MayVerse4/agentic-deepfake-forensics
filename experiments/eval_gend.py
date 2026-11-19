"""Face models vs GenD (WACV 2026) on the same media. Resumable.
  frames : random 1500 real + 1500 fake face crops from Dataset/Test (held out from Spatial/Frequency training);
           Spatial + Frequency through the unchanged InferenceEngine, GenD on the same crop.
  omnifake / omnifake_av : OmniFake-OOD videos; full engine (Spatial/Frequency/Temporal/Fusion) + GenD averaged over
           the engine's own face crops of the main track (crop margin = the engine's, not GenD's 1.3x RetinaFace crop).
Run from the project root:  .venv-win\\Scripts\\python.exe experiments\\eval_gend.py frames|omnifake|omnifake_av
Writes results/benchmarks/gend_<mode>_scores.csv
"""
import glob, os, random, sys
os.environ.setdefault("REASONING_LLM", "none")
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "backend")
import cv2, pandas as pd
from PIL import Image

MODE = sys.argv[1]
OUT = f"results/benchmarks/gend_{MODE}_scores.csv"
if MODE == "frames":
    random.seed(0)
    items = [(p, 0) for p in random.sample(sorted(glob.glob("data/Dataset/Test/Real/*.jpg")), 1500)] + \
            [(p, 1) for p in random.sample(sorted(glob.glob("data/Dataset/Test/Fake/*.jpg")), 1500)]
else:
    D = {"omnifake": "data/benchmarks/omnifake_ood/video", "omnifake_av": "data/benchmarks/omnifake_ood/audiovisual"}[MODE]
    lab = pd.read_csv(os.path.join(D, "labels.csv"))
    items = [(os.path.join(D, f), l) for f, l in zip(lab.file, lab.label)]

done = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=["path"])
have, rows = set(done.path), done.to_dict("records")
from agent.inference_engine import InferenceEngine
from detectors.gend_detector import GenDDetector
eng, gd = InferenceEngine(), GenDDetector()
if MODE == "omnifake_av":
    eng.video_processor.max_frames = 64   # memory cap for this run (app default 500)

crops = {}
_orig = eng._build_track_sequences
def _grab(frame_data):
    crops.clear()
    for fd in frame_data:
        for f in fd.get("faces", []):
            if f.get("track_id") is not None:
                crops.setdefault(f["track_id"], []).append(f["face"])
    return _orig(frame_data)
eng._build_track_sequences = _grab

for i, (p, y) in enumerate(items, 1):
    if p in have:
        continue
    rec = dict(path=p, label=y, face=0)
    try:
        if MODE == "frames":
            rec["gend"] = gd.predict([Image.open(p).convert("RGB")])[0]
        out = eng.analyze(p)
        tr = out.get("per_track_summary") or []
        if tr:
            t = max(tr, key=lambda t: t["frames_seen"])
            rec.update(face=1, spatial=t["spatial_prob_fake"], frequency=t["frequency_prob_fake"],
                       temporal=t["temporal_prob_fake"], fusion=t["combined_prob_fake"])
            if MODE != "frames":
                c = crops.get(t.get("track_id")) or max(crops.values(), key=len, default=[])
                if c:
                    s = gd.predict([Image.fromarray(cv2.cvtColor(x, cv2.COLOR_BGR2RGB)) for x in c[::2][:16]])
                    rec["gend"] = sum(s) / len(s)
    except Exception as e:
        rec["error"] = str(e)[:120]
    rows.append(rec)
    if i % 5 == 0 or i == len(items):
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(i, len(items), flush=True)
pd.DataFrame(rows).to_csv(OUT, index=False)
print("done", len(rows))
