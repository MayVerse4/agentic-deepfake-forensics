"""Fast comparison of every temporal and fusion checkpoint (deployed + archive).
Temporal: the 46 local FaceForensics++ videos (data/test_media/ff_videos), app pipeline (decode -> detect -> track -> crop),
16-crop windows, video score = mean window probability -> AUC. Face crops are cached once.
Fusion: new_val_fusion_data_3branch.csv (86 held-out videos with saved branch scores) -> AUC/accuracy.
Output: results/retrain/temporal_fusion_selection.json"""
import os, sys, glob, json, pickle
import numpy as np, torch, pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")); sys.path.insert(0, REPO); os.chdir(REPO)
from agent.inference_engine import InferenceEngine, CHECKPOINT_DIR
from temporal_model import TemporalModel
from fusion_model import FusionModel
OUT = os.path.join(REPO, "results", "retrain"); cache = os.path.join(OUT, "ffvid_crops.pkl")
res = {"temporal": [], "fusion": []}
# ---- fusion (cheap) ----
v = pd.read_csv("new_val_fusion_data_3branch.csv")
for p in ["weights/trained/fusion_model_4input.pth"] + sorted(glob.glob("archive/**/fusion*.pth", recursive=True)):
    try:
        m = FusionModel.load(p, "cpu"); X = torch.tensor([FusionModel.build_input(r.spatial, r.frequency, r.temporal, use_frequency=m.use_frequency) for r in v.itertuples()], dtype=torch.float32)
        with torch.no_grad(): s = torch.sigmoid(m(X)).view(-1).numpy()
        r = dict(ckpt=p, inputs=4 if m.use_frequency else 3, auc=round(roc_auc_score(v.label, s), 4), acc=round(accuracy_score(v.label, s > 0.5), 4))
    except Exception as e: r = dict(ckpt=p, error=str(e)[:150])
    print(r, flush=True); res["fusion"].append(r)
# ---- temporal ----
eng = InferenceEngine()
if os.path.exists(cache): vids = pickle.load(open(cache, "rb"))
else:
    vids = []
    for f in sorted(glob.glob("data/test_media/ff_videos/*/*")):
        y = 0 if os.sep + "Real" + os.sep in f or "/Real/" in f else 1
        raw, _ = eng._load_frames(f); fd = eng._build_frame_data(raw, eng.frame_processor.process_frames(raw))
        fd, _ = eng.quality_checker.process_frames(fd); fd = eng.face_detector.detect_faces(fd)
        from preprocessing.face_tracker import FaceTracker
        fd = FaceTracker().track_faces(fd); fd = eng.face_cropper.crop_faces(fd)
        seq = eng._build_track_sequences(fd)
        if not seq: print("no face", f); continue
        tid = max(seq, key=lambda t: len(seq[t])); vids.append((f, y, seq[tid])); print("cropped", f, len(seq[tid]), flush=True)
    pickle.dump(vids, open(cache, "wb"))
cands = ["weights/trained/temporal_model_best_phase1.pth"] + sorted(glob.glob("archive/**/temporal*.pth", recursive=True))
for p in cands:
    r = dict(ckpt=p)
    try:
        sd = torch.load(p, map_location="cpu"); sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
        m = None
        for ua in (True, False):
            try: m = TemporalModel(freeze_backbone=False, use_anomaly=ua); m.load_state_dict(sd); break
            except Exception: m = None
        if m is None: raise RuntimeError("state dict does not match TemporalModel")
        m = m.to(eng.device).eval(); ys, ss = [], []
        for f, y, wins in vids:
            sc = []
            for crops in wins:
                t = eng._crops_to_tensor(crops)
                with torch.no_grad(): lg, _ = m(t, eng._compute_temporal_frequency_gpu(t) if m.use_anomaly else None)
                sc.append(torch.sigmoid(lg).item())
            ys.append(y); ss.append(np.mean(sc))
        ss = np.array(ss); r.update(auc=round(roc_auc_score(ys, ss), 4), acc=round(accuracy_score(ys, ss > 0.5), 4), n=len(ys))
    except Exception as e: r["error"] = str(e)[:150]
    print(r, flush=True); res["temporal"].append(r); torch.cuda.empty_cache()
json.dump(res, open(os.path.join(OUT, "temporal_fusion_selection.json"), "w"), indent=1)
