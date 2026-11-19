"""Lip-sync / audio-visual deepfake test with AVH-Align (Smeu et al., CVPR 2025; CC BY-NC-SA 4.0; weights trained
unsupervised on real AV-Deepfake1M videos). Runs in ext/.venv-avh (old torch + fairseq python code).
Per video: re-encode to 25 fps -> dlib landmarks -> 96x96 mouth crop (authors' crop_patch) + 16 kHz wav ->
AV-HuBERT large (self_large_vox_433h) audio & visual features -> AVH-Align MLP -> score = logsumexp(-output) over
frames (authors' eval.py). Higher = more likely fake.
Run:  ..\\ext\\.venv-avh\\Scripts\\python.exe experiments\\run_avh_align.py <video_dir_with_labels.csv> <out_csv>
"""
import os, subprocess, sys, tempfile
import numpy as np
np.float = np.float64; np.int = np.int_
EXT = r"C:\PROJECTS\major\ext"
AVH = os.path.join(EXT, "AVH-Align"); HUB = os.path.join(EXT, "av_hubert", "avhubert")
sys.path[:0] = [HUB, AVH]   # avhubert utils.py must win over AVH-Align utils.py
import cv2, dlib, pandas as pd, torch
import imageio_ffmpeg
FF = imageio_ffmpeg.get_ffmpeg_exe()
_argv, sys.argv = sys.argv, sys.argv[:1]   # avhubert picks absolute imports only when argv has 1 entry
import deepfake_preprocess as P
import deepfake_feature_extraction as X
from model import FusionModel
sys.argv = _argv
from fairseq import checkpoint_utils as _cu
_up = _cu._upgrade_state_dict
def _up_slim(state):   # slim checkpoint keeps only args/cfg/model: add the bookkeeping fairseq expects (unused at inference)
    state.setdefault("optimizer_history", [{"criterion_name": "x", "optimizer_name": "x", "lr_scheduler_state": {"best": 0}, "num_updates": 0}])
    state.setdefault("extra_state", {"train_iterator": {"epoch": 0}})
    state.setdefault("last_optimizer_state", None)
    return _up(state)
_cu._upgrade_state_dict = _up_slim

VDIR, OUT = sys.argv[1], sys.argv[2]
lab = pd.read_csv(os.path.join(VDIR, "labels.csv"))
done = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=["file"])
have, rows = set(done.file), done.to_dict("records")

det, pred = dlib.get_frontal_face_detector(), dlib.shape_predictor(os.path.join(AVH, P.FACE_PREDICTOR_PATH))
mean_face = np.load(os.path.join(AVH, P.MEAN_FACE_PATH))
model, task = X.load_model(os.path.join(HUB, "self_large_vox_433h_slim.pt"))
tf = X.load_transforms(task)
fm = FusionModel().cuda().eval()
fm.load_state_dict(torch.load(os.path.join(AVH, "checkpoints", "AVH-Align_AV1M.pt"), weights_only=False)["state_dict"])
tmp = tempfile.mkdtemp()


def score_video(src):
    v25, wav, roi = (os.path.join(tmp, n) for n in ("v25.mp4", "a.wav", "roi.mp4"))
    subprocess.run([FF, "-y", "-loglevel", "quiet", "-i", src, "-r", "25", "-an", v25])
    subprocess.run([FF, "-y", "-loglevel", "quiet", "-i", src, "-vn", "-ac", "1", "-ar", "16000", wav])
    if not os.path.exists(wav) or os.path.getsize(wav) < 2000:
        return None, "no audio"
    cap = cv2.VideoCapture(v25); frames = []
    while True:
        ok, f = cap.read()
        if not ok: break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))   # skvideo returned RGB
    cap.release(); frames = np.array(frames)
    lms = [P.detect_landmark(f, det, pred) for f in frames]
    if all(l is None for l in lms):
        return None, "no face"
    lms = P.landmarks_interpolate(lms)
    rois = P.crop_patch(v25, lms, mean_face, P.STABLE_PNTS_IDS, P.STD_SIZE, window_margin=12, start_idx=48,
                        stop_idx=68, crop_height=96, crop_width=96)
    P.write_video_ffmpeg(rois, roi, FF)
    fa, fv, _ = X.extract_features(model, roi, wav, tf, False)
    with torch.no_grad():
        v = torch.from_numpy(fv).cuda(); a = torch.from_numpy(fa).cuda()
        v = v / torch.linalg.norm(v, dim=-1, keepdim=True); a = a / torch.linalg.norm(a, dim=-1, keepdim=True)
        return float(torch.logsumexp(-fm(v, a), dim=0).squeeze().cpu()), ""


for i, r in enumerate(lab.itertuples(), 1):
    if r.file in have:
        continue
    try:
        s, note = score_video(os.path.join(VDIR, r.file))
    except Exception as e:
        s, note = None, str(e)[:120]
    rows.append({"file": r.file, "label": r.label, "avh_score": s, "note": note})
    if i % 5 == 0 or i == len(lab):
        pd.DataFrame(rows).to_csv(OUT, index=False); print(i, len(lab), flush=True)
pd.DataFrame(rows).to_csv(OUT, index=False)
print("done", len(rows))
