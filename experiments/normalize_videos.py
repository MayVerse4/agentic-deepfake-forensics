"""Removes the format shortcut in the GenVideo shard (all real clips are 640x360 @ 24 fps, each generator
has its own size/fps): every clip, real and AI, is centre-cropped to a square, resized to 256x256,
and re-encoded with the same codec (mp4v) at 8 fps. Writes data/benchmarks/genvideo_norm/ (+ the .json labels).
Run: .venv-win\\Scripts\\python.exe experiments\\normalize_videos.py
"""
import glob, os, shutil
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
SRC, DST, S = os.path.join(os.path.dirname(HERE), "data", "benchmarks", "genvideo"), os.path.join(os.path.dirname(HERE), "data", "benchmarks", "genvideo_norm"), 256
os.makedirs(DST, exist_ok=True)
ok = bad = 0
for p in sorted(glob.glob(os.path.join(SRC, "*.mp4"))):
    out = os.path.join(DST, os.path.basename(p))
    cap = cv2.VideoCapture(p); frames = []
    while len(frames) < 64:
        r, f = cap.read()
        if not r:
            break
        h, w = f.shape[:2]; m = min(h, w)
        f = f[(h - m) // 2:(h - m) // 2 + m, (w - m) // 2:(w - m) // 2 + m]
        frames.append(cv2.resize(f, (S, S), interpolation=cv2.INTER_AREA))
    cap.release()
    if len(frames) < 8:
        bad += 1; continue
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 8, (S, S))
    for f in frames:
        vw.write(f)
    vw.release()
    shutil.copy(p[:-4] + ".json", out[:-4] + ".json"); ok += 1
print(f"normalized {ok} clips, skipped {bad} unreadable/too short")
