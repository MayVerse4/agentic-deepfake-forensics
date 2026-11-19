"""
Whole-image AI-generation detector (no face needed), using Community Forensics
(Park & Owens, CVPR 2025; https://github.com/JeongsooP/Community-Forensics, MIT licence).
Trained on images from 4,803 different generators; ranked first of 23 open detectors in
the 2026 benchmark arXiv:2602.07814 (78% mean accuracy across 12 datasets).

    from detectors.ai_image_detector import AIImageDetector
    det = AIImageDetector()
    det.score_image("city.jpg")     # -> {"prob_ai": 0.93}
    det.score_video("clip.mp4")     # -> {"prob_ai": ..., "frame_probs": [...], "frames": n}

Works from pixels only, so stripped metadata does not matter.
"""
import contextlib
import os
import sys
import threading

import cv2
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "third_party", "commfor")
HF_MODEL = "OwensLab/commfor-model-384"
HF_PREPROC = "OwensLab/commfor-data-preprocessor"
_CLASH = ("models", "dataloader", "utils", "custom_transforms", "custom_sampler", "dataprocessor_hf")
_MODS = {}


@contextlib.contextmanager
def _isolated():
    """Community Forensics imports top-level modules named models/dataloader/utils, which the
    project also has. Swap them in only while this code runs."""
    clash = lambda k: k in _CLASH or k.startswith("models.") or k.startswith("utils.")
    saved = {k: v for k, v in sys.modules.items() if clash(k)}
    for k in saved:
        del sys.modules[k]
    sys.modules.update(_MODS)
    if "wandb" not in sys.modules:          # only used for training logs; not installed (protobuf clash with TF)
        import importlib.machinery, types
        stub = types.ModuleType("wandb")
        # a real ModuleSpec: libraries such as accelerate call importlib.util.find_spec("wandb"),
        # which raises on a module whose __spec__ is None
        stub.__spec__ = importlib.machinery.ModuleSpec("wandb", None)
        sys.modules["wandb"] = stub
    sys.path.insert(0, SRC)
    try:
        yield
    finally:
        sys.path.remove(SRC)
        for k in [k for k in sys.modules if clash(k)]:
            _MODS[k] = sys.modules.pop(k)
        sys.modules.update(saved)


class AIImageDetector:
    _lock = threading.Lock()

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        with _isolated():
            import models
            import dataprocessor_hf as dphf
            self.proc = dphf.CommForImageProcessor.from_pretrained(HF_PREPROC, size=384)
            self.model = models.ViTClassifier.from_pretrained(HF_MODEL, device=self.device).to(self.device).eval()

    @torch.no_grad()
    def _probs(self, pil_images):
        with self._lock, _isolated():
            x = self.proc(pil_images, mode="test")["pixel_values"].to(self.device)
            if x.ndim == 3:                      # processor drops the batch dim for a single image
                x = x[None]
            return torch.sigmoid(self.model(x)).flatten().float().cpu().numpy()

    def score_image(self, path_or_pil):
        img = Image.open(path_or_pil) if isinstance(path_or_pil, str) else path_or_pil
        return {"prob_ai": round(float(self._probs([img.convert("RGB")])[0]), 4)}

    def score_video(self, path, n_frames=8):
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        frames = []
        for idx in np.linspace(0, total - 1, min(n_frames, total)).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, f = cap.read()
            if ok:
                frames.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
        cap.release()
        if not frames:
            return None
        p = self._probs(frames)
        # median is robust to a few odd frames (scene cuts, black frames)
        return {"prob_ai": round(float(np.median(p)), 4), "frame_probs": [round(float(v), 3) for v in p],
                "frames": len(frames)}


if __name__ == "__main__":
    d = AIImageDetector()
    for p in sys.argv[1:]:
        r = d.score_video(p) if p.lower().endswith((".mp4", ".mov", ".avi", ".mkv")) else d.score_image(p)
        print(f"{os.path.basename(p):32s} P(AI-generated) = {r['prob_ai']:.3f}")
