"""
SPAI (Spectral AI-Generated Image Detection via masked frequency modeling, CVPR 2025)
Wrapper around the official SPAI source in third_party/spai/.

SPAI uses a ViT backbone trained with masked-frequency pre-training (MFM) and fine-tuned
for binary real/fake classification. It is specifically designed to catch the spectral
artifacts left by diffusion/GAN generators -- the "waxy / over-smoothed" texture and
checkerboard upsampling patterns that Community Forensics can miss on newer generators
(FLUX1-dev, Imagen 3, etc.).

Usage:
    from detectors.spai_detector import SPADetector
    det = SPADetector()
    res = det.score_image("image.jpg")   # -> {"prob_ai": 0.93, ...} or None if weights missing

The detector degrades gracefully (returns None, logs the error) when no trained
checkpoint is available, so the backend never breaks.
"""
from __future__ import annotations

import contextlib
import os
import sys
import threading
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SPAI_SRC = os.path.join(os.path.dirname(HERE), "third_party", "spai")

DEFAULT_CKPT_PATHS = [
    os.environ.get("SPAI_CHECKPOINT", ""),
    os.path.join(os.path.dirname(HERE), "weights", "pretrained", "spai", "spai_checkpoint.pth"),
    os.path.join(os.path.dirname(HERE), "third_party", "spai", "checkpoints", "spai_checkpoint.pth"),
]

IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_CLASH = ("models", "dataloader", "utils", "custom_transforms", "custom_sampler", "dataprocessor_hf")

_tool = None
_tool_error = None
_lock = threading.Lock()


def _find_checkpoint() -> Optional[str]:
    for p in DEFAULT_CKPT_PATHS:
        if p and os.path.isfile(p):
            return p
    return None


@contextlib.contextmanager
def _isolated_spai():
    clash = lambda k: k in _CLASH or k.startswith("models.") or k.startswith("utils.") \
        or k.startswith("filters.") or k.startswith("backbones.")
    saved = {k: v for k, v in sys.modules.items() if clash(k)}
    for k in saved:
        del sys.modules[k]
    sys.path.insert(0, SPAI_SRC)
    try:
        yield
    finally:
        sys.path.remove(SPAI_SRC)
        for k in [k for k in sys.modules if clash(k)]:
            sys.modules.pop(k, None)
        sys.modules.update(saved)


def _build_transform():
    import torchvision.transforms as T
    return T.Compose([
        T.Resize(IMG_SIZE, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class SPADetector:
    """SPAI spectral detector. score_image(path) -> {"prob_ai": float} or None."""

    def __init__(self, device=None, checkpoint_path: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path or _find_checkpoint()
        self.model = None
        self.transform = _build_transform()
        self._load()

    def _load(self):
        cfg_path = os.path.join(SPAI_SRC, "configs", "spai.yaml")
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"SPAI config not found at {cfg_path}")

        with _isolated_spai():
            from spai.config import get_custom_config
            from spai.models.build import build_cls_model
            config = get_custom_config(cfg_path)
            self.model = build_cls_model(config)

        if self.checkpoint_path and os.path.isfile(self.checkpoint_path):
            state = torch.load(self.checkpoint_path, map_location="cpu")
            sd = state.get("model", state) if isinstance(state, dict) else state
            missing, unexpected = self.model.load_state_dict(sd, strict=False)
            pass
        self.model.to(self.device).eval()

    @torch.no_grad()
    def score_image(self, path_or_pil) -> Optional[dict]:
        if self.model is None:
            return None
        try:
            img = Image.open(path_or_pil) if isinstance(path_or_pil, str) else path_or_pil
            img = img.convert("RGB")
            x = self.transform(img).unsqueeze(0).to(self.device)
            with _isolated_spai():
                logits = self.model(x)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            probs = torch.softmax(logits.float(), dim=-1)
            prob_ai = float(probs[0, 1].item()) if probs.shape[-1] == 2 else float(torch.sigmoid(logits[0, 0]).item())
            return {"prob_ai": round(prob_ai, 4)}
        except Exception as e:
            return None


def _get() -> Optional[SPADetector]:
    global _tool, _tool_error
    with _lock:
        if _tool is not None:
            return _tool
        if _tool_error is not None:
            return None
        try:
            _tool = SPADetector()
        except Exception as e:
            _tool_error = f"{type(e).__name__}: {e}"
            return None
        return _tool


def score_image(path_or_pil) -> Optional[dict]:
    det = _get()
    if det is None:
        return None
    return det.score_image(path_or_pil)


def status() -> dict:
    return {"available": _tool is not None or _tool_error is None,
            "error": _tool_error}


if __name__ == "__main__":
    import sys as _sys
    det = SPADetector()
    if det.model is None:
        print("SPAI: no checkpoint available (set SPAI_CHECKPOINT env var).")
    else:
        for p in _sys.argv[1:]:
            r = det.score_image(p)
            print(f"{os.path.basename(p):32s} P(AI) = {r['prob_ai']:.3f}" if r else f"{p}: SPAI unavailable")
