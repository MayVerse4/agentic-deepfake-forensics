"""
Manipulation localization for ANY image region (not only faces), using TruFor
(Guillaro et al., CVPR 2023; https://github.com/grip-unina/TruFor).

    from detectors.localization import Localizer
    res = Localizer().analyze("photo.jpg")
    res["score"]        # 0-1, probability the image is manipulated (global integrity)
    res["regions"]      # [{"box": [x, y, w, h], "area_pct": .., "mean_heat": ..}, ...]
    res["overlay"]      # data:image/jpeg;base64 -- heatmap blended over the image, boxes drawn
    res["heatmap"]      # data:image/jpeg;base64 -- raw localization map
    res["reliability"]  # mean of TruFor's confidence map (low = treat the heatmap with care)

TruFor works from pixels only (RGB + Noiseprint++ camera-noise fingerprint), so it is
unaffected by stripped metadata. Licence: non-profit / research use only (see trufor_src/).
"""
import base64
import contextlib
import os
import sys
import threading

import cv2
import numpy as np
import torch
from torch.nn import functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "third_party", "trufor", "test_docker", "src")
WEIGHTS = os.path.join(os.path.dirname(HERE), "weights", "pretrained", "trufor", "trufor.pth.tar")
MAX_SIDE = 1536          # keeps a 6 GB laptop GPU safe; larger images are downscaled
HEAT_THRESHOLD = 0.5     # pixel is "manipulated" above this
MIN_REGION_PCT = 0.15    # ignore specks smaller than this % of the image


_TRUFOR_MODS = {}


@contextlib.contextmanager
def _trufor_imports():
    """TruFor's code imports top-level modules named "config" and "models" (also lazily, inside
    forward()). The project has its own modules with those names. Swap TruFor's in for the
    duration of a call and restore the project's afterwards, so neither side breaks."""
    clash = lambda k: k in ("config", "models") or k.startswith("models.")
    saved = {k: v for k, v in sys.modules.items() if clash(k)}
    for k in saved:
        del sys.modules[k]
    sys.modules.update(_TRUFOR_MODS)
    sys.path.insert(0, SRC)
    try:
        yield
    finally:
        sys.path.remove(SRC)
        for k in [k for k in sys.modules if clash(k)]:
            _TRUFOR_MODS[k] = sys.modules.pop(k)
        sys.modules.update(saved)


def _b64(img, q=85):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode() if ok else None


class Localizer:
    _lock = threading.Lock()

    def __init__(self, device=None):
        with _trufor_imports():
            from config import _C as config               # noqa: E402  (TruFor's own yacs config)
            from models.cmx.builder_np_conf import myEncoderDecoder
            config.defrost()
            config.merge_from_file(os.path.join(SRC, "trufor.yaml"))
            config.TEST.MODEL_FILE = WEIGHTS
            config.freeze()
            self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
            ckpt = torch.load(WEIGHTS, map_location=self.device, weights_only=False)
            self.model = myEncoderDecoder(cfg=config)      # builds sub-modules -> needs TruFor's "models"
            self.model.load_state_dict(ckpt["state_dict"])
            self.model.to(self.device).eval()

    @torch.no_grad()
    def analyze(self, path_or_bgr, box_min=0.5):
        img = cv2.imread(path_or_bgr) if isinstance(path_or_bgr, str) else path_or_bgr
        if img is None:
            return None
        h0, w0 = img.shape[:2]
        scale = min(1.0, MAX_SIDE / max(h0, w0))
        work = cv2.resize(img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else img
        rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
        x = torch.tensor(rgb.transpose(2, 0, 1), dtype=torch.float)[None] / 256.0   # same as TruFor's loader

        with self._lock, _trufor_imports():
            try:
                pred, conf, det, _ = self.model(x.to(self.device))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                self.model.to("cpu"); pred, conf, det, _ = self.model(x); self.model.to(self.device)

        heat = F.softmax(pred[0], dim=0)[1].cpu().numpy()          # manipulation probability per pixel
        reli = torch.sigmoid(conf[0])[0].cpu().numpy() if conf is not None else None
        score = float(torch.sigmoid(det).item()) if det is not None else float(heat.max())
        heat = cv2.resize(heat, (w0, h0), interpolation=cv2.INTER_LINEAR)
        return {"score": round(score, 4), "reliability": round(float(reli.mean()), 3) if reli is not None else None,
                **render(img, heat, draw_boxes=score >= box_min)}


def render(img, heat, draw_boxes):
    """Heat map (per-pixel P(edited), image size) -> region boxes + overlay images.
    Shared by TruFor and Mesorch. Boxes are drawn only when the image-level decision says
    "manipulated"; weak heat on an authentic image would otherwise look like evidence."""
    h0, w0 = img.shape[:2]
    mask = (heat > HEAT_THRESHOLD).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    regions = []
    for i in range(1, n):
        x0, y0, w, h, area = stats[i]
        pct = 100.0 * area / (h0 * w0)
        if pct >= MIN_REGION_PCT:
            regions.append({"box": [int(x0), int(y0), int(w), int(h)], "area_pct": round(pct, 2),
                            "mean_heat": round(float(heat[lab == i].mean()), 3)})
    regions.sort(key=lambda r: -r["area_pct"])

    heat_vis = cv2.applyColorMap((np.clip(heat, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heat_vis, 0.4, 0)
    t = max(2, w0 // 300)
    for r in (regions[:5] if draw_boxes else []):
        x0, y0, w, h = r["box"]
        cv2.rectangle(overlay, (x0, y0), (x0 + w, y0 + h), (0, 0, 255), t)
    small = lambda im: im if w0 <= 900 else cv2.resize(im, (900, int(h0 * 900 / w0)))
    return {"manipulated_pct": round(100.0 * float(mask.mean()), 2), "regions": regions[:5],
            "overlay": _b64(small(overlay)), "heatmap": _b64(small(heat_vis)), "_heat": heat}


if __name__ == "__main__":
    loc = Localizer()
    for p in sys.argv[1:]:
        r = loc.analyze(p)
        print(f"{os.path.basename(p):28s} score={r['score']:.3f}  manipulated={r['manipulated_pct']:5.2f}%  "
              f"reliability={r['reliability']}  regions={[x['box'] for x in r['regions']]}")
