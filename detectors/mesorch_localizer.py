"""Mesorch (Zhu et al., AAAI 2025, MIT licence) edited-region localization.
Official code in third_party/mesorch (unchanged); weights in weights/pretrained/mesorch/ (Google Drive link in its README).
Trained on CASIA2, FantasticReality, IMD2020, tampCOCO, so CocoGlide (GLIDE inpainting) is unseen.

MesorchLocalizer().analyze(path_or_bgr) -> {"score", "heat"}
  heat  : per-pixel P(edited), at the original image size
  score : image-level score = mean of the top 1% pixel probabilities (Mesorch has no image head)
"""
import argparse, os, sys, types
import cv2, numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "third_party", "mesorch")
WEIGHTS = os.path.join(os.path.dirname(HERE), "weights", "pretrained", "mesorch", "mesorch-98.pth")
SIZE = 512
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def _load_class():
    # mesorch.py imports IMDLBenCo only for its model registry; a stub registry is enough for inference
    reg = types.ModuleType("IMDLBenCo.registry")
    reg.MODELS = types.SimpleNamespace(register_module=lambda *a, **k: (lambda c: c))
    sys.modules.setdefault("IMDLBenCo", types.ModuleType("IMDLBenCo"))
    sys.modules["IMDLBenCo.registry"] = reg
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "extractor" or k.startswith("extractor.")}
    sys.path.insert(0, SRC)
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("mesorch_model", os.path.join(SRC, "mesorch.py"))
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        return mod.MesorchFull
    finally:
        sys.path.remove(SRC)
        for k in [k for k in sys.modules if k == "extractor" or k.startswith("extractor.")]:
            sys.modules.pop(k)
        sys.modules.update(saved)


class MesorchLocalizer:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _load_class()(image_size=SIZE)
        with torch.serialization.safe_globals([argparse.Namespace]):   # checkpoint stores its training args
            ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
        missing, unexpected = self.model.load_state_dict(ckpt["model"], strict=False)
        assert not [k for k in missing if "loss" not in k], f"missing weights: {missing[:5]}"
        self.model.eval().to(self.device)

    @torch.no_grad()
    def analyze(self, img):
        bgr = cv2.imread(img) if isinstance(img, str) else img
        h, w = bgr.shape[:2]
        rgb = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(((rgb / 255.0 - MEAN) / STD).transpose(2, 0, 1)).float()[None].to(self.device)
        out = self.model(x, torch.zeros(1, 1, SIZE, SIZE, device=self.device))
        heat = cv2.resize(out["pred_mask"][0, 0].float().cpu().numpy(), (w, h), interpolation=cv2.INTER_LINEAR)
        flat = np.sort(heat.ravel())
        return {"score": float(flat[-max(1, flat.size // 100):].mean()), "heat": heat}
