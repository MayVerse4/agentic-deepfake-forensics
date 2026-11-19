"""GAPL (Generator-Aware Prototype Learning; Qin et al., CVPR 2026; MIT) whole-image AI-generation detector.
Follows the authors' gapl_src/inference_image.py exactly: GAPLModel(fe_path=None, proto_path=None,
freeze_backbone=False), weights ckpt['model'] + ckpt['prototype'] from the official checkpoint
(huggingface.co/AbyssLumine/GAPL, weights/pretrained/gapl/checkpoint.pt), CenterCrop(224) + ImageNet mean/std, sigmoid.

GAPLDetector().score_image(path_or_pil) -> {"prob_ai"}
gapl_src/models.py is loaded by file path (its module name "models" would clash with the project's), and its
hard-coded Linux CLIP cache path is redirected to the same model id, openai/clip-vit-large-patch14.
Every checkpoint tensor must load (no silent strict=False gaps).
"""
import importlib.util
import os
import threading

import torch
import torchvision.transforms as T
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "third_party", "gapl", "models.py")
WEIGHTS = os.path.join(os.path.dirname(HERE), "weights", "pretrained", "gapl", "checkpoint.pt")
CLIP_ID = "openai/clip-vit-large-patch14"


class GAPLDetector:
    _lock = threading.Lock()

    def __init__(self, device=None):
        import transformers
        spec = importlib.util.spec_from_file_location("gapl_models", SRC)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        orig = transformers.CLIPModel.from_pretrained.__func__
        mod.CLIPModel = type("CLIPModelRedirect", (transformers.CLIPModel,), {
            "from_pretrained": classmethod(lambda cls, *a, **k: orig(transformers.CLIPModel, CLIP_ID))})
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = mod.GAPLModel(fe_path=None, proto_path=None, freeze_backbone=False)
        ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if missing or unexpected:
            raise RuntimeError(f"GAPL checkpoint mismatch: {len(missing)} missing, {len(unexpected)} unexpected")
        model.load_prototype(ckpt["prototype"])
        self.model = model.to(self.device).eval()
        self.tf = T.Compose([T.CenterCrop(224), T.ToTensor(),
                             T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    @torch.no_grad()
    def score_image(self, img):
        im = (Image.open(img) if isinstance(img, str) else img).convert("RGB")
        x = self.tf(im).unsqueeze(0).to(self.device)
        with self._lock:
            p = float(self.model(x).sigmoid().item())
        return {"prob_ai": round(p, 4)}
