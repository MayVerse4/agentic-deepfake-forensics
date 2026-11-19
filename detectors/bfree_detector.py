"""Second whole-image AI detector: B-Free (Guillaro et al., CVPR 2025, GRIP-UNINA; non-profit licence).
DINOv2 ViT-B/14 with registers, trained "bias-free": fakes are Stable-Diffusion re-generations of the SAME
real images (plus local inpainting), so it learns generation artifacts rather than content/format bias.
Weights: weights/pretrained/bfree/BFREE_dino2reg4 (www.grip.unina.it, MD5 f3f53fa647848b16cf81c913f148a198).

BFreeDetector().score_image(path_or_pil) -> {"logit", "prob_ai"}   (5 crops of 504x504, averaged by the model)
Official code (third_party/bfree/code) is loaded by file path, so its top-level "networks"/"utils" names never
clash with the project's.
"""
import importlib.util
import os

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
WRAPPER = os.path.join(os.path.dirname(HERE), "third_party", "bfree", "code", "networks", "wrapper5crops.py")
WEIGHTS = os.path.join(os.path.dirname(HERE), "weights", "pretrained", "bfree", "BFREE_dino2reg4", "model_epoch_best.pth")
ARCH = "vit_base_patch14_reg4_dinov2.lvd142m"
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


class BFreeDetector:
    def __init__(self, device=None):
        import timm
        spec = importlib.util.spec_from_file_location("bfree_wrapper5crops", WRAPPER)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = mod.Wrapper5crops(timm.create_model(ARCH, num_classes=1, pretrained=False), 504)
        model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True)["model"])
        self.model = model.to(self.device).eval()

    @torch.no_grad()
    def score_image(self, img):
        im = (Image.open(img) if isinstance(img, str) else img).convert("RGB")
        if min(im.size) < 32:                        # degenerate image: nothing to analyse
            raise ValueError(f"image too small for B-Free: {im.size}")
        x = torch.from_numpy(((np.asarray(im) / 255.0 - MEAN) / STD).transpose(2, 0, 1)).float()[None]
        out = self.model(x.to(self.device)).float().cpu().numpy()
        logit = float(out[0, 0] if out.shape[1] == 1 else out[0, 1] - out[0, 0])
        return {"logit": round(logit, 4), "prob_ai": round(float(1 / (1 + np.exp(-logit))), 4)}
