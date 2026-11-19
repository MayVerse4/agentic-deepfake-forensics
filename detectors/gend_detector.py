"""GenD face-deepfake detector (Yermakov et al., "Deepfake Detection that Generalizes Across Benchmarks", WACV 2026;
MIT; weights yermandy/GenD_CLIP_L_14). Re-implemented here from the released modeling_gend.py so no downloaded code
is executed: CLIP ViT-L/14 vision tower (LayerNorms fine-tuned) -> pooler_output -> L2-normalise -> Linear(1024, 2).
Input: a face crop (PIL RGB); preprocessing = the CLIP processor (resize 224, centre crop, CLIP mean/std).
prob_fake = softmax(logits)[1].
"""
import os
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(os.path.dirname(HERE), "weights", "trained", "gend", "model.safetensors")
BACKBONE = "openai/clip-vit-large-patch14"


class GenDDetector:
    def __init__(self, device=None):
        from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.proc = CLIPImageProcessor.from_pretrained(BACKBONE)
        self.vision = CLIPVisionModel(CLIPVisionConfig.from_pretrained(BACKBONE))
        self.vision = getattr(self.vision, "vision_model", self.vision)
        sd = load_file(CKPT)
        pre = "feature_extractor.vision_model."
        vis = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
        missing, unexpected = self.vision.load_state_dict(vis, strict=False)
        missing = [m for m in missing if "position_ids" not in m]
        if missing or unexpected:
            raise RuntimeError(f"GenD weights mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
        self.w, self.b = sd["model.linear.weight"], sd["model.linear.bias"]
        self.vision.to(self.device).eval().half() if self.device == "cuda" else self.vision.eval()
        self.w, self.b = self.w.to(self.device).float(), self.b.to(self.device).float()

    @torch.no_grad()
    def predict(self, faces):
        """faces: list of PIL RGB face crops -> list of prob_fake."""
        if not faces:
            return []
        x = self.proc(images=faces, return_tensors="pt")["pixel_values"].to(self.device)
        x = x.half() if self.device == "cuda" else x
        out = []
        for i in range(0, len(x), 16):
            f = self.vision(pixel_values=x[i:i + 16]).pooler_output.float()
            out += F.linear(F.normalize(f, dim=1), self.w, self.b).softmax(-1)[:, 1].tolist()
        return out
