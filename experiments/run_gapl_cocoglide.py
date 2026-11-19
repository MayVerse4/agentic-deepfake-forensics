"""Scores AIGIBench with GAPL (CVPR 2026, MIT; official checkpoint AbyssLumine/GAPL/checkpoint.pt) in .venv-eval,
following the authors' gapl_src/inference_image.py exactly: GAPLModel(fe_path=None, proto_path=None,
freeze_backbone=False), load ckpt['model'] + ckpt['prototype'], CenterCrop(224) + ImageNet mean/std, sigmoid.
Only change: models.py hard-codes a Linux CLIP cache path; CLIPModel.from_pretrained is redirected to the
same model id ("openai/clip-vit-large-patch14"). Missing/unexpected weights are printed, never hidden.
Run: .venv-eval\\Scripts\\python.exe experiments\\run_gapl_official.py
"""
import os, sys
import pandas as pd, torch
import torchvision.transforms as T
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(HERE)
ROOT = os.path.join(os.path.dirname(HERE), "data", "benchmarks"); OUT = os.path.join(PROJ, "results", "benchmarks", "gapl_cocoglide_scores.csv")
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "third_party", "gapl"))
import transformers
_orig = transformers.CLIPModel.from_pretrained
transformers.CLIPModel.from_pretrained = lambda *a, **k: _orig("openai/clip-vit-large-patch14")
import models
models.CLIPModel.from_pretrained = transformers.CLIPModel.from_pretrained

dev = "cuda"
model = models.GAPLModel(fe_path=None, proto_path=None, freeze_backbone=False)
ckpt = torch.load(os.path.join(os.path.dirname(HERE), "weights", "pretrained", "gapl", "checkpoint.pt"), map_location="cpu", weights_only=True)
missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
print(f"checkpoint keys {len(ckpt['model'])}; missing {len(missing)}; unexpected {len(unexpected)}")
print("  missing sample:", missing[:5]); print("  unexpected sample:", unexpected[:5])
model.load_prototype(ckpt["prototype"])
model.to(dev).eval()
tf = T.Compose([T.CenterCrop(224), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

done = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=["path", "gapl"])
have, rows = set(done.path), done.to_dict("records")
files = [f"{c}/{f}" for c in ("real", "fake") for f in sorted(os.listdir(os.path.join(ROOT, c)))]
with torch.no_grad():
    for i, rel in enumerate(files, 1):
        if rel in have:
            continue
        try:
            x = tf(Image.open(os.path.join(ROOT, rel)).convert("RGB")).unsqueeze(0).to(dev)
        except Exception:
            print("skip unreadable", rel); continue
        rows.append({"path": rel, "gapl": float(model(x).sigmoid().item())})
        if i % 100 == 0:
            pd.DataFrame(rows).to_csv(OUT, index=False); print(f"  {i}/{len(files)}", flush=True)
pd.DataFrame(rows).to_csv(OUT, index=False)
print("done", len(rows))
