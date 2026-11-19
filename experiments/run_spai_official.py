"""Runs SPAI's official inference CLI (python -m spai infer) on AIGIBench, unchanged, in .venv-eval.
Only addition: allowlists yacs CfgNode (the training config stored inside spai.pth) for torch's safe loader,
instead of turning safe loading off. Writes results/benchmarks/spai/ (official per-image scores CSV).
Run: .venv-eval\Scripts\python.exe experiments\run_spai_official.py
"""
import csv, os, sys
import torch
from yacs.config import CfgNode
torch.serialization.add_safe_globals([CfgNode])



def main():
    HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(HERE)
    ROOT = os.path.join(os.path.dirname(HERE), "data", "benchmarks", "aigibench"); OUT = os.path.join(PROJ, "results", "benchmarks", "spai")
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for sub in sorted(os.listdir(ROOT)):
        for cls in ("0_real", "1_fake"):
            d = os.path.join(ROOT, sub, cls)
            for f in sorted(os.listdir(d)):
                try:                                   # skip unreadable files (same 998 images as the other detectors)
                    from PIL import Image
                    Image.open(os.path.join(d, f)).convert("RGB")
                except Exception:
                    print("skip unreadable", f); continue
                rows.append({"image": f"{sub}/{cls}/{f}", "split": "test", "class": cls[0]})
    csv_path = os.path.join(OUT, "aigibench_input.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["image", "split", "class"]); w.writeheader(); w.writerows(rows)
    print(len(rows), "images")
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "third_party", "spai")); os.chdir(os.path.join(os.path.dirname(HERE), "third_party", "spai"))
    from spai.__main__ import cli
    sys.argv = ["spai", "infer", "--cfg", os.path.join(os.path.dirname(HERE), "third_party", "spai", "configs", "spai.yaml"), "--input", csv_path, "--input-csv-root-dir", ROOT, "--output", OUT,
                "--model", os.path.join(os.path.dirname(HERE), "weights", "pretrained", "spai", "spai.pth"), "--resize-to", "1024", "--opt", "DATA.NUM_WORKERS", "2", "--opt", "DATA.PIN_MEMORY", "False"]
    cli()


if __name__ == "__main__":
    main()
