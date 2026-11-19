"""LipSync-A (She et al., ICML 2026; CC BY-NC-SA 4.0; open on Hugging Face, no access form).
Downloads up to N paired real/fake clips for the 6 pure lip-sync generators (mouth edited, rest of the video real).
Run: .venv-win\Scripts\python.exe experiments\fetch_lipsync_a.py [N]"""
import os, sys, random
from huggingface_hub import HfApi, hf_hub_download
R = "AnsonShe/LipSync-A"; N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "benchmarks", "lipsync_a")
GENS = os.environ.get("GENS", "wav2lip,talklip,DINet,IP_LAP,LatentSync,OmniSync").split(",")
api = HfApi(); files = [f.path for f in api.list_repo_tree(R, repo_type="dataset", recursive=True) if getattr(f, "size", None)]
for g in GENS:
    fake = {os.path.splitext(os.path.basename(p))[0]: p for p in files if p.startswith(g + "/1_fake/")}
    real = {os.path.splitext(os.path.basename(p))[0]: p for p in files if p.startswith(g + "/0_real/")}
    keys = sorted(set(fake) & set(real)); random.Random(0).shuffle(keys)
    for k in keys[:N]:
        for p in (fake[k], real[k]):
            hf_hub_download(R, p, repo_type="dataset", local_dir=OUT)
    print(g, "pairs:", min(N, len(keys)), flush=True)
