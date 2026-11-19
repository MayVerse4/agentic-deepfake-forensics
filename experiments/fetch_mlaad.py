"""Voice-clone training data (fake side): MLAAD (Mueller et al., CC-BY-NC-4.0, gated; HF login needed).
N clips from every English generator folder, chosen at random (seed 0). Resumable; curl with retries because the HF
connection resets often on this network.
Run: .venv-win\\Scripts\\python.exe experiments\\fetch_mlaad.py [n_per_model=30]
Writes data/benchmarks/voice_train/mlaad/<model>/<file>.wav and data/benchmarks/voice_train/mlaad_index.csv
"""
import json, os, random, subprocess, sys, urllib.parse
import pandas as pd

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "data", "benchmarks", "voice_train", "mlaad"); os.makedirs(OUT, exist_ok=True)
IDX = os.path.join(os.path.dirname(HERE), "data", "benchmarks", "voice_train", "mlaad_index.csv")
TOK = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
API = "https://huggingface.co/api/datasets/mueller91/MLAAD/tree/main/"
RES = "https://huggingface.co/datasets/mueller91/MLAAD/resolve/main/"


def curl(url, out=None):
    cmd = ["curl", "-sfL", "--retry", "8", "--retry-all-errors", "-H", f"Authorization: Bearer {TOK}", url]
    if out:
        cmd += ["-o", out]
    r = subprocess.run(cmd, capture_output=True)
    return r.returncode == 0 and (out is not None or r.stdout), r.stdout


def ls(path):
    ok, b = curl(API + urllib.parse.quote(path))
    return json.loads(b) if ok else []


rows = pd.read_csv(IDX).to_dict("records") if os.path.exists(IDX) else []
have = {r["path"] for r in rows}
random.seed(0)
models = [m["path"] for m in ls("fake/en") if m["type"] == "directory"]
print(len(models), "generators", flush=True)
for k, m in enumerate(models, 1):
    files = sorted(f["path"] for f in ls(m) if f["path"].endswith(".wav"))
    pick = random.sample(files, min(N, len(files)))
    name = m.split("/")[-1]
    os.makedirs(os.path.join(OUT, name), exist_ok=True)
    for p in pick:
        if p in have:
            continue
        dst = os.path.join(OUT, name, os.path.basename(p))
        ok, _ = curl(RES + urllib.parse.quote(p), dst)
        if ok and os.path.getsize(dst) > 1000:
            rows.append({"path": p, "file": os.path.relpath(dst, HERE), "generator": name, "label": 1})
    pd.DataFrame(rows).to_csv(IDX, index=False)
    print(k, len(models), name, len(rows), flush=True)
print("done", len(rows))
