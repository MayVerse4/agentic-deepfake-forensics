"""Random sample of OmniFake-OOD (Omni-Fake, CVPR 2026; CC-BY-4.0; out-of-distribution test set: other sources and
generators than its training set). Labels: REAL / FULL_SYNTHETIC / TAMPERED (+ box for tampered images).
Fetched through the Hugging Face datasets-server, so the full set is not downloaded.
Run: .venv-win\\Scripts\\python.exe experiments\\fetch_omnifake_ood.py <config: image|audio|video> [n_chunks] [chunk]
Writes data/benchmarks/omnifake_ood/<config>/ files + labels.csv
"""
import json, os, random, sys, time, urllib.request
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = sys.argv[1]
N_CHUNKS = int(sys.argv[2]) if len(sys.argv) > 2 else 15
CHUNK = int(sys.argv[3]) if len(sys.argv) > 3 else 50
TOTAL = {"image": 95645, "audio": 88977, "video": 2844, "audiovisual": 6271}[CFG]
URL = ("https://datasets-server.huggingface.co/rows?dataset=Zlatan-Ye/OmniFake-OOD&config={c}&split=test"
       "&offset={o}&length={n}")
OUT = os.path.join(os.path.dirname(HERE), "data", "benchmarks", "omnifake_ood", CFG); os.makedirs(OUT, exist_ok=True)
LAB = os.path.join(OUT, "labels.csv")


def get(url, binary=False):
    for k in range(5):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                b = r.read()
                return b if binary else json.loads(b)
        except Exception:
            time.sleep(4 * (k + 1))
    return None


done = pd.read_csv(LAB) if os.path.exists(LAB) else pd.DataFrame(columns=["id"])
have, rows = set(done.id), done.to_dict("records")
random.seed(0)
offsets = sorted(random.sample(range(0, TOTAL - CHUNK), N_CHUNKS))
media_key = {"image": "image", "audio": "audio", "video": "video", "audiovisual": "video"}[CFG]
for i, off in enumerate(offsets, 1):
    page = get(URL.format(c=CFG, o=off, n=CHUNK))
    for r in (page or {}).get("rows", []):
        row = r["row"]
        if row["id"] in have:
            continue
        m = row.get(media_key)
        src = (m[0] if isinstance(m, list) else m or {}).get("src") if m else None
        if not src:
            continue
        ext = os.path.splitext(src.split("?")[0])[1] or ".bin"
        data = get(src, binary=True)
        if not data:
            continue
        fn = row["id"] + ext
        open(os.path.join(OUT, fn), "wb").write(data)
        rows.append({"id": row["id"], "file": fn, "label": row["label"], "box": json.dumps(row.get("box")),
                     **{k: row[k] for k in ("interval", "start", "end") if k in row}})
    pd.DataFrame(rows).to_csv(LAB, index=False)
    print(f"  {CFG}: chunk {i}/{N_CHUNKS}, {len(rows)} files", flush=True)
print(pd.DataFrame(rows).label.value_counts().to_dict())
