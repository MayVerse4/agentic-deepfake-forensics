"""Extract a balanced DF40 subset from the zips to plain files (zip reads were the bottleneck).
Train: all real frames + fakes spread evenly over methods (~2x real). Val/test: balanced samples.
Writes data/benchmarks/df40_x/index.csv with columns path,label,method,video,split. Zips are left untouched."""
import os, csv, random, zipfile, sys
import zipfile_deflate64  # noqa: F401  (e4e.zip uses Deflate64)
from collections import defaultdict
random.seed(0)
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "benchmarks")
SRC, DST = os.path.join(ROOT, "df40"), os.path.join(ROOT, "df40_x")
rows = list(csv.DictReader(open(os.path.join(SRC, "df40_index.csv"))))
pick = []
for split, fake_mult, cap_real in (("train", 2.0, None), ("val", 1.0, 5000), ("test", 1.0, 4000)):
    rs = [r for r in rows if r["split"] == split]
    real = [r for r in rs if r["label"] == "0"]
    if cap_real: real = random.sample(real, min(cap_real, len(real)))
    fakes = defaultdict(list)
    for r in rs:
        if r["label"] == "1": fakes[r["method"]].append(r)
    per = int(len(real) * fake_mult / len(fakes)) + 1
    pick += real + [r for m in fakes.values() for r in random.sample(m, min(per, len(m)))]
by_zip = defaultdict(list)
for r in pick: by_zip[r["zip"]].append(r)
out = []
for k, (zp, rs) in enumerate(sorted(by_zip.items())):
    z = zipfile.ZipFile(os.path.join(SRC, zp))
    for r in rs:
        rel = os.path.join(r["split"], r["method"], r["video"].replace("/", "_"), os.path.basename(r["member"]))
        p = os.path.join(DST, rel); os.makedirs(os.path.dirname(p), exist_ok=True)
        if not os.path.exists(p):
            with open(p, "wb") as f: f.write(z.read(r["member"]))
        out.append((rel, r["label"], r["method"], r["video"], r["split"]))
    print(f"{k + 1}/{len(by_zip)} {zp} {len(rs)}", flush=True)
with open(os.path.join(DST, "index.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["path", "label", "method", "video", "split"]); w.writerows(out)
from collections import Counter
print("done", len(out), Counter((o[4], o[1]) for o in out))
