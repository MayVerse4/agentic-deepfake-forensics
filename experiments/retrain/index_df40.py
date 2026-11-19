"""Index DF40 zips (pre-cropped 256px faces) into a CSV: zip, member, label, method, video, split.
Split is by source video (hash), so frames of one video never cross train/val. Test-set zips -> split=test."""
import zipfile, glob, os, csv, hashlib, re, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "benchmarks", "df40")
rows = []
for z in sorted(glob.glob(os.path.join(ROOT, "*", "*.zip"))):
    part = os.path.basename(os.path.dirname(z)); name = os.path.basename(z)[:-4]
    for m in zipfile.ZipFile(z).namelist():
        if not m.lower().endswith((".png", ".jpg")) or "/landmarks/" in m or os.path.basename(m).startswith("._"):
            continue
        parts = m.split("/"); video = parts[-2]
        if part == "real":
            label, method = 0, name
        elif "/real/" in m:
            label, method = 0, name + "_real"
        else:
            label, method = 1, name
        if part == "test":
            split = "test"
        else:
            src = re.split(r"[_+]", video)[0]  # source identity for swaps like 402_453
            h = int(hashlib.md5(src.encode()).hexdigest(), 16) % 10
            split = "val" if h == 0 else "train"
        rows.append((os.path.relpath(z, ROOT), m, label, method, video, split))
out = os.path.join(ROOT, "df40_index.csv")
with open(out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["zip", "member", "label", "method", "video", "split"]); w.writerows(rows)
from collections import Counter
c = Counter((r[5], r[2]) for r in rows)
print(out, len(rows), dict(c))
