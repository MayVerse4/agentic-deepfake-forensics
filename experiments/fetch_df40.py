"""DF40 (Yan et al., NeurIPS 2024 D&B; CC BY-NC 4.0; links from github.com/YZY-stack/DF40 README, access form filled).
Downloads: train fakes (31 known methods, FF++ domain), FF++ and Celeb-DF real crops, and the test zips of the
9 methods that are NOT in the train set (held-out "unseen methods" test). Resumable: skips files already present.
Run: .venv-win\\Scripts\\python.exe experiments\\fetch_df40.py
Writes data/benchmarks/df40/{train,test,real}/*.zip (unzipping is a separate step).
"""
import os, re, subprocess, time, urllib.request
import gdown

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(os.path.dirname(HERE), "data", "benchmarks", "df40")
TRAIN = "1U8meBbqVvmUkc5GD0jxct6xe6Gwk9wKD"
TEST = "1980LCMAutfWvV6zvdxhoeIa67TmzKLQ_"
REAL = {"ff_real.zip": "1dHJdS0NZ6wpewbGA5B0PdIBS9gz28pdb", "cdf_real.zip": "1FGZ3aYsF-Yru50rPLoT5ef8-2Nkt4uBw"}
UNSEEN = {"CollabDiff", "deepfacelab", "e4e", "heygen", "MidJourney", "stargan", "starganv2", "styleclip",
          "whichfaceisreal"}


def folder_files(fid):
    html = urllib.request.urlopen(f"https://drive.google.com/embeddedfolderview?id={fid}").read().decode("utf8")
    return re.findall(r'https://drive\.google\.com/file/d/([\w-]+)/view[^"]*".*?flip-entry-title">([^<]+)', html, re.S)


def get(fid, dst):
    if os.path.exists(dst) and os.path.getsize(dst) > 1_000_000:
        return True
    for k in range(2):
        try:
            if gdown.download(id=fid, output=dst + ".part", quiet=True, resume=True):
                os.replace(dst + ".part", dst); return True
        except Exception as e:
            print("  retry", os.path.basename(dst), str(e)[:100], flush=True)
            time.sleep(20)
    return False


jobs = [(fid, os.path.join(ROOT, "real", n)) for n, fid in REAL.items()]
jobs += [(fid, os.path.join(ROOT, "test", n)) for fid, n in folder_files(TEST) if n[:-4] in UNSEEN]
jobs += [(fid, os.path.join(ROOT, "train", n)) for fid, n in folder_files(TRAIN)]
print(len(jobs), "files", flush=True)
for i, (fid, dst) in enumerate(jobs, 1):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    ok = get(fid, dst)
    print(i, len(jobs), os.path.relpath(dst, ROOT), "ok" if ok else f"FAILED  browser: https://drive.google.com/uc?id={fid}",
          f"{os.path.getsize(dst) / 1e9:.2f} GB" if ok else "", flush=True)
print("done")
