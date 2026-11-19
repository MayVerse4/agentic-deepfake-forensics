"""Downloads a small, fixed random sample of AIGIBench (Li et al. 2025, "Is AI-Generated Image Detection a Solved
Problem?", CC BY-NC-SA 4.0) without downloading the 2-23 GB zips: remotezip reads single files by HTTP range.
Each test subset has its own 0_real / 1_fake folders. Writes data/benchmarks/aigibench/<subset>/{0_real,1_fake}/.
Run: .venv-win\\Scripts\\python.exe experiments\\fetch_aigibench.py [n_per_class]
"""
import os, random, sys
from remotezip import RemoteZip

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "data", "benchmarks", "aigibench")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
BASE = "https://huggingface.co/datasets/HorizonTEL/AIGIBench/resolve/main/test/{}.zip"
# newest generators + the two "in the wild" sets (community-shared AI images, social-media real/fake)
SUBSETS = ["Midjourney", "FLUX1-dev", "SD3", "Imagen3", "DALLE-3", "SDXL", "CommunityAI", "SocialRF",
           "InstantID", "StyleGAN-XL"]
IMG = (".jpg", ".jpeg", ".png", ".webp")

for s in SUBSETS:
    for attempt in range(3):
        try:
            with RemoteZip(BASE.format(s)) as z:
                names = [n for n in z.namelist() if n.lower().endswith(IMG)]
                for cls in ("0_real", "1_fake"):
                    pool = sorted(n for n in names if f"/{cls}/" in n)
                    random.seed(0)
                    pick = random.sample(pool, min(N, len(pool)))
                    d = os.path.join(OUT, s, cls); os.makedirs(d, exist_ok=True)
                    have = set(os.listdir(d))
                    for n in pick:
                        fn = n.replace("/", "_")
                        if fn not in have:
                            with open(os.path.join(d, fn), "wb") as f:
                                f.write(z.read(n))
                    print(f"  {s}/{cls}: {len(pick)} of {len(pool)}", flush=True)
            break
        except Exception as e:
            print(f"  {s}: retry ({type(e).__name__}: {e})", flush=True)
print("done")
