"""Physics check for videos: NSG-VD (Zhang et al., NeurIPS 2025 spotlight, Apache-2.0).
Real videos obey probability-flow conservation (how pixel likelihood moves in space vs time);
generated videos break it. Feature = Normalized Spatiotemporal Gradient from a pre-trained
diffusion model (OpenAI 256x256 uncond, in a local checkpoint folder), scored by deep-MMD against
100 reference real videos with the authors' released checkpoint (trained on Pika / SEINE only).

Data: one GenVideo shard (sshaiton/GenVideo-50k-shard, shard 0) in data/benchmarks/genvideo/:
real clips + 10 generators. The official code is run unchanged from third_party/nsgvd
(this script chdir's there, because the code uses relative paths).
Run: .venv-win\\Scripts\\python.exe experiments\\eval_nsgvd.py [ckpt_name]
"""
import glob, json, os, random, sys
import cv2, numpy as np, pandas as pd, torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "third_party", "nsgvd")
VID = os.path.join(os.path.dirname(HERE), "data", "benchmarks", os.environ.get("GENVIDEO_DIR", "genvideo"))
FEAT = VID + "_nsg"
OUT = os.path.join(os.path.dirname(HERE), "results", "benchmarks")
CKPT = sys.argv[1] if len(sys.argv) > 1 else "standard-Pika-d"
TAG = "" if os.path.basename(VID) == "genvideo" else "_" + os.path.basename(VID)
N_FRAMES, SIZE, STEPS, N_REF = 8, 224, 5, 100
os.makedirs(FEAT, exist_ok=True)

os.chdir(SRC); sys.path.insert(0, SRC)
from data.utils import get_score_fn
from models.deep_mmd import deep_MMD
from models.tall import SingleSwinBlockDiscriminator
from utils.mmd_utils import MMD_batch2
dev = "cuda"


def frames(path):
    """8 evenly spaced frames, as the official ffmpeg step does; LANCZOS to 224, values in [0,1]"""
    cap = cv2.VideoCapture(path); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, n // N_FRAMES); out = []
    for i in range(N_FRAMES):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step); ok, f = cap.read()
        if not ok:
            break
        img = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).resize((SIZE, SIZE), Image.Resampling.LANCZOS)
        out.append(torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0)
    cap.release()
    return torch.stack(out) if len(out) == N_FRAMES else None


@torch.no_grad()
def nsg(score_fn, x):
    """official _get_score_features + _get_score_velocity + get_data(velocity) for one video (T,C,H,W)"""
    t = torch.tensor((STEPS + 1) / 1000, device=dev)
    score = score_fn(2 * x - 1, t.expand(x.shape[0]))
    score = torch.split(score, score.shape[1] // 2, dim=1)[0]                  # (T,C,H,W)
    d = torch.zeros_like(x)
    d[1:-1] = (x[2:] - x[:-2]) / 2; d[0] = x[1] - x[0]; d[-1] = x[-1] - x[-2]
    v = score / ((score * d).sum(dim=(1, 2, 3)).view(-1, 1, 1, 1) + 1e-10)
    v[0] *= 100; v[-1] *= 100
    return v


# 1) features (cached, so reruns with another checkpoint are fast)
meta = []
for j in sorted(glob.glob(os.path.join(VID, "*.json"))):
    m = json.load(open(j)); m["path"] = j[:-5] + ".mp4"; m["id"] = os.path.basename(j)[:-5]; meta.append(m)
todo = [m for m in meta if not os.path.exists(os.path.join(FEAT, m["id"] + ".npy"))]
if todo:
    score_fn = get_score_fn(dev)
    for i, m in enumerate(todo, 1):
        x = frames(m["path"])
        if x is not None:
            np.save(os.path.join(FEAT, m["id"] + ".npy"), nsg(score_fn, x.to(dev)).half().cpu().numpy())
        if i % 50 == 0:
            print(f"  NSG features {i}/{len(todo)}", flush=True)
    del score_fn; torch.cuda.empty_cache()
meta = [m for m in meta if os.path.exists(os.path.join(FEAT, m["id"] + ".npy"))]
# near-static clips make the NSG denominator ~0 and overflow; they cannot be scored, so report and skip them
finite = lambda m: np.isfinite(np.load(os.path.join(FEAT, m["id"] + ".npy")).astype(np.float32)).all()
skipped = [m["id"] for m in meta if not finite(m)]
meta = [m for m in meta if m["id"] not in skipped]
print(f"skipped {len(skipped)} clips with non-finite NSG features (near-static video): {skipped}")
load = lambda ms: torch.stack([torch.from_numpy(np.load(os.path.join(FEAT, m["id"] + ".npy")).astype(np.float32)) for m in ms])

# 2) deep-MMD against 100 held-out reference real videos
model = deep_MMD(SingleSwinBlockDiscriminator(num_features=300), sigma=1000, sigma0=0.1, epsilon=10,
                 img_size=SIZE, is_yy_zero=True, is_smooth=True)
model.load_state_dict(torch.load(os.path.join(SRC, "ckpts", CKPT + ".pth"), weights_only=True))
model.to(dev).eval()
random.seed(0)
reals = [m for m in meta if m["label"] == "real"]; random.shuffle(reals)
ref, test = reals[:N_REF], reals[N_REF:] + [m for m in meta if m["label"] == "fake"]
with torch.no_grad():
    ref_x = load(ref).to(dev)
    _, ref_f = model.net(ref_x, out_feature=True)
    stats = []
    for k in range(0, len(test), 8):
        x = load(test[k:k + 8]).to(dev)
        _, f = model.net(x, out_feature=True)
        s = MMD_batch2(torch.cat([ref_f, f]), ref_f.shape[0], torch.cat([ref_x, x]).view(ref_x.shape[0] + x.shape[0], -1),
                       model.sigma, model.sigma0_u, model.ep, is_smooth=True)
        stats += s.cpu().tolist()

from sklearn.metrics import roc_auc_score
df = pd.DataFrame({"id": [m["id"] for m in test], "generator": [m["generator"] for m in test],
                   "label": [int(m["label"] == "fake") for m in test], "nsg_mmd": stats})
df.to_csv(os.path.join(OUT, f"nsgvd_{CKPT}{TAG}_eval.csv"), index=False)
auc = roc_auc_score(df.label, df.nsg_mmd)
real_s = df[df.label == 0].nsg_mmd
t95 = float(np.quantile(real_s, 0.95))                     # threshold giving 5% false alarms on real
print(f"\nNSG-VD ({CKPT}) on GenVideo shard 0: {int((df.label == 0).sum())} real (+{N_REF} reference) vs {int(df.label.sum())} AI videos")
print(f"overall AUC {auc:.3f}   (official threshold >1: AI detected {(df[df.label == 1].nsg_mmd > 1).mean() * 100:.1f}%, "
      f"real flagged {(real_s > 1).mean() * 100:.1f}%)")
print(f"at 5% false alarms on real: AI detected {(df[df.label == 1].nsg_mmd > t95).mean() * 100:.1f}%")
print("\nper generator (AUC vs the real test videos; 'seen' = generator used to train the checkpoint):")
per = {}
for g, d in df[df.label == 1].groupby("generator"):
    a = roc_auc_score(np.r_[np.zeros(len(real_s)), np.ones(len(d))], np.r_[real_s, d.nsg_mmd])
    det = (d.nsg_mmd > t95).mean() * 100
    per[g] = (len(d), round(float(a), 3), round(float(det), 1))
    print(f"  {g:16s} n={len(d):3d}  AUC {a:.3f}  detected@5%FPR {det:5.1f}%{'  (seen)' if g in ('Pika', 'SEINE') else ''}")

sys.path.insert(0, os.path.dirname(HERE))
from experiments.bench_record import save
save({"benchmark": f"NSG-VD physics video detector ({CKPT}) on GenVideo shard 0 (10 generators){TAG}",
      "n_real_test": int((df.label == 0).sum()), "n_ref_real": N_REF, "n_ai": int(df.label.sum()),
      "auc": round(float(auc), 4), "ai_detected_at_5pct_fpr": round(float((df[df.label == 1].nsg_mmd > t95).mean() * 100), 1),
      "per_generator (n, AUC, detected@5%FPR)": per, "per_video_csv": f"nsgvd_{CKPT}{TAG}_eval.csv"},
     OUT, f"nsgvd_{CKPT}{TAG}_summary.json")
