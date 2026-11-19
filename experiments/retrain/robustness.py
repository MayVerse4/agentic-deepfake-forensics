"""Robustness of face checkpoints to common real-world degradations (reviewer request: "does it survive social media?").
Each degradation is applied deterministically to every image, then frame-level AUC is measured on FF++ validation and the
DF40 unseen test set. Degradations follow the DeeperForensics / common-perturbation protocol at two severities.
Usage: python robustness.py name=ckpt [...]   -> results/retrain/robustness.json"""
import os, sys, io, json, zlib
import numpy as np, torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.metrics import roc_auc_score
import paper_stats as P

def jpeg(q):
    def f(im):
        b = io.BytesIO(); im.save(b, "JPEG", quality=q); b.seek(0); return Image.open(b).convert("RGB")
    return f
def down(s):
    def f(im):
        w, h = im.size; return im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    return f
def blur(r): return lambda im: im.filter(ImageFilter.GaussianBlur(r))
def noise(sd):
    def f(im):
        a = np.asarray(im).astype(np.float32); g = np.random.default_rng(zlib.crc32(im.tobytes()[:4096]))
        return Image.fromarray(np.clip(a + g.normal(0, sd, a.shape), 0, 255).astype(np.uint8))
    return f
def social(im):  # WhatsApp-like: downscale to 0.5, JPEG 60
    return jpeg(60)(down(0.5)(im))

DEG = {"clean": lambda im: im, "jpeg75": jpeg(75), "jpeg40": jpeg(40), "down0.5": down(0.5), "down0.25": down(0.25),
       "blur1": blur(1.0), "blur2": blur(2.0), "noise5": noise(5), "noise15": noise(15), "social": social}

class Deg(Dataset):
    def __init__(self, paths, name): self.p, self.name = paths, name  # name, not function: picklable on Windows
    def __len__(self): return len(self.p)
    def __getitem__(self, i): return P.TF(DEG[self.name](Image.open(self.p[i]).convert("RGB")))

@torch.no_grad()
def run(m, paths, name):
    out = []
    for x in DataLoader(Deg(paths, name), batch_size=96, num_workers=2):
        out.append(torch.nan_to_num(m(x.cuda()).float().view(-1)).cpu())  # fp32: fp16 gave NaN
    return torch.cat(out).numpy()

if __name__ == "__main__":
    from spatial_model import DeepfakeSpatialModel
    sets = P.data(); res = {}
    for name, ck in (a.split("=", 1) for a in sys.argv[1:]):
        m = DeepfakeSpatialModel(); m.load_state_dict(torch.load(os.path.join(P.REPO, ck), map_location="cpu")); m = m.cuda().eval()
        res[name] = {}
        for k in ("ff_val", "df40_test"):
            y = np.array([i[1] for i in sets[k]]); paths = [i[0] for i in sets[k]]
            res[name][k] = {d: round(roc_auc_score(y, run(m, paths, d)), 4) for d in DEG}
            print(name, k, res[name][k], flush=True)
    json.dump(res, open(os.path.join(P.OUT, "robustness.json"), "w"), indent=1)
