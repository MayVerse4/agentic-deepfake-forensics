"""Fine-tune the Spatial EfficientNet-B0 on DF40 (+ FF++ real) from the deployed FF++ checkpoint.
Fixes vs. models/spatial/train_spatial_model.py: compression/blur/resize augmentation, balanced real/fake sampling,
video-level splits (index_df40.py), AMP, warmup+cosine, model selected by val AUC (not loss), AUC/EER/TPR@5%FPR logged,
DF40 unseen test sets reported every epoch (never used for selection). Reads the extracted subset (extract_df40.py).
Original checkpoint is never overwritten; output -> results/retrain/spatial_df40_*.pth
"""
import os, sys, io, csv, json, time, math, random, zipfile, argparse
import numpy as np, torch, torch.nn as nn
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.metrics import roc_auc_score, roc_curve

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "models", "spatial"))
from spatial_model import DeepfakeSpatialModel  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=5); ap.add_argument("--per_epoch", type=int, default=60000)
ap.add_argument("--bs", type=int, default=48); ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--workers", type=int, default=2); ap.add_argument("--patience", type=int, default=2)
ap.add_argument("--val_n", type=int, default=10000); ap.add_argument("--test_n", type=int, default=8000)
ap.add_argument("--time_only", action="store_true")
ap.add_argument("--resume", default=""); ap.add_argument("--ff_mix", action="store_true")
ap.add_argument("--tag", default="df40"); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
DATA = os.path.join(REPO, "data", "benchmarks", "df40_x"); OUT = os.path.join(REPO, "weights", "trained", "retrain"); os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, f"spatial_{a.tag}_log.jsonl")

class Degrade:
    """Random JPEG / blur / down-up resize: the main cause of poor cross-dataset transfer is compression mismatch."""
    def __call__(self, im):
        if random.random() < 0.3:
            s = random.choice([0.5, 0.67, 0.8]); w, h = im.size
            im = im.resize((int(w * s), int(h * s)), Image.BILINEAR).resize((w, h), Image.BILINEAR)
        if random.random() < 0.2:
            im = im.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.5)))
        if random.random() < 0.5:
            b = io.BytesIO(); im.save(b, "JPEG", quality=random.randint(30, 95)); b.seek(0); im = Image.open(b).convert("RGB")
        return im
NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
TF_TR = T.Compose([T.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)), T.RandomHorizontalFlip(), Degrade(),
                   T.ColorJitter(0.1, 0.1, 0.1), T.ToTensor(), NORM])
TF_EV = T.Compose([T.Resize((224, 224)), T.ToTensor(), NORM])

class ZipFaces(Dataset):
    def __init__(self, rs, tf): self.rs, self.tf, self.z = rs, tf, {}
    def __len__(self): return len(self.rs)
    def __getitem__(self, i):
        r = self.rs[i]
        im = Image.open(os.path.join(DATA, r["path"])).convert("RGB")
        return self.tf(im), torch.tensor(float(r["label"]))

def loader(rs, tf, shuffle):
    return DataLoader(ZipFaces(rs, tf), batch_size=a.bs, shuffle=shuffle, num_workers=a.workers, pin_memory=False,
                      persistent_workers=False, drop_last=shuffle)

def metrics(y, p):
    auc = roc_auc_score(y, p); fpr, tpr, _ = roc_curve(y, p)
    eer = fpr[np.nanargmin(np.abs(1 - tpr - fpr))]; t5 = tpr[np.searchsorted(fpr, 0.05, side="right") - 1]
    return dict(auc=round(auc, 4), eer=round(float(eer), 4), tpr5=round(float(t5), 4), acc=round(float(((p > .5) == y).mean()), 4))

@torch.no_grad()
def evaluate(m, rs):
    m.eval(); ys, ps = [], []
    for x, y in loader(rs, TF_EV, False):
        with torch.autocast("cuda", dtype=torch.float16):
            ps.append(torch.sigmoid(m(x.cuda(non_blocking=True)).float().squeeze(1)).cpu()); ys.append(y)
    return metrics(torch.cat(ys).numpy(), torch.cat(ps).numpy())

if __name__ == "__main__":  # required on Windows: DataLoader workers re-import this file
    rows = list(csv.DictReader(open(os.path.join(DATA, "index.csv"))))
    by = {k: [r for r in rows if r["split"] == k] for k in ("train", "val", "test")}
    tr_real = [r for r in by["train"] if r["label"] == "0"]; tr_fake = [r for r in by["train"] if r["label"] == "1"]
    FFD = os.path.join(REPO, "data", "Dataset_Frames")
    def ff_rows(split):
        return [dict(path=os.path.join(FFD, split, d, f), label=str(l)) for l, d in ((0, "real"), (1, "fake")) for f in os.listdir(os.path.join(FFD, split, d))]
    ff_val = ff_rows("Validation")
    if a.ff_mix:  # keep FF++ performance (no-loss rule): FF++ Train frames join the training pool
        ff_tr = ff_rows("Train"); tr_real += [r for r in ff_tr if r["label"] == "0"]; tr_fake += [r for r in ff_tr if r["label"] == "1"]
    def bal_sample(rs, n):
        re_ = [r for r in rs if r["label"] == "0"]; fk = [r for r in rs if r["label"] == "1"]
        return random.sample(re_, min(n // 2, len(re_))) + random.sample(fk, min(n // 2, len(fk)))
    val_rows = bal_sample(by["val"], a.val_n); test_rows = bal_sample(by["test"], a.test_n)
    print(f"train real {len(tr_real)} fake {len(tr_fake)} | val {len(val_rows)} | test {len(test_rows)}", flush=True)

    dev = torch.device("cuda")
    m = DeepfakeSpatialModel(); m.load_state_dict(torch.load(a.resume or os.path.join(REPO, "weights", "trained", "spatial_model_finetuned_ff.pth"), map_location="cpu"))
    m.to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * (a.per_epoch // a.bs); warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler(); crit = nn.BCEWithLogitsLoss()

    if not a.time_only:
        e0 = dict(epoch=0, val=evaluate(m, val_rows), ff=evaluate(m, ff_val), test=evaluate(m, test_rows)); print("before:", e0, flush=True)
        open(LOG, "a").write(json.dumps(e0) + "\n")
    best, bad = -1, 0
    for ep in range(1, a.epochs + 1):
        ep_rows = random.sample(tr_real, min(a.per_epoch // 2, len(tr_real))) + random.sample(tr_fake, a.per_epoch // 2)
        m.train(); t0 = time.time(); tot = n = 0
        for i, (x, y) in enumerate(loader(ep_rows, TF_TR, True)):
            x, y = x.to(dev, non_blocking=True), y.to(dev)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = crit(m(x).squeeze(1), y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(y); n += len(y)
            if i % 100 == 0:
                el = time.time() - t0; print(f"ep{ep} {i}/{a.per_epoch // a.bs} loss {tot / n:.4f} {el / max(i, 1):.2f}s/it", flush=True)
            if a.time_only and i == 30:
                print(f"TIME: {el / 30 * (a.per_epoch // a.bs) / 60:.1f} min per epoch of {a.per_epoch}", flush=True); sys.exit()
        v, te, ffv = evaluate(m, val_rows), evaluate(m, test_rows), evaluate(m, ff_val)
        rec = dict(epoch=ep, loss=round(tot / n, 4), min=round((time.time() - t0) / 60, 1), val=v, ff=ffv, test=te); print(rec, flush=True)
        open(LOG, "a").write(json.dumps(rec) + "\n")
        torch.save(m.state_dict(), os.path.join(OUT, f"spatial_{a.tag}_last.pth"))
        sel = (v["auc"] + ffv["auc"]) / 2 if a.ff_mix else v["auc"]  # select on DF40 val + FF++ val
        if sel > best:
            best, bad = sel, 0; torch.save(m.state_dict(), os.path.join(OUT, f"spatial_{a.tag}_best.pth")); print("  saved best", flush=True)
        else:
            bad += 1
            if bad >= a.patience: print("early stop", flush=True); break
    print("done; best val AUC", best)
