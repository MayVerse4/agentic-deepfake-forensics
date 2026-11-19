"""Score every spatial / frequency checkpoint on the SAME validation data and pick the best by validation only.
Selection score = mean(FF++ validation AUC, DF40 validation AUC). DF40 unseen test AUC is printed for information
only and never used to choose. Output: results/retrain/checkpoint_selection.json"""
import os, sys, csv, json, random, glob
import numpy as np, torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "models", "spatial")); sys.path.insert(0, os.path.join(REPO, "models", "frequency"))
from spatial_model import DeepfakeSpatialModel  # noqa: E402
from frequency_model import load_frequency_model, compute_high_freq_image_gpu  # noqa: E402

NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
TF = {"spatial": T.Compose([T.Resize((224, 224)), T.ToTensor(), NORM]),
      "freq": T.Compose([T.Resize((224, 224)), T.PILToTensor()])}

class Files(Dataset):
    def __init__(self, items, tf): self.items, self.tf = items, tf
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.tf(Image.open(self.items[i][0]).convert("RGB")), self.items[i][1]

@torch.no_grad()
def auc(fn, items, kind):
    ys, ps = [], []
    for x, y in DataLoader(Files(items, TF[kind]), batch_size=64, num_workers=2):
        ps.append(fn(x.cuda()).float().view(-1).cpu()); ys.append(y)
    return round(roc_auc_score(torch.cat(ys).numpy(), torch.cat(ps).numpy()), 4)

def bal(rs, n, seed=0):
    r = random.Random(seed); re_ = [x for x in rs if x[1] == 0]; fk = [x for x in rs if x[1] == 1]
    return r.sample(re_, min(n // 2, len(re_))) + r.sample(fk, min(n // 2, len(fk)))

if __name__ == "__main__":
    V = os.path.join(REPO, "data", "Dataset_Frames", "Validation")
    ff = [(os.path.join(V, d, f), l) for l, d in ((0, "real"), (1, "fake")) for f in os.listdir(os.path.join(V, d))]
    D = os.path.join(REPO, "data", "benchmarks", "df40_x"); idx = list(csv.DictReader(open(os.path.join(D, "index.csv"))))
    dval = bal([(os.path.join(D, r["path"]), int(r["label"])) for r in idx if r["split"] == "val"], 10000)
    dval_noffhq = bal([(os.path.join(D, r["path"]), int(r["label"])) for r in idx if r["split"] == "val" and r["method"] != "ffhq_real"], 10000)
    dtest = [(os.path.join(D, r["path"]), int(r["label"])) for r in idx if r["split"] == "test"]
    cands = [("spatial", p) for p in [os.path.join(REPO, "weights", "trained", "spatial_model_finetuned_ff.pth")]
             + sorted(glob.glob(os.path.join(REPO, "weights", "trained", "retrain", "spatial_*.pth")))
             + sorted(glob.glob(os.path.join(REPO, "archive", "checkpoints_*", "*spatial*.pth")))]
    cands += [("freq", p) for p in [os.path.join(REPO, "weights", "trained", "frequency_resnet18_std.pth")]
              + sorted(glob.glob(os.path.join(REPO, "weights", "trained", "retrain", "freq_*.pth")))
              + sorted(glob.glob(os.path.join(REPO, "archive", "checkpoints_*", "*frequency*.pth")))]
    res = []
    for kind, p in cands:
        name = os.path.relpath(p, REPO)
        try:
            if kind == "spatial":
                m = DeepfakeSpatialModel(); m.load_state_dict(torch.load(p, map_location="cpu")); m = m.cuda().eval()
                fn = lambda x, m=m: torch.sigmoid(m(x))
            else:
                m = load_frequency_model(p, "cuda"); norm = "std" if "std" in name or "retrain" in name else "minmax"
                fn = lambda x, m=m, norm=norm: torch.sigmoid(m(compute_high_freq_image_gpu(x.float(), mask_radius=0.25, norm=norm)))
            r = dict(kind=kind, ckpt=name, ff_val=auc(fn, ff, kind), df40_val=auc(fn, dval, kind),
                     df40_val_no_ffhq=auc(fn, dval_noffhq, kind), df40_test_info_only=auc(fn, dtest, kind))
            r["select"] = round((r["ff_val"] + r["df40_val"]) / 2, 4)
        except Exception as e:
            r = dict(kind=kind, ckpt=name, error=str(e)[:120])
        print(r, flush=True); res.append(r); torch.cuda.empty_cache()
    best = {k: max((r for r in res if r["kind"] == k and "select" in r), key=lambda r: r["select"]) for k in ("spatial", "freq")}
    json.dump(dict(results=res, best=best), open(os.path.join(REPO, "results", "retrain", "checkpoint_selection.json"), "w"), indent=1)
    print("BEST:", json.dumps(best, indent=1))
