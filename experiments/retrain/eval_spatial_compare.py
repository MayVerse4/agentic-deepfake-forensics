"""Compare deployed vs retrained Spatial model: FF++ Validation frames (Dataset_Frames) and per-test-set DF40 breakdown."""
import os, sys, csv, json
from collections import defaultdict
import numpy as np, torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.metrics import roc_auc_score, roc_curve

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "models", "spatial"))
from spatial_model import DeepfakeSpatialModel  # noqa: E402
TF = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

class Files(Dataset):
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return TF(Image.open(self.items[i][0]).convert("RGB")), self.items[i][1]

@torch.no_grad()
def scores(m, items):
    out = []
    for x, _ in DataLoader(Files(items), batch_size=96, num_workers=2):
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(torch.sigmoid(m(x.cuda()).float().squeeze(1)).cpu())
    return torch.cat(out).numpy()

def met(y, p):
    y = np.asarray(y); r = dict(n=len(y), acc=round(float(((p > .5) == y).mean()), 4))
    if len(set(y)) == 2:
        fpr, tpr, _ = roc_curve(y, p); r["auc"] = round(roc_auc_score(y, p), 4)
        r["tpr5"] = round(float(tpr[np.searchsorted(fpr, 0.05, side="right") - 1]), 4)
        r["fpr_at_0.5"] = round(float((p[y == 0] > .5).mean()), 4)
    else:
        r["fake_caught_at_0.5" if y[0] == 1 else "real_flagged_at_0.5"] = round(float((p > .5).mean()), 4)
    return r

if __name__ == "__main__":
    models = {}
    for k, f in (("deployed", "weights/trained/spatial_model_finetuned_ff.pth"), ("df40ff", "weights/trained/retrain/spatial_df40ff_best.pth"),
                 ("df40ffhq", "weights/trained/retrain/spatial_df40ffhq_best.pth")):
        m = DeepfakeSpatialModel(); m.load_state_dict(torch.load(os.path.join(REPO, f), map_location="cpu")); models[k] = m.cuda().eval()
    # FF++ validation frames
    V = os.path.join(REPO, "data", "Dataset_Frames", "Validation"); ff = []
    for lab, d in ((0, "real"), (1, "fake")):
        ff += [(os.path.join(V, d, f), lab, f.split("_")[0] if lab else "real") for f in sorted(os.listdir(os.path.join(V, d)))]
    # DF40 test set
    D = os.path.join(REPO, "data", "benchmarks", "df40_x")
    te = [(os.path.join(D, r["path"]), int(r["label"]), r["method"]) for r in csv.DictReader(open(os.path.join(D, "index.csv"))) if r["split"] == "test"]
    res = {}
    for name, items in (("FF++ validation", ff), ("DF40 test", te)):
        y = np.array([i[1] for i in items]); groups = defaultdict(list)
        for j, i in enumerate(items): groups[i[2]].append(j)
        for k, m in models.items():
            p = scores(m, items); res[(name, k, "ALL")] = met(y, p)
            for g, idx in sorted(groups.items()):
                res[(name, k, g)] = met(y[idx], p[idx])
            if name == "DF40 test":  # per test set AUC: fake set vs its own real set
                for g in groups:
                    if not g.endswith("_real") and g + "_real" in groups:
                        idx = groups[g] + groups[g + "_real"]; res[(name, k, g + " AUC")] = met(y[idx], p[idx])
    for key in sorted(res): print(" | ".join(key), res[key])
    json.dump({" | ".join(k): v for k, v in res.items()}, open(os.path.join(REPO, "results", "retrain", "spatial_compare.json"), "w"), indent=1)
