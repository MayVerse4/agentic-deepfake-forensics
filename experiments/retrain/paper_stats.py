"""Reviewer-grade statistics for face (spatial) checkpoints.

Protocol (all choices fixed on validation, then applied once to test):
  * Scores cached per image for FF++ validation, DF40 validation, DF40 unseen test (all 9 sets).
  * Frame-level and video-level (mean of frame scores per video) AUC.
  * 95% CIs by CLUSTER bootstrap over videos (frames of one video are correlated, so resampling frames
    would give falsely narrow intervals). 2000 resamples.
  * Paired comparison vs the deployed model: DeLong test (video-level) + paired cluster-bootstrap p-value.
  * Operating point: threshold set on validation at 5% false-positive rate, then test TPR / FPR reported at
    that fixed threshold (no peeking at test).
  * Calibration: ECE (15 bins) and Brier score, before and after temperature scaling fitted on validation.
Usage: python paper_stats.py name=ckpt [name=ckpt ...]   (first one is the reference, normally the deployed model)
Output: results/retrain/paper_stats.json and scores_<name>.npz
"""
import os, sys, csv, json, re
from collections import defaultdict
import numpy as np, torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score, roc_curve

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(REPO, "results", "retrain")
sys.path.insert(0, os.path.join(REPO, "models", "spatial"))
TF = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
B = 2000; RNG = np.random.default_rng(0)


class Files(Dataset):
    def __init__(self, paths): self.p = paths
    def __len__(self): return len(self.p)
    def __getitem__(self, i): return TF(Image.open(self.p[i]).convert("RGB"))


def data():
    sets = {}
    V = os.path.join(REPO, "data", "Dataset_Frames", "Validation"); it = []
    for l, d in ((0, "real"), (1, "fake")):
        for f in sorted(os.listdir(os.path.join(V, d))):
            it.append((os.path.join(V, d, f), l, "ff_" + re.sub(r"_frame\d+\.\w+$", "", f), "ff_real" if l == 0 else f.split("_")[0]))
    sets["ff_val"] = it
    D = os.path.join(REPO, "data", "benchmarks", "df40_x")
    rows = list(csv.DictReader(open(os.path.join(D, "index.csv"))))
    for sp in ("val", "test"):
        sets["df40_" + sp] = [(os.path.join(D, r["path"]), int(r["label"]), r["method"] + "/" + r["video"], r["method"]) for r in rows if r["split"] == sp]
    return sets


@torch.no_grad()
def score(ckpt, paths):
    from spatial_model import DeepfakeSpatialModel
    m = DeepfakeSpatialModel(); m.load_state_dict(torch.load(ckpt, map_location="cpu")); m = m.cuda().eval(); out = []
    for x in DataLoader(Files(paths), batch_size=96, num_workers=2):
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(m(x.cuda()).float().view(-1).cpu())
    return torch.cat(out).numpy()  # logits


def video_level(y, s, vid):
    g = defaultdict(list)
    for yi, si, vi in zip(y, s, vid): g[vi].append((yi, si))
    keys = sorted(g); return np.array([g[k][0][0] for k in keys]), np.array([np.mean([t[1] for t in g[k]]) for k in keys]), keys


def cluster_boot(y, s_list, vid):
    """Bootstrap over videos; returns array (B, len(s_list)) of AUCs on the same resamples (paired)."""
    g = defaultdict(list)
    for i, v in enumerate(vid): g[v].append(i)
    groups = [np.array(ix) for ix in g.values()]; out = []
    for _ in range(B):
        ix = np.concatenate([groups[j] for j in RNG.integers(0, len(groups), len(groups))])
        if len(set(y[ix])) < 2: continue
        out.append([roc_auc_score(y[ix], s[ix]) for s in s_list])
    return np.array(out)


def delong(y, a, b):
    """Paired DeLong test for two correlated AUCs (Sun & Xu 2014 fast version). Returns (auc_a, auc_b, p)."""
    def mid(x):
        J = np.argsort(x); Z = x[J]; N = len(x); T_ = np.zeros(N); i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]: j += 1
            T_[i:j] = 0.5 * (i + j - 1) + 1; i = j
        out = np.empty(N); out[J] = T_; return out
    pos, neg = y == 1, y == 0; m, n = pos.sum(), neg.sum(); aucs, V10, V01 = [], [], []
    for s in (a, b):
        tx, ty, tz = mid(s[pos]), mid(s[neg]), mid(s)
        aucs.append((tz[pos].sum() - m * (m + 1) / 2) / (m * n))
        V10.append((tz[pos] - tx) / n); V01.append(1 - (tz[neg] - ty) / m)
    S = np.cov(np.array(V10)) / m + np.cov(np.array(V01)) / n
    z = (aucs[0] - aucs[1]) / np.sqrt(max(S[0, 0] + S[1, 1] - 2 * S[0, 1], 1e-12))
    return aucs[0], aucs[1], float(2 * stats.norm.sf(abs(z)))


def ece(y, p, bins=15):
    e = np.linspace(0, 1, bins + 1); tot = 0.0
    for lo, hi in zip(e[:-1], e[1:]):
        k = (p > lo) & (p <= hi)
        if k.any(): tot += k.mean() * abs(y[k].mean() - p[k].mean())
    return float(tot)


def sig(z): return 1 / (1 + np.exp(-z))


def analyse(name, S, ref=None):
    r = {}
    yv = np.concatenate([S[k]["y"] for k in ("ff_val", "df40_val")]); zv = np.concatenate([S[k]["z"] for k in ("ff_val", "df40_val")])
    # threshold at 5% FPR on validation (frame level)
    thr = float(np.quantile(zv[yv == 0], 0.95))
    # temperature fitted on validation
    Tt = minimize_scalar(lambda t: -np.mean(yv * np.log(sig(zv / t) + 1e-9) + (1 - yv) * np.log(1 - sig(zv / t) + 1e-9)),
                         bounds=(0.05, 20), method="bounded").x
    r["val_threshold_logit_at_5pct_fpr"] = round(thr, 4); r["temperature"] = round(float(Tt), 4)
    for k in ("ff_val", "df40_val", "df40_test"):
        y, z, vid = S[k]["y"], S[k]["z"], S[k]["vid"]; p = sig(z)
        yv_, sv_, keys = video_level(y, z, vid)
        bt = cluster_boot(y, [z], vid)[:, 0]
        d = dict(n_frames=int(len(y)), n_videos=len(keys), auc_frame=round(roc_auc_score(y, z), 4),
                 auc_frame_ci95=[round(float(np.percentile(bt, 2.5)), 4), round(float(np.percentile(bt, 97.5)), 4)],
                 auc_video=round(roc_auc_score(yv_, sv_), 4),
                 tpr_at_val_thr=round(float((z[y == 1] > thr).mean()), 4), fpr_at_val_thr=round(float((z[y == 0] > thr).mean()), 4),
                 ece_raw=round(ece(y, p), 4), brier_raw=round(float(np.mean((p - y) ** 2)), 4),
                 ece_temp=round(ece(y, sig(z / Tt)), 4), brier_temp=round(float(np.mean((sig(z / Tt) - y) ** 2)), 4))
        if k == "df40_test":  # per test set, fake vs its own reals
            per = {}
            meth = np.array(S[k]["meth"])
            for mth in sorted(set(meth)):
                if mth.endswith("_real") or mth + "_real" not in set(meth): continue
                ix = (meth == mth) | (meth == mth + "_real")
                per[mth] = round(roc_auc_score(y[ix], z[ix]), 4)
            d["per_set_auc"] = per
        if ref is not None:
            Rz = ref[k]["z"]; assert np.array_equal(ref[k]["y"], y)
            ry, rs, _ = video_level(y, Rz, vid)
            _, _, p_dl = delong(yv_, sv_, rs)
            pb = cluster_boot(y, [z, Rz], vid); diff = pb[:, 0] - pb[:, 1]
            d["vs_reference"] = dict(delta_auc_frame=round(float(roc_auc_score(y, z) - roc_auc_score(y, Rz)), 4),
                                     delta_ci95=[round(float(np.percentile(diff, 2.5)), 4), round(float(np.percentile(diff, 97.5)), 4)],
                                     p_cluster_bootstrap=round(float(2 * min((diff <= 0).mean(), (diff >= 0).mean())), 4),
                                     p_delong_video=round(p_dl, 6))
        r[k] = d
    return r


if __name__ == "__main__":
    pairs = [a.split("=", 1) for a in sys.argv[1:]]
    sets = data(); allres = {}; ref = None
    for name, ck in pairs:
        cache = os.path.join(OUT, f"scores_{name}.npz")
        if os.path.exists(cache):
            c = np.load(cache); S = {k: dict(z=c[k]) for k in sets}
        else:
            S = {k: dict(z=score(os.path.join(REPO, ck), [i[0] for i in sets[k]])) for k in sets}
            np.savez(cache, **{k: S[k]["z"] for k in sets})
        for k in sets:
            S[k].update(y=np.array([i[1] for i in sets[k]]), vid=[i[2] for i in sets[k]], meth=[i[3] for i in sets[k]])
        allres[name] = analyse(name, S, ref); print(name, json.dumps(allres[name]["df40_test"].get("vs_reference", {})), flush=True)
        if ref is None: ref = S
    json.dump(allres, open(os.path.join(OUT, "paper_stats.json"), "w"), indent=1)
    print(json.dumps(allres, indent=1))
