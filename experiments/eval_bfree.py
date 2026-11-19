"""B-Free vs Community Forensics (our current whole-image AI detector), separate and combined.
  A) CommunityForensics-Eval sample: the SAME rows as eval_ai_detector.py (same seed/offsets), both detectors
     re-scored here so every image has both scores. NB: this is Community Forensics' own eval set (home ground).
  B) CocoGlide (AI inpainting): B-Free added as a 4th tool to TruFor + Mesorch + Community Forensics.
  C) GenVideo, format-normalized clips (genvideo_norm): median over 8 frames.
Combined = logistic regression; protocol as before: held-out half, 5 random splits, threshold for <=5% FPR.
Run: .venv-win\\Scripts\\python.exe experiments\\eval_bfree.py [A|B|C ...]
"""
import base64, io, json, os, random, sys, time, urllib.request
import cv2, numpy as np, pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
OUT = os.path.join(os.path.dirname(HERE), "results", "benchmarks")
PARTS = sys.argv[1:] or ["A", "B", "C"]
from detectors.bfree_detector import BFreeDetector
from experiments.bench_record import save
bf = BFreeDetector()
_cf = None


def cf():
    global _cf
    if _cf is None:
        from detectors.ai_image_detector import AIImageDetector
        _cf = AIImageDetector()
    return _cf


def at_fpr(y, s, ty, ts, max_fpr=0.05):
    fpr, _, th = roc_curve(ty, ts); t = th[np.where(fpr <= max_fpr)[0][-1]]
    return ((s > t) & (y == 1)).sum() / (y == 1).sum(), ((s > t) & (y == 0)).sum() / (y == 0).sum(), t


def compare(df, methods, title, fname, group_col=None):
    res = {k: [] for k in methods}; per = {k: {} for k in methods}
    for seed in range(5):
        d = df.sample(frac=1, random_state=seed); a, b = d.iloc[:len(d) // 2], d.iloc[len(d) // 2:]
        for k, cols in methods.items():
            if len(cols) == 1:
                sa, sb = a[cols[0]].values, b[cols[0]].values
            else:
                lr = LogisticRegression(max_iter=1000).fit(a[cols], a.label)
                sa, sb = lr.predict_proba(a[cols])[:, 1], lr.predict_proba(b[cols])[:, 1]
            c, f, t = at_fpr(b.label.values, sb, a.label.values, sa)
            res[k].append((roc_auc_score(b.label, sb), c, f))
            if group_col:
                for g, gb in b[b.label == 1].assign(s=sb[b.label.values == 1]).groupby(group_col):
                    per[k].setdefault(g, []).append((gb.s > t).mean())
    print(f"\n{title}\n{'method':40s} {'AUC':>6s} {'caught':>8s} {'real flagged':>13s}")
    summary = {"benchmark": title}
    for k, v in res.items():
        v = np.array(v)
        print(f"{k:40s} {v[:, 0].mean():6.3f} {v[:, 1].mean() * 100:7.1f}% {v[:, 2].mean() * 100:12.1f}%")
        summary[k] = f"AUC {v[:, 0].mean():.3f}, caught {v[:, 1].mean() * 100:.1f}%, real flagged {v[:, 2].mean() * 100:.1f}%"
    if group_col:
        ks = list(methods)
        print(f"\ncaught per {group_col} (AI images only):\n{'':24s}" + "".join(f"{k[:22]:>24s}" for k in ks))
        for g in sorted(per[ks[0]], key=lambda g: -df[df[group_col] == g].shape[0]):
            print(f"{str(g)[:24]:24s}" + "".join(f"{np.mean(per[k][g]) * 100:23.1f}%" for k in ks))
        summary["per_group_caught_pct"] = {str(g): [round(float(np.mean(per[k][g]) * 100), 1) for k in ks] for g in per[ks[0]]}
    save(summary, OUT, fname)


# ---------------- A) CommunityForensics-Eval sample (same rows as eval_ai_detector.py) ----------------
if "A" in PARTS:
    cache = os.path.join(OUT, "bfree_commforeval_scores.csv")
    if not os.path.exists(cache):
        URL = ("https://datasets-server.huggingface.co/rows?dataset=OwensLab/CommunityForensics-Eval"
               "&config=default&split=CompEval&offset={o}&length={n}")
        random.seed(0); offsets = sorted(random.sample(range(0, 51836 - 15), 70))
        part = cache + ".partial.csv"
        rows = pd.read_csv(part).to_dict("records") if os.path.exists(part) else []
        start = rows[-1]["chunk"] if rows else 0
        for i, off in enumerate(offsets, 1):
            if i <= start:
                continue
            for k in range(4):
                try:
                    with urllib.request.urlopen(URL.format(o=off, n=15), timeout=60) as r:
                        got = json.load(r)["rows"]; break
                except Exception:
                    got = []; time.sleep(3 * (k + 1))
            for r in got:
                row = r["row"]; v = row["image_data"]
                try:
                    img = (Image.open(io.BytesIO(urllib.request.urlopen(v["src"], timeout=60).read())) if isinstance(v, dict)
                           else Image.open(io.BytesIO(base64.b64decode(v.strip('"'))))).convert("RGB")
                except Exception:
                    continue
                try:
                    sc = {"commfor": cf().score_image(img)["prob_ai"], "bfree": bf.score_image(img)["prob_ai"]}
                except ValueError:
                    continue                          # degenerate image (e.g. 1x1)
                rows.append({"chunk": i, "model_name": row["model_name"], "label": int(row["label"]), **sc})
            if i % 5 == 0:
                print(f"  A: {i}/70 chunks, {len(rows)} images", flush=True)
                pd.DataFrame(rows).to_csv(part, index=False)
        pd.DataFrame(rows).to_csv(cache, index=False)
    df = pd.read_csv(cache)
    compare(df, {"Community Forensics alone (current)": ["commfor"], "B-Free alone": ["bfree"],
                 "Both combined": ["commfor", "bfree"]},
            f"AI images: CommunityForensics-Eval sample ({len(df)} images, {df[df.label == 1].model_name.nunique()} generators)",
            "bfree_commforeval_summary.json", group_col="model_name")

# ---------------- B) CocoGlide: B-Free as 4th tool in the edit combiner ----------------
if "B" in PARTS:
    cache = os.path.join(OUT, "cocoglide_bfree_scores.csv")
    loc = pd.read_csv(os.path.join(OUT, "localization_eval.csv"))[["file", "label", "score"]].rename(columns={"score": "TruFor"})
    if not os.path.exists(cache):
        rows = [{"file": r.file, "bfree": bf.score_image(os.path.join(os.path.dirname(HERE), "data", "benchmarks", "fake" if r.label else "real", r.file))["prob_ai"]}
                for r in loc.itertuples()]
        pd.DataFrame(rows).to_csv(cache, index=False)
    df = loc.merge(pd.read_csv(cache), on="file")
    df = df.merge(pd.read_csv(os.path.join(OUT, "cocoglide_ai_scores.csv")).rename(columns={"prob_ai": "commfor"}), on="file")
    df = df.merge(pd.read_csv(os.path.join(OUT, "mesorch_cocoglide_eval.csv"))[["file", "score"]].rename(columns={"score": "Mesorch"}), on="file")
    compare(df, {"B-Free alone": ["bfree"], "Community Forensics alone": ["commfor"],
                 "3 tools (current app)": ["TruFor", "Mesorch", "commfor"],
                 "3 tools + B-Free": ["TruFor", "Mesorch", "commfor", "bfree"]},
            "AI-inpainted edits: CocoGlide (512 real + 512 GLIDE-inpainted)", "bfree_cocoglide_summary.json")

# ---------------- C) GenVideo, format-normalized ----------------
if "C" in PARTS:
    cache = os.path.join(OUT, "genvideo_norm_bfree_scores.csv")
    base = pd.read_csv(os.path.join(OUT, "genvideo_genvideo_norm_ai_scores.csv"))
    meta = pd.read_csv(os.path.join(OUT, "nsgvd_standard-Pika-d_genvideo_norm_eval.csv"))[["id", "generator", "label"]]
    done = pd.read_csv(cache) if os.path.exists(cache) else pd.DataFrame(columns=["id", "bfree"])
    todo = base[~base.id.isin(done.id)].id
    if len(todo):
        rows = []
        for i, vid in enumerate(todo, 1):
            cap = cv2.VideoCapture(os.path.join(os.path.dirname(HERE), "data", "benchmarks", "genvideo_norm", vid + ".mp4"))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1; ps = []
            for idx in np.linspace(0, n - 1, 8).astype(int):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx)); ok, f = cap.read()
                if ok:
                    ps.append(bf.score_image(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))["prob_ai"])
            cap.release()
            rows.append({"id": vid, "bfree": float(np.median(ps)) if ps else np.nan})
            if i % 100 == 0 or i == len(todo):
                print(f"  C: {len(done) + i}/{len(base)}", flush=True)
                pd.concat([done, pd.DataFrame(rows)]).to_csv(cache, index=False)
    df = meta.merge(base.rename(columns={"prob_ai": "commfor"}), on="id").merge(pd.read_csv(cache), on="id").dropna()
    compare(df, {"Community Forensics alone (current)": ["commfor"], "B-Free alone": ["bfree"],
                 "Both combined": ["commfor", "bfree"]},
            f"AI videos: GenVideo, format-normalized ({int((df.label == 0).sum())} real, {int(df.label.sum())} AI)",
            "bfree_genvideo_summary.json", group_col="generator")
