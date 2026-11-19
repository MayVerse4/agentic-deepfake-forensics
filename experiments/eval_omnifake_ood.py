"""OmniFake-OOD (Omni-Fake, CVPR 2026; out-of-distribution test set, CC-BY-4.0): our detectors on a random sample
fetched by fetch_omnifake_ood.py. Images are 384x384 as released.
  image: REAL vs FULL_SYNTHETIC -> whole-image AI detectors (CF, B-Free, GAPL, combinations)
         REAL vs TAMPERED       -> edit detectors (TruFor, Mesorch, CF, B-Free combination)
  audio: REAL vs fake (FULL_SYNTHETIC + TAMPERED) -> voice detectors (mo-thecreator, Gustking, both)
  video: REAL vs fake -> Community Forensics (median of 8 frames), the app's video AI check
Two views: (1) "as deployed": the app's fixed weights (fitted on other data) -> AUC only, no refitting;
(2) the usual protocol (5 splits, refit on the tuning half, threshold for <=5% real flagged).
Run: .venv-win\\Scripts\\python.exe experiments\\eval_omnifake_ood.py [image|audio|video ...]
"""
import os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
D = os.path.join(os.path.dirname(HERE), "data", "benchmarks", "omnifake_ood"); B = os.path.join(os.path.dirname(HERE), "results", "benchmarks")
PARTS = sys.argv[1:] or ["image", "audio", "video"]
from experiments.make_scoreboard import protocol  # noqa: E402
from experiments.bench_record import save  # noqa: E402
sig = lambda z: 1 / (1 + np.exp(-z))
lines = []


def scored(cfg, cols, fn):
    """Per-file scores, cached in results/benchmarks/omnifake_<cfg>_scores.csv (resumable)."""
    lab = pd.read_csv(os.path.join(D, cfg, "labels.csv"))
    cache = os.path.join(B, f"omnifake_{cfg}_scores.csv")
    old = pd.read_csv(cache) if os.path.exists(cache) else pd.DataFrame(columns=["id"])
    todo = lab[~lab.id.isin(old.id)]
    if len(todo):
        rows = old.to_dict("records")
        for i, r in enumerate(todo.itertuples(), 1):
            try:
                rows.append({"id": r.id, **fn(os.path.join(D, cfg, r.file))})
            except Exception as e:
                print("  skip", r.file, type(e).__name__)
            if i % 50 == 0:
                pd.DataFrame(rows).to_csv(cache, index=False); print(f"  {cfg}: {i}/{len(todo)}", flush=True)
        pd.DataFrame(rows).to_csv(cache, index=False)
    return lab.merge(pd.read_csv(cache), on="id")


def report(title, df, methods, deployed):
    print(f"\n{title} ({int((df.label == 0).sum())} real / {int(df.label.sum())} fake)")
    lines.append(f"\n### {title}\n{int((df.label == 0).sum())} real / {int(df.label.sum())} fake\n\n"
                 "| Detectors | AUC (app weights, no refit) | AUC | Caught | Real flagged | Accuracy |\n|---|---|---|---|---|---|\n")
    out = {}
    for k, cols in methods.items():
        r = protocol(df, cols)
        dep = roc_auc_score(df.label, deployed[k](df)) if k in deployed else float("nan")
        out[k] = f"AUC(app) {dep:.3f}, AUC {r['auc']:.3f}, caught {r['caught']:.1f}%, flagged {r['flagged']:.1f}%, acc {r['acc']:.1f}%"
        print(f"  {k:30s} {out[k]}")
        lines.append(f"| {k} | {'' if np.isnan(dep) else f'{dep:.3f}'} | {r['auc']:.3f} | {r['caught']:.1f}% | {r['flagged']:.1f}% | {r['acc']:.1f}% |\n")
    save({"benchmark": f"OmniFake-OOD: {title}", **out}, B, f"omnifake_{title.split(':')[0].lower().replace(' ', '_')}_summary.json")


if "image" in PARTS:
    from detectors.ai_image_detector import AIImageDetector
    from detectors.bfree_detector import BFreeDetector
    from detectors.gapl_detector import GAPLDetector
    from detectors.localization import Localizer
    from detectors.mesorch_localizer import MesorchLocalizer
    cf, bf, gp, tf, me = AIImageDetector(), BFreeDetector(), GAPLDetector(), Localizer(), MesorchLocalizer()
    df = scored("image", None, lambda p: {"commfor": cf.score_image(p)["prob_ai"], "bfree": bf.score_image(p)["prob_ai"],
                                          "gapl": gp.score_image(p)["prob_ai"], "trufor": tf.analyze(p)["score"],
                                          "mesorch": me.analyze(p)["score"]})
    from detectors import backend_hooks as H
    ai = df[df.label.isin(["REAL", "FULL_SYNTHETIC"])].assign(label=lambda d: (d.label != "REAL").astype(int))
    report("AI image: REAL vs FULL_SYNTHETIC", ai,
           {"Community Forensics": ["commfor"], "B-Free": ["bfree"], "GAPL": ["gapl"],
            "CF + B-Free (app now)": ["commfor", "bfree"], "CF + B-Free + GAPL (proposed)": ["commfor", "bfree", "gapl"]},
           {"Community Forensics": lambda d: d.commfor, "B-Free": lambda d: d.bfree, "GAPL": lambda d: d.gapl,
            "CF + B-Free (app now)": lambda d: sig(H.AI_IMG_B + H.AI_IMG_W[0] * d.commfor + H.AI_IMG_W[1] * d.bfree),
            "CF + B-Free + GAPL (proposed)": lambda d: sig(-3.882 + 2.979 * d.commfor + 3.704 * d.bfree + 3.514 * d.gapl)})
    ed = df[df.label.isin(["REAL", "TAMPERED"])].assign(label=lambda d: (d.label != "REAL").astype(int))
    report("Edited image: REAL vs TAMPERED", ed,
           {"TruFor": ["trufor"], "Mesorch": ["mesorch"], "4 tools (app now)": ["trufor", "mesorch", "commfor", "bfree"],
            "4 tools + GAPL": ["trufor", "mesorch", "commfor", "bfree", "gapl"]},
           {"TruFor": lambda d: d.trufor, "Mesorch": lambda d: d.mesorch,
            "4 tools (app now)": lambda d: sig(H.COMBO_B + H.COMBO_W[0] * d.trufor + H.COMBO_W[1] * d.mesorch
                                               + H.COMBO_W[2] * d.commfor + H.COMBO_W[3] * d.bfree)})

if "audio" in PARTS:
    from detectors.voice_detector import VoiceDetector, VOICE_B, VOICE_W
    vd = VoiceDetector()
    df = scored("audio", None, lambda p: {k: v for k, v in vd.score_audio(p).items() if k in ("mo_score", "gustking_score")})
    df = df.assign(label=(df.label != "REAL").astype(int)).dropna(subset=["mo_score", "gustking_score"])
    report("Voice: REAL vs fake speech", df,
           {"mo-thecreator": ["mo_score"], "Gustking": ["gustking_score"], "mo + Gustking (app)": ["mo_score", "gustking_score"]},
           {"mo-thecreator": lambda d: d.mo_score, "Gustking": lambda d: d.gustking_score,
            "mo + Gustking (app)": lambda d: sig(VOICE_B + VOICE_W[0] * d.mo_score + VOICE_W[1] * d.gustking_score)})

if "video" in PARTS:
    from detectors.ai_image_detector import AIImageDetector
    cf = AIImageDetector()
    df = scored("video", None, lambda p: {"commfor": cf.score_video(p, 8)["prob_ai"]})
    df = df.assign(label=(df.label != "REAL").astype(int))
    report("Video: REAL vs fake video", df, {"Community Forensics, 8 frames (app)": ["commfor"]},
           {"Community Forensics, 8 frames (app)": lambda d: d.commfor})

with open(os.path.join(os.path.dirname(B), "RESULTS.md"), "a", encoding="utf8") as f:
    f.write("\n## OmniFake-OOD (Omni-Fake, CVPR 2026; out-of-distribution test sample)\n"
            "\"AUC (app weights)\" = the app's combination exactly as deployed (weights fitted on other datasets);"
            " the other columns refit on half of this data (5 splits, <=5% real flagged).\n" + "".join(lines))
print("\ntables added to results/RESULTS.md")
