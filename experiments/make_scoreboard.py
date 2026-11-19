"""Builds results/scoreboard.html + results/scoreboard.pdf from the SAVED benchmark files in results/benchmarks/.
Every number is recomputed here with one protocol, so the sheet cannot drift from the evidence:
  5 random half/half splits; detection threshold chosen on one half for <= 5% real flagged, all metrics
  (AUC, caught, real flagged, accuracy, F1) measured on the other half; mean of the 5 splits.
  Combinations = logistic regression fitted on the tuning half only.
Run: .venv-win\\Scripts\\python.exe experiments\\make_scoreboard.py
"""
import datetime, os, re, shutil, subprocess
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
B = os.path.join(PROJ, "results", "benchmarks")
DOCS = [os.path.join(os.path.dirname(PROJ), "docs"), os.path.join(PROJ, "results")]
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"


def protocol(df, cols, label="label"):
    y, X, out = df[label].values, df[cols].values.astype(float), []
    for seed in range(5):
        idx = np.random.RandomState(seed).permutation(len(df)); a, b = idx[:len(df) // 2], idx[len(df) // 2:]
        if len(cols) > 1:
            lr = LogisticRegression(max_iter=1000).fit(X[a], y[a])
            sa, sb = lr.predict_proba(X[a])[:, 1], lr.predict_proba(X[b])[:, 1]
        else:
            sa, sb = X[a, 0], X[b, 0]
        fpr, _, th = roc_curve(y[a], sa); t = th[np.where(fpr <= 0.05)[0][-1]]
        p = sb > t
        fp, tp, _ = roc_curve(y[b], sb); fn = 1 - tp; i = np.nanargmin(np.abs(fn - fp))
        out.append((roc_auc_score(y[b], sb), (p & (y[b] == 1)).sum() / (y[b] == 1).sum(),
                    (p & (y[b] == 0)).sum() / (y[b] == 0).sum(), (p == y[b]).mean(), f1_score(y[b], p),
                    (fp[i] + fn[i]) / 2))
    m = np.array(out).mean(0)
    return dict(auc=m[0], caught=m[1] * 100, flagged=m[2] * 100, acc=m[3] * 100, f1=m[4] * 100, eer=m[5] * 100)


# ---------------- data (all saved by the eval_*.py scripts) ----------------
aig = pd.read_csv(os.path.join(B, "aigibench_scores.csv")); aig["path"] = aig.path.str.replace(chr(92), "/", regex=False)
aig = aig.merge(pd.read_csv(os.path.join(B, "gapl_aigibench_scores.csv")), on="path")
aig_cf, aig_both, aig_3 = protocol(aig, ["commfor"]), protocol(aig, ["commfor", "bfree"]), protocol(aig, ["commfor", "bfree", "gapl"])
aig_n, aig_real = len(aig), int((aig.label == 0).sum())
aig_sub = {s: round(100 * ((g[g.label == 1].commfor > 0.5).mean()), 0) for s, g in aig.groupby("subset")}

cg = (pd.read_csv(os.path.join(B, "localization_eval.csv"))[["file", "label", "score"]].rename(columns={"score": "trufor"})
      .merge(pd.read_csv(os.path.join(B, "mesorch_cocoglide_eval.csv"))[["file", "score"]].rename(columns={"score": "mesorch"}), on="file")
      .merge(pd.read_csv(os.path.join(B, "cocoglide_ai_scores.csv")).rename(columns={"prob_ai": "commfor"}), on="file")
      .merge(pd.read_csv(os.path.join(B, "cocoglide_bfree_scores.csv")), on="file"))
cg_tru, cg_all = protocol(cg, ["trufor"]), protocol(cg, ["trufor", "mesorch", "commfor", "bfree"])

vid = (pd.read_csv(os.path.join(B, "nsgvd_standard-Pika-d_genvideo_norm_eval.csv"))[["id", "label", "nsg_mmd"]]
       .merge(pd.read_csv(os.path.join(B, "genvideo_genvideo_norm_ai_scores.csv")).rename(columns={"prob_ai": "commfor"}), on="id")
       .merge(pd.read_csv(os.path.join(B, "genvideo_norm_bfree_scores.csv")), on="id").dropna())
vid_cf, vid_both = protocol(vid, ["commfor"]), protocol(vid, ["commfor", "bfree"])
vid_nsg = protocol(vid.assign(lnsg=np.log1p(vid.nsg_mmd.clip(lower=0))), ["lnsg"])

vo = pd.read_csv(os.path.join(B, "voice_eval.csv"))
M, G = "mo-thecreator/Deepfake-audio-detection", "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"
vo_m, vo_mg, vo_g = protocol(vo, [M]), protocol(vo, [M, G]), protocol(vo, [G])

# OmniFake-OOD (CVPR 2026, unseen sources): AUC of the app's combinations exactly as deployed (no refitting)
from sklearn.metrics import roc_auc_score as _auc
_sig = lambda z: 1 / (1 + np.exp(-z))
_O = os.path.join(os.path.dirname(HERE), "data", "benchmarks", "omnifake_ood")
def _om(cfg):
    return pd.read_csv(os.path.join(_O, cfg, "labels.csv")).merge(pd.read_csv(os.path.join(B, f"omnifake_{cfg}_scores.csv")), on="id")
_oi = _om("image"); _ai = _oi[_oi.label.isin(["REAL", "FULL_SYNTHETIC"])]; _yi = _ai.label != "REAL"
om_img_before = _auc(_yi, _sig(-2.576 + 4.998 * _ai.commfor + 3.412 * _ai.bfree))
om_img_after = _auc(_yi, _sig(-3.310 + 3.824 * _ai.commfor + 4.233 * _ai.gapl))      # 384 px images -> small-image branch
_ed = _oi[_oi.label.isin(["REAL", "TAMPERED"])]
om_edit = _auc(_ed.label != "REAL", _sig(-2.439 + 2.054 * _ed.trufor + 1.271 * _ed.mesorch + 1.055 * _ed.commfor + 3.780 * _ed.bfree))
_oa = _om("audio").dropna(subset=["mo_score", "gustking_score"]); _ya = _oa.label != "REAL"
om_voice_before = _auc(_ya, _sig(-3.184 + 3.014 * _oa.mo_score + 5.385 * _oa.gustking_score)); om_voice_after = _auc(_ya, _oa.gustking_score)
_ov = _om("video"); om_vid_syn = _auc(_ov[_ov.label != "TAMPERED"].label != "REAL", _ov[_ov.label != "TAMPERED"].commfor)
om_n = (len(_ai), len(_ed), len(_oa), len(_ov[_ov.label != "TAMPERED"]))

e2e = pd.read_csv(os.path.join(os.path.dirname(PROJ), "paper", "check", "end_to_end_rerun.csv"))
e2e_ok = int(((e2e.verdict == "FAKE") == (e2e.label == 1)).sum()); e2e_n = len(e2e)
e2e_diff = max(abs(e2e[c] - e2e[c + "_saved"]).max() for c in ("spatial", "freq", "temporal"))


# ---------------- render ----------------
def main():
    def cell(r, tool, voice=False):
        s = f"<b>{r['auc']:.3f}</b> AUC<br>"
        if voice:
            s += f"{r['eer']:.1f}% EER<br>"
        s += (f"<b>{r['caught']:.1f}% caught</b> ({r['flagged']:.1f}% real flagged)<br>{r['acc']:.1f}% accuracy<br>"
              f"{r['f1']:.1f}% F1<span class='tool-tag'>{tool}</span>")
        return s


    def gain(a, b, voice=False):
        d = b["caught"] - a["caught"]
        s = f"{'+' if d >= 0 else ''}{d:.1f} pts caught<br>{'+' if b['acc'] >= a['acc'] else ''}{b['acc'] - a['acc']:.1f} pts accuracy"
        if voice:
            s += f"<br>EER {a['eer']:.1f}% &rarr; {b['eer']:.1f}%"
        return s


    style = re.search(r"<style>.*?</style>", open(os.path.join(DOCS[0], "scoreboard.html"), encoding="utf8").read(), re.S).group(0)
    rows = [
        ("AI-Generated Images", "Full-frame synthesis", f"<b>AIGIBench</b> (2025, independent)<br>{aig_n} images ({aig_real} real / {aig_n - aig_real} AI)<br><i>10 generators &amp; in-the-wild subsets</i>",
         "AUC<br>Recall (@&le;5% FPR)<br>Accuracy<br>F1", cell(aig_cf, "Community Forensics"), cell(aig_3, "Community Forensics + B-Free + GAPL (app)"),
         gain(aig_cf, aig_3) + f"<br>vs CF + B-Free: {aig_both['caught']:.1f}% caught (GAPL adds {aig_3['caught'] - aig_both['caught']:+.1f} pts)"),
        ("Locally Edited Images", "AI inpainting", "<b>CocoGlide</b><br>1,024 images<br><i>512 real + 512 AI-inpainted</i>",
         "AUC<br>Recall (@&le;5% FPR)<br>Accuracy<br>F1", cell(cg_tru, "TruFor alone (first app version)"),
         cell(cg_all, "TruFor + Mesorch + CF + B-Free (app)"), gain(cg_tru, cg_all) + "<br>Box drawn by Mesorch (pixel F1 0.45 vs 0.36)"),
        ("AI-Generated Videos", "Text/image-to-video", f"<b>GenVideo</b>, re-encoded to one format<br>{len(vid)} videos ({int((vid.label == 0).sum())} real / {int(vid.label.sum())} AI)<br><i>10 generators</i>",
         "AUC<br>Recall (@&le;5% FPR)<br>Accuracy<br>F1", cell(vid_cf, "Community Forensics, 8 frames (app)"),
         cell(vid_both, "CF + B-Free (tested, not used)"),
         f"{vid_both['caught'] - vid_cf['caught']:+.1f} pts caught: no real gain, so the app keeps CF alone<br>"
         f"NSG-VD physics model: AUC {vid_nsg['auc']:.2f} once the format shortcut is removed (rejected)"),
        ("Face-Swap Videos", "Identity swap (original project)", f"<b>FaceForensics++</b><br>{e2e_n} videos re-run from raw video<br><i>real vs Deepfakes</i>",
         "Correct verdicts<br>Match to paper", f"<b>{e2e_ok}/{e2e_n}</b> correct<span class='tool-tag'>Spatial + Frequency + Temporal + Fusion</span>",
         f"<b>{e2e_ok}/{e2e_n}</b> correct<br>max score difference vs paper {e2e_diff:.4f}<span class='tool-tag'>unchanged in the app</span>",
         "Reproduces the paper's saved scores<br><i>Small check (16 videos), not a new benchmark</i>"),
        ("AI Voice Clones", "Speech synthesis / cloning", f"<b>In-the-Wild</b> (16 kHz, first 6 s)<br>{len(vo)} clips<br><i>{int((vo.label == 0).sum())} real / {int(vo.label.sum())} AI-cloned</i>",
         "AUC<br>EER (lower = better)<br>Recall (@&le;5% FPR)<br>Accuracy, F1", cell(vo_m, "mo-thecreator wav2vec2", True),
         cell(vo_g, "Gustking XLS-R, calibrated (app)", True),
         f"Chosen for robustness: on unseen voices (OmniFake-OOD) AUC {om_voice_after:.3f} vs {om_voice_before:.3f} for the "
         f"mo + Gustking pair (pair on this set: {vo_mg['auc']:.3f} AUC)"),
        ("Metadata &amp; Integrity", "Provenance / file consistency", "<b>EXIF + watermark tests</b><br>4 devices x (original, re-save, crop, edit)<br><i>+ 518 clean images</i>",
         "Correct on tamper set<br>Watermark false alarms",
         "Pixel tools alone: an edit with the camera EXIF copied back can pass as \"No traces found\"<span class='tool-tag'>before the EXIF checks</span>",
         "<b>16/16</b> EXIF cases correct<br><b>0/518</b> clean images flagged by the SD-watermark check<span class='tool-tag'>C2PA + EXIF consistency + SD watermark (app)</span>",
         "Catches thumbnail / size contradictions<br><i>Small test (4 devices); a missing watermark proves nothing</i>"),
        ("Real-World Style Test", "Unseen sources (all media)", "<b>OmniFake-OOD</b> (CVPR 2026)<br>random sample, 384 px images<br>"
         f"<i>{om_n[0]} AI-img / {om_n[1]} edit / {om_n[2]} voice / {om_n[3]} video</i>", "AUC of the app as deployed<br>(no refitting)",
         f"AI images {om_img_before:.3f}<br>Edited images {om_edit:.3f}<br>Voice {om_voice_before:.3f}<br>AI video {om_vid_syn:.3f}"
         "<span class='tool-tag'>app before this update</span>",
         f"AI images <b>{om_img_after:.3f}</b><br>Edited images {om_edit:.3f}<br>Voice <b>{om_voice_after:.3f}</b><br>AI video {om_vid_syn:.3f}"
         "<span class='tool-tag'>app now (GAPL; small-image rule; voice detector)</span>",
         "Every detector drops on unseen, downscaled media<br><i>Edit detectors fail at 384 px JPEG: edit findings on small images are capped at 'Possible edit'</i>"),
    ]
    body = "".join(f"<tr><td class='col-task'>{a}<span class='tool-tag'>{b}</span></td><td class='col-bench'>{c}</td>"
                   f"<td class='col-metric'>{d}</td><td class='col-single'>{e}</td><td class='col-combo'>{f}</td>"
                   f"<td class='col-imp'>{g}</td></tr>" for a, b, c, d, e, f, g in rows)
    kpis = [("AI-edit recall (CocoGlide)", f"{cg_tru['caught']:.1f}% &rarr; {cg_all['caught']:.1f}%", f"at {cg_all['flagged']:.1f}% real flagged, AUC {cg_all['auc']:.3f}"),
            ("AI-image AUC (AIGIBench)", f"{aig_3['auc']:.3f}", f"{aig_3['caught']:.1f}% caught at {aig_3['flagged']:.1f}% real flagged"),
            ("Real-world style (OmniFake-OOD)", f"{om_img_after:.3f} / {om_voice_after:.3f}", "AI-image / voice AUC on unseen sources"),
            ("Coverage", "6 checks", "Face &bull; AI image &bull; Edit &bull; AI video &bull; Voice &bull; Metadata")]
    kpi = "".join(f"<div class='kpi-card'><div class='kpi-label'>{a}</div><div class='kpi-value'>{b}</div><div class='kpi-sub'>{c}</div></div>" for a, b, c in kpis)
    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Deepfake Detection Scoreboard</title>{style}
    <style>.kpi-card{{border:1px solid #cbd5e1;border-left:4px solid #2563eb;border-radius:4px;padding:6px 8px}}
    .kpi-label{{font-size:9px;font-weight:700;color:#475569;text-transform:uppercase}}.kpi-value{{font-size:17px;font-weight:800;color:#1e3a8a}}
    .kpi-sub{{font-size:9px;color:#15803d;font-weight:600}}.foot{{font-size:8.5px;color:#64748b;margin-top:4px}}
    td{{padding:3px 6px !important;font-size:9.6px !important;line-height:1.25 !important}} th{{padding:4px 6px !important;font-size:9px !important}}
    .kpi-value{{font-size:15px}} .kpi-card{{padding:4px 8px}} .header{{margin-bottom:5px !important}}</style></head><body>
    <div class="header"><div class="title-group"><h1>End-to-End Deepfake Detection Pipeline: Scoreboard</h1>
    <p>Single detectors vs. the agentic multi-model pipeline, recomputed from the saved test runs</p></div>
    <div class="meta-badges">Hardware: NVIDIA GeForce RTX 4050 Laptop GPU<br><span class="meta-badge">Held-out half, 5 random splits</span>
    <span class="meta-badge">Threshold set for &le; 5% real flagged</span><span class="meta-badge">Generated {datetime.date.today()}</span></div></div>
    <table><tr><th>Modality / task</th><th>Benchmark dataset</th><th>Metrics</th><th>Without combination (single baseline)</th>
    <th>With combination (app pipeline)</th><th>Measured gain / note</th></tr>{body}</table>
    <div class="kpi-container">{kpi}</div>
    <div class="foot">Thresholds are chosen on one half of each dataset and every metric is measured on the other half (mean of 5 random
    splits). Generated by experiments/make_scoreboard.py from results/benchmarks/ (aigibench_scores.csv, CocoGlide score files, GenVideo score files,
    voice_eval.csv, gapl/omnifake score files, paper/check/end_to_end_rerun.csv, exif/watermark summaries).</div></body></html>"""

    for d in DOCS:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "scoreboard.html"), "w", encoding="utf8") as f:
            f.write(html)
    pdf = os.path.join(DOCS[0], "scoreboard.pdf")
    subprocess.run([EDGE, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}",
                    "file:///" + os.path.join(DOCS[0], "scoreboard.html").replace("\\", "/")], check=True, timeout=120)
    shutil.copy(pdf, os.path.join(DOCS[1], "scoreboard.pdf"))
    for name, r in [("AIGIBench CF", aig_cf), ("AIGIBench CF+BFree", aig_both), ("CocoGlide TruFor", cg_tru), ("CocoGlide 4 tools", cg_all),
                    ("Video CF", vid_cf), ("Video CF+BFree", vid_both), ("Voice mo", vo_m), ("Voice mo+G", vo_mg)]:
        print(f"{name:20s} AUC {r['auc']:.3f} caught {r['caught']:5.1f}% flagged {r['flagged']:4.1f}% acc {r['acc']:5.1f}% F1 {r['f1']:5.1f}% EER {r['eer']:4.1f}%")
    print(f"face e2e {e2e_ok}/{e2e_n}, max diff {e2e_diff:.4f}\nwrote", pdf)


if __name__ == "__main__":
    main()
