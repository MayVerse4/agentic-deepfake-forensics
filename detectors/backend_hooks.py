"""
Glue between the new tools in this folder and the existing backend (backend/forensics_api.py).
Keeps the backend change small; all new logic and thresholds live here.

Tools
  TruFor (localization.py)            -> edited REGIONS anywhere in an image / video frame
  Community Forensics (ai_image_detector.py) -> is the WHOLE image AI-generated?
Both work from pixels only, so stripped metadata does not matter.
  SPAI (spai_detector.py), GAPL (gapl_detector.py) -> extra AI-image detectors: NOT used in decisions until
      benchmarked (eval_aigibench.py); see results/RESULTS.md
  Voice (voice_detector.py)           -> AI-cloned voice in audio files (wav2vec2 pair, In-the-Wild calibrated)
"""
import threading

import cv2
import numpy as np

# ---- decision thresholds -------------------------------------------------------------
EDIT_STRONG = 0.80       # TruFor score >= : "Edited region detected"
EDIT_WEAK = 0.50         # TruFor score >= : "Possible edit"
MIN_RELIABILITY = 0.70   # TruFor's own confidence; below this the heatmap is not trusted
AI_STRONG = 0.85         # Community Forensics P(AI) >= : "Likely AI-generated"
AI_WEAK = 0.50           # >= : "Possibly AI-generated"
VIDEO_LOC_FRAMES = 4     # key frames checked by TruFor in a video
# TruFor is trained on still photos; on compressed video frames weak scores are unreliable
# (measured: a real broadcast video scored 0.54 on its TV-graphics overlay, a face-swap video 0.48).
# For video, only a STRONG score is reported as evidence.
VIDEO_EDIT_MIN = EDIT_STRONG
VIDEO_AI_FRAMES = 8      # frames scored by the AI-image detector in a video

# ---- images: TruFor + Mesorch + AI detector combined (eval_combined_all.py) ----------------
# Logistic regression fitted on CocoGlide (512 real + 512 AI-inpainted). Thresholds chosen on
# 10-fold out-of-fold scores: 0.78 -> 47% of AI edits caught at 3% real flagged; 0.70 -> 52% at 5%.
# (TruFor alone: 29% at 5%.) Mesorch draws the region outline (pixel F1 0.45 vs TruFor 0.36).
# Updated with B-Free (Guillaro et al., CVPR 2025) as a 4th tool (eval_bfree.py): 10-fold out-of-fold thresholds
# 0.79 -> 62% of AI edits caught at 3% real flagged; 0.93 -> 52% at 1%; 0.56 -> 70% at 5%.
COMBO_W = (2.054, 1.271, 1.055, 3.780)   # TruFor, Mesorch, Community Forensics P(AI), B-Free P(AI)
COMBO_B = -2.439
COMBO_STRONG = 0.79
COMBO_VERY_STRONG = 0.93                 # ~1% real flagged -> High confidence
COMBO_WEAK = 0.56

# ---- images: whole-image AI = Community Forensics + B-Free (CommunityForensics-Eval sample, 899 images, 17 gens) --
# 10-fold out-of-fold: 0.78 -> 81% of AI images caught at 1% real flagged; 0.62 -> 90% at 5%.
# (Community Forensics alone: 88% at 6%.) Videos keep Community Forensics alone (B-Free added nothing on video).
AI_IMG_W = (4.998, 3.412)                # Community Forensics, B-Free (fallback when GAPL is unavailable)
AI_IMG_B = -2.576
AI_IMG_STRONG = 0.78
AI_IMG_WEAK = 0.62
# + GAPL (Qin et al., CVPR 2026), fitted on AIGIBench (998 images, 10-fold out-of-fold thresholds):
#   CF + B-Free + GAPL: 0.884 -> 83.6% caught at 1% real flagged; 0.422 -> 95.4% at 5%   (app before: 81% / 90%)
#   OmniFake-OOD (unseen): AUC 0.844 vs 0.756 for CF + B-Free.
AI3_W, AI3_B, AI3_STRONG, AI3_WEAK = (2.979, 3.704, 3.514), -3.882, 0.884, 0.422
# Small images (shorter side < SMALL_SIDE): B-Free needs 504 px crops and fails on small images (OmniFake-OOD
# 384 px: AUC 0.53), so it is left out: CF + GAPL, 0.825 -> 74.8% at 1%; 0.684 -> 91.0% at 5%; OmniFake AUC 0.885.
AI2_W, AI2_B, AI2_STRONG, AI2_WEAK = (3.824, 4.233), -3.310, 0.825, 0.684
# Edit detectors read camera-noise traces that downscaling and re-compression destroy: on OmniFake-OOD
# (384 px JPEG) TruFor / Mesorch scored AUC 0.53 / 0.51 (CocoGlide, full size: 0.74). On small images an edit
# finding is therefore never reported above "Possible edit".
SMALL_SIDE = 512
WHOLE_IMAGE_PCT = 60.0                   # Mesorch "region" covering more than this = whole picture, not a local edit

_tools, _errors = {}, {}

_lock = threading.Lock()


def _get(name):
    with _lock:
        if name not in _tools and name not in _errors:
            try:
                if name == "trufor":
                    from detectors.localization import Localizer
                    _tools[name] = Localizer()
                elif name == "mesorch":
                    from detectors.mesorch_localizer import MesorchLocalizer
                    _tools[name] = MesorchLocalizer()
                elif name == "bfree":
                    from detectors.bfree_detector import BFreeDetector
                    _tools[name] = BFreeDetector()
                elif name == "gapl":
                    from detectors.gapl_detector import GAPLDetector
                    _tools[name] = GAPLDetector()
                elif name == "voice":
                    from detectors.voice_detector import VoiceDetector
                    _tools[name] = VoiceDetector()
                else:
                    from detectors.ai_image_detector import AIImageDetector
                    _tools[name] = AIImageDetector()
            except Exception as e:      # weights missing, no internet on first run, GPU issue ... never break a job
                _errors[name] = f"{type(e).__name__}: {e}"
    return _tools.get(name)


def status():
    return {"trufor": "trufor" in _tools or "trufor" not in _errors, "ai_detector": "ai" in _tools or "ai" not in _errors,
            "mesorch": "mesorch" in _tools or "mesorch" not in _errors,
            "bfree": "bfree" in _tools or "bfree" not in _errors, "gapl": "gapl" in _tools or "gapl" not in _errors, "voice": "voice" in _tools or "voice" not in _errors, "errors": dict(_errors)}


# ---- localization ----------------------------------------------------------------------
def _video_frames(path, n):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out = []
    for idx in np.linspace(0, total - 1, min(n, total) + 2)[1:-1].astype(int):   # skip first/last frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, f = cap.read()
        if ok:
            out.append((round(idx / fps, 2), f))
    cap.release()
    return out


def localize(path, media_type="image"):
    loc = _get("trufor")
    if loc is None:
        return None
    try:
        if media_type == "video":
            best = None
            per_frame = []
            for t, frame in _video_frames(path, VIDEO_LOC_FRAMES):
                r = loc.analyze(frame, box_min=VIDEO_EDIT_MIN)
                per_frame.append({"t": t, "score": r["score"]})
                if best is None or r["score"] > best["score"]:
                    best = dict(r, t=t)
            if best is None:
                return None
            best["frames"] = per_frame
        else:
            best = loc.analyze(path)
    except Exception:
        return None
    if best and media_type == "video":
        best.pop("_heat", None)
    return best


def mesorch_score(path):
    """Images only: Mesorch heat map + image score (mean of top 1% pixels), or None."""
    m = _get("mesorch")
    if m is None:
        return None
    try:
        return m.analyze(path)
    except Exception:
        return None


def bfree_score(path):
    """Images only: B-Free P(AI-generated), or None."""
    b = _get("bfree")
    if b is None:
        return None
    try:
        return b.score_image(path)["prob_ai"]
    except Exception:
        return None


def combine(loc, mes, ai):
    """Learned 4-tool edit score for images; None when any tool is missing."""
    if not (loc and mes and ai and ai.get("bfree") is not None):
        return None
    z = (COMBO_B + COMBO_W[0] * loc["score"] + COMBO_W[1] * mes["score"] + COMBO_W[2] * ai["commfor"]
         + COMBO_W[3] * ai["bfree"])
    return float(1 / (1 + np.exp(-z)))


def gapl_score(path):
    """Images only: GAPL P(AI-generated), or None."""
    g = _get("gapl")
    if g is None:
        return None
    try:
        return g.score_image(path)["prob_ai"]
    except Exception:
        return None


def ai_image_combined(ai, bf, gp=None, small=False):
    """Images: one calibrated P(AI-generated) from the available detectors, with its own thresholds ("thr").
    Large images: CF + B-Free + GAPL. Small images: CF + GAPL (B-Free is unreliable below 504 px)."""
    if ai is None:
        return ai
    cf = ai["prob_ai"]
    if gp is not None and small:
        z, thr, used = AI2_B + AI2_W[0] * cf + AI2_W[1] * gp, (AI2_STRONG, AI2_WEAK), "Community Forensics + GAPL"
    elif gp is not None and bf is not None:
        z = AI3_B + AI3_W[0] * cf + AI3_W[1] * bf + AI3_W[2] * gp
        thr, used = (AI3_STRONG, AI3_WEAK), "Community Forensics + B-Free + GAPL"
    elif bf is not None:
        z, thr, used = AI_IMG_B + AI_IMG_W[0] * cf + AI_IMG_W[1] * bf, (AI_IMG_STRONG, AI_IMG_WEAK), "Community Forensics + B-Free"
    else:
        return ai
    return dict(ai, commfor=cf, bfree=bf, gapl=gp, prob_ai=round(float(1 / (1 + np.exp(-z))), 4), combined=True,
                thr=thr, used=used, small_image=small)


# ---- whole-image AI generation ------------------------------------------------------------
def ai_score(path, media_type="image"):
    det = _get("ai")
    if det is None:
        return None
    try:
        return det.score_video(path, VIDEO_AI_FRAMES) if media_type == "video" else det.score_image(path)
    except Exception:
        return None


# ---- one call for the backend --------------------------------------------------------------
def voice_score(path):
    """Audio deepfake / voice-clone detector (wav2vec2 ensemble)."""
    vd = _get("voice")
    if vd is None:
        return None
    try:
        return vd.score_audio(path)
    except Exception:
        return None


def audio_verdict(voice):
    if not voice:
        return {"final_score": 0.0, "classification": "Cannot Determine", "confidence_level": "Low",
                "top_contributing_signal": "Voice analysis unavailable"}, "Voice analysis could not be completed."
    p = voice["prob_ai"]
    return {
        "final_score": round(p * 100, 1),
        "classification": voice["verdict"],
        "confidence_level": voice["confidence"],
        "top_contributing_signal": "Voice-clone detector (wav2vec2 XLS-R, calibrated)",
    }, (f"The voice-clone detector gives a calibrated P(AI-cloned) = {p:.2f} ({voice['confidence']} confidence). "
        "Tested on two real-world voice sets (AUC 0.947 and 0.978); detectors can miss voices from new cloning tools.")


def pixel_tools(path, media_type):
    """Runs relevant detectors. Returns {"localization", "ai", "voice", "map_entry", "measurements", "steps"}."""
    if media_type == "audio":
        v = voice_score(path)
        m, steps = [], []
        if v:
            text = (f"AI Voice Clone Detection: {v['verdict']} (P = {v['prob_ai']:.2f}, {v['confidence']} confidence; "
                    f"raw wav2vec2 XLS-R score {v.get('raw_score')}).")
            m.append(text)
            steps.append(("Voice clone check", text))
        return {"localization": None, "ai": None, "voice": v, "map_entry": None, "measurements": m, "steps": steps}
    loc = localize(path, media_type)
    ai = ai_score(path, media_type)
    small = False
    if media_type == "image":
        im = cv2.imread(path)
        small = im is not None and min(im.shape[:2]) < SMALL_SIDE
        ai = ai_image_combined(ai, bfree_score(path), gapl_score(path), small)
    res = {"localization": None, "ai": ai, "map_entry": None, "measurements": [], "steps": []}

    if loc and media_type == "image":
        mes = mesorch_score(path)
        combo = combine(loc, mes, ai)
        if mes is not None:
            from detectors.localization import render
            img = cv2.imread(path)
            drawn = render(img, mes["heat"], draw_boxes=combo is not None and combo >= COMBO_WEAK)
            loc.update(regions=drawn["regions"], overlay=drawn["overlay"], heatmap=drawn["heatmap"],
                       manipulated_pct=drawn["manipulated_pct"], mesorch_score=round(mes["score"], 4))
        loc.pop("_heat", None)
        if combo is not None:
            loc["combined"] = round(combo, 4)
            flagged = combo >= COMBO_WEAK
            caption = ("Heat = likelihood that a region was edited (Mesorch; pixels only, works with metadata removed). "
                       "Decision = TruFor + Mesorch + Community Forensics + B-Free combined"
                       + (". Red boxes mark the suspected edited regions." if flagged and loc["regions"]
                          else ". No region exceeds the edit threshold."))
            res["map_entry"] = {"key": "trufor", "title": "Manipulated-region localization", "image": loc["overlay"],
                                "caption": caption}
            where = "; ".join(f"box x={b['box'][0]}, y={b['box'][1]}, {b['box'][2]}x{b['box'][3]} px ({b['area_pct']}% of image)"
                              for b in loc["regions"][:3]) if flagged and loc["regions"] else "no edited region above threshold"
            m = (f"Edit detection (combined): {combo:.2f} [TruFor {loc['score']:.2f}, Mesorch {mes['score']:.2f}, "
                 f"Community Forensics {ai['commfor']:.2f}, B-Free {ai['bfree']:.2f}]; {where}.")
            res["measurements"].append(m)
            res["steps"].append(("Localization", m.replace("Edit detection (combined): ", "Combined edit score ")))
            res["localization"] = {k: v for k, v in loc.items() if k not in ("overlay", "heatmap")}
            loc = None                     # handled; skip the TruFor-only reporting below

    if loc:
        flagged = loc["score"] >= (VIDEO_EDIT_MIN if media_type == "video" else EDIT_WEAK)
        where_t = f" (most suspicious frame at {loc['t']:.1f}s)" if "t" in loc else ""
        caption = ("Heat = likelihood that a region was edited (TruFor; pixels only, works with metadata removed)"
                   + (f"; shown on the most suspicious of {len(loc['frames'])} key frames" if "frames" in loc else "")
                   + (". Red boxes mark the suspected edited regions." if flagged else ". No region exceeds the edit threshold."))
        res["map_entry"] = {"key": "trufor", "title": "Manipulated-region localization", "image": loc["overlay"],
                            "caption": caption}
        if flagged and loc["regions"]:
            where = "; ".join(f"box x={b['box'][0]}, y={b['box'][1]}, {b['box'][2]}x{b['box'][3]} px ({b['area_pct']}% of frame)"
                              for b in loc["regions"][:3])
            m = (f"Localization (TruFor): manipulation score {loc['score']:.2f}, reliability {loc['reliability']}"
                 f"{where_t}; suspected edited region(s): {where}.")
        else:
            m = (f"Localization (TruFor): manipulation score {loc['score']:.2f}, reliability {loc['reliability']}; "
                 "no edited region above threshold.")
        res["measurements"].append(m)
        res["steps"].append(("Localization", m.replace("Localization (TruFor): ", "TruFor: ")))
        res["localization"] = {k: v for k, v in loc.items() if k not in ("overlay", "heatmap")}

    if ai:
        n = f" (median of {ai['frames']} frames)" if "frames" in ai else ""
        if ai.get("combined"):
            parts = ", ".join(f"{k} {ai[v]:.2f}" for k, v in (("Community Forensics", "commfor"), ("B-Free", "bfree"),
                                                               ("GAPL", "gapl")) if ai.get(v) is not None)
            m = (f"Whole-image AI generation ({ai['used']}): P(AI-generated) = {ai['prob_ai']:.2f} [{parts}]"
                 + (f"; small image (< {SMALL_SIDE} px), so B-Free is left out and edit findings are capped at 'Possible edit'"
                    if ai.get("small_image") else "") + ".")
        else:
            m = (f"Whole-image AI generation (Community Forensics): P(AI-generated) = {ai['prob_ai']:.2f}{n}.")
        res["measurements"].append(m)
        res["steps"].append(("AI-generation check", m.replace("Whole-image AI generation (Community Forensics): ",
                                                               "Community Forensics: ")))
    return res


# ---- EXIF consistency (exif_consistency.py) --------------------------------------------------
def exif_checks(path, media):
    """Adds EXIF consistency findings to the metadata result, in place. A thumbnail or size mismatch
    proves the file was changed after capture, so a "camera original" assessment becomes "edited"."""
    if media.get("media_type") != "image":
        return
    try:
        from detectors.exif_consistency import check
        found = check(path, (media.get("device") or {}).get("class"))
    except Exception:
        return
    media["findings"].extend({"level": f["level"], "text": f["text"]} for f in found)
    media["exif_consistency"] = found
    if any(f["level"] == "high" or f["kind"] == "size" for f in found) and \
            media.get("assessment") in ("camera_original", "stripped", "c2pa_signed"):
        media["assessment"] = "edited"

    # invisible Stable Diffusion watermark (watermark_check.py): 0 false alarms on 518 clean images,
    # so a hit is treated like a metadata AI declaration; a miss says nothing (JPEG/resizing removes it)
    try:
        from detectors.watermark_check import check as wm_check
        wm = wm_check(path)
    except Exception:
        return
    media["watermark"] = wm
    if wm["found"]:
        media["findings"].append({"level": "high", "text":
            f"Invisible AI watermark found: {wm['which']} ({wm['match'] * 100:.0f}% of watermark bits match; clean photos "
            "stay below 70%). The image was produced by a Stable Diffusion pipeline."})
        media["assessment"] = "ai_provenance"


def exif_upgrade(media, v, note):
    """A thumbnail/size contradiction in the camera's own EXIF proves the picture changed after capture.
    When the pixel tools found nothing (or only weak signs), that file-level evidence decides."""
    strong = [f for f in media.get("exif_consistency") or [] if f["level"] == "high" or f["kind"] == "size"]
    if not strong or (v and v["classification"] in ("Edited region detected", "Likely AI-generated")):
        return v, note
    return {"final_score": max(80.0, (v or {}).get("final_score", 0)), "classification": "Edited (metadata evidence)",
            "confidence_level": "Medium", "top_contributing_signal": "EXIF consistency (thumbnail / size)"}, \
        ("The camera's own metadata contradicts the picture: " + strong[0]["text"] +
         (" The pixel-level detectors " + ("found only weak signs" if v and v["classification"] != "No traces found"
                                           else "did not localise the change") + ", so the file-level evidence decides."))


# ---- agent decision rules ------------------------------------------------------------------
def _ai_thr(ai):
    """(strong, weak) thresholds for the whole-image AI probability: image combination or video detector."""
    if ai and ai.get("thr"):
        return ai["thr"]
    return (AI_IMG_STRONG, AI_IMG_WEAK) if ai and ai.get("combined") else (AI_STRONG, AI_WEAK)


def no_face_verdict(loc, ai):
    """Verdict when no face is found (metadata AI declarations are handled before this).
    Order: edited region -> whole-image AI -> nothing found (never claims 'authentic')."""
    is_video = bool(loc and "frames" in loc)
    if loc and loc.get("combined") is not None:
        return _combined_verdict(loc, ai)
    if loc:
        s, rel = loc["score"], loc.get("reliability") or 0.0
        if s >= EDIT_STRONG and rel >= MIN_RELIABILITY:
            return {"final_score": round(s * 100, 1), "classification": "Edited region detected",
                    "confidence_level": "High" if s >= 0.95 else "Medium",
                    "top_contributing_signal": "Manipulation localization (TruFor)"}, \
                f"Pixel-level localization (TruFor) found a likely edited region (score {s:.2f}); this works without metadata."
    ai_strong, ai_weak = _ai_thr(ai)
    if ai and ai["prob_ai"] >= ai_strong:
        return {"final_score": round(ai["prob_ai"] * 100, 1), "classification": "Likely AI-generated",
                "confidence_level": "Medium",   # never High: detectors miss/false-alarm on new generators
                "top_contributing_signal": f"Whole-image AI detectors ({ai.get('used', 'Community Forensics + B-Free')})" if ai.get("combined") else "Whole-image AI detector (Community Forensics)"}, \
            (f"The whole-image AI detector rates this as AI-generated (P = {ai['prob_ai']:.2f}); it works from pixels, "
             "without metadata. Confidence is capped at Medium because detectors are less reliable on the newest generators.")

    if loc and loc["score"] >= EDIT_WEAK and not is_video:
        return {"final_score": round(loc["score"] * 100, 1), "classification": "Possible edit", "confidence_level": "Low",
                "top_contributing_signal": "Manipulation localization (TruFor)"}, \
            f"Pixel-level localization (TruFor) found weak evidence of an edited region (score {loc['score']:.2f})."
    if ai and ai["prob_ai"] >= ai_weak:
        return {"final_score": round(ai["prob_ai"] * 100, 1), "classification": "Possibly AI-generated",
                "confidence_level": "Low", "top_contributing_signal": f"Whole-image AI detectors ({ai.get('used', 'Community Forensics + B-Free')})" if ai.get("combined") else "Whole-image AI detector (Community Forensics)"}, \
            f"The whole-image AI detector gives weak evidence of AI generation (P = {ai['prob_ai']:.2f})."

    if loc or ai:
        parts = ([f"no edited region (TruFor {loc['score']:.2f})"] if loc else []) + \
                ([f"no sign of whole-image AI generation (P = {ai['prob_ai']:.2f})"] if ai else [])
        return {"final_score": round(max([loc["score"] if loc else 0, ai["prob_ai"] if ai else 0]) * 100, 1),
                "classification": "No traces found", "confidence_level": "Low",
                "top_contributing_signal": "Pixel-level checks"}, \
            ("Pixel-level checks found " + " and ".join(parts) + ". This lowers suspicion but is not proof of "
             "authenticity: detectors can miss new generators and subtle edits.")
    return None, None


def _combined_verdict(loc, ai):
    """Images with all three tools. A high combined score with a LOCAL Mesorch region = edited region;
    a high score explained by the whole-image AI detector (no local region) = AI-generated picture."""
    c, p = loc["combined"], ai["prob_ai"]
    ai_strong, ai_weak = _ai_thr(ai)
    detail = (f"combined {c:.2f}: TruFor {loc['score']:.2f}, Mesorch {loc.get('mesorch_score', 0):.2f}, "
              f"AI detectors {p:.2f}")
    local = [r for r in loc.get("regions", []) if r["area_pct"] < WHOLE_IMAGE_PCT]
    # a raised whole-image AI score without a strong TruFor edit = AI-generated picture, not a local edit
    ai_explains = p >= ai_weak and loc["score"] < EDIT_STRONG
    # small image: keep the strong verdict only when TruFor AND Mesorch are both near-certain
    # (OmniFake-OOD 384 px: 0 of 268 real images reached >= 0.95 on both)
    both_sure = loc["score"] >= 0.95 and loc.get("mesorch_score", 0) >= 0.95
    if c >= COMBO_STRONG and local and not ai_explains and ai.get("small_image") and not both_sure:
        return {"final_score": round(c * 100, 1), "classification": "Possible edit", "confidence_level": "Low",
                "top_contributing_signal": "Edit detection (TruFor + Mesorch + Community Forensics + B-Free)"}, \
            (f"The pixel-level detectors point to a locally edited region ({detail}), but the image is small "
             f"(< {SMALL_SIDE} px). Downscaling and re-compression erase the traces these detectors rely on, so the "
             "finding is reported only as a possible edit.")
    if c >= COMBO_STRONG and local and not ai_explains:
        return {"final_score": round(c * 100, 1), "classification": "Edited region detected",
                "confidence_level": "High" if c >= COMBO_VERY_STRONG else "Medium",
                "top_contributing_signal": "Edit detection (TruFor + Mesorch + Community Forensics + B-Free)"}, \
            (f"Three pixel-level detectors together indicate a locally edited region ({detail}); "
             "this works without metadata.")
    if p >= ai_strong or (ai_explains and c >= COMBO_STRONG):
        return {"final_score": round(max(p, c) * 100, 1), "classification": "Likely AI-generated", "confidence_level": "Medium",
                "top_contributing_signal": f"Whole-image AI detectors ({ai.get('used', 'Community Forensics + B-Free')})" if ai.get("combined") else "Whole-image AI detector (Community Forensics)"}, \
            (f"The whole-image AI detector rates this as AI-generated (P = {p:.2f}); it works from pixels, without "
             "metadata. Confidence is capped at Medium because detectors are less reliable on the newest generators.")

    if c >= COMBO_WEAK:
        return {"final_score": round(c * 100, 1), "classification": "Possible edit", "confidence_level": "Low",
                "top_contributing_signal": "Edit detection (TruFor + Mesorch + Community Forensics + B-Free)"}, \
            f"The combined pixel-level detectors give weak evidence of editing ({detail})."
    if p >= ai_weak:
        return {"final_score": round(p * 100, 1), "classification": "Possibly AI-generated", "confidence_level": "Low",
                "top_contributing_signal": f"Whole-image AI detectors ({ai.get('used', 'Community Forensics + B-Free')})" if ai.get("combined") else "Whole-image AI detector (Community Forensics)"}, \
            f"The whole-image AI detector gives weak evidence of AI generation (P = {p:.2f})."
    return {"final_score": round(c * 100, 1), "classification": "No traces found", "confidence_level": "Low",
            "top_contributing_signal": "Pixel-level checks"}, \
        (f"Pixel-level checks found no edited region and no sign of AI generation ({detail}). This lowers suspicion "
         "but is not proof of authenticity: detectors can miss new generators and subtle edits.")


def face_notes(loc, ai):
    """Cross-check notes when a face IS present: the trained face models decide (as validated in
    the paper); the pixel tools add warnings when they see something the face models cannot."""
    notes = []
    edit_min = VIDEO_EDIT_MIN if (loc and "frames" in loc) else EDIT_WEAK
    if loc and loc["score"] >= edit_min:
        notes.append(f"Pixel-level localization (TruFor) flags a likely edited region (score {loc['score']:.2f}, "
                     f"{len(loc['regions'])} region(s)). The face verdict does not cover edits outside the face; "
                     "inspect the localization map.")
    if ai and ai["prob_ai"] >= _ai_thr(ai)[1]:
        notes.append(f"The whole-image AI detector rates the picture as AI-generated (P = {ai['prob_ai']:.2f}). "
                     "The face models detect face swaps in real footage; a fully generated image is a different case.")
    return notes








