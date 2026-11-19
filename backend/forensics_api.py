"""
Analysis backend for the frontend's Media Analysis page. Run from the project root:
    .venv/bin/python -m uvicorn backend.forensics_api:app --host 0.0.0.0 --port 8001

Every score shown comes from the trained models (inference_engine.py); the
verdict and confidence come from the reasoning layer (agentic_reasoning.py).
Metadata forensics and image maps are measured from the uploaded file. The
local LLM (backend/llm.py) only writes the explanation and answers questions.

Flow: POST /jobs -> {job_id}; poll GET /jobs/{id} for staged progress, agent
steps and the streaming explanation. POST /jobs/{id}/ask streams a follow-up
answer grounded in that job's evidence.
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # MUST be before any TF/PyTorch imports!

import base64
import shutil
import sys
import tempfile
import threading
import time
import uuid

import cv2
import torch
import torchvision  # noqa: F401 -- MUST be imported before TensorFlow to prevent Triton segfaults
from fastapi import Body, Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

import image_forensics  # noqa: E402
import llm  # noqa: E402
import metadata_probe  # noqa: E402
import source_verification  # noqa: E402
from detectors import backend_hooks  # noqa: E402  -- new tools (TruFor localization) live in detectors/
import store  # noqa: E402
from llm import BRANCH_NAMES, fmt_t, r0  # noqa: E402

ALLOWED = {".jpg", ".jpeg", ".png", ".bmp", ".mp4", ".mov", ".avi", ".mkv",   # what InferenceEngine reads
           ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}                  # audio -> voice-clone check (detectors/)
BRANCH_MODELS = {
    "spatial": "EfficientNet-B0 inference",
    "frequency": "ResNet18 on FFT high-pass",
    "temporal": "ResNet18+LSTM over 16-frame sequences",
}
SEGMENT_THRESHOLD = 60   # per-frame score (0-100) that counts as "hot" for localisation
SEGMENT_MIN_SAMPLES = 3  # consecutive hot samples needed to call it a segment

app = FastAPI(title="Deepfake Forensics API")
# The frontend reaches this API through its /backend proxy (same origin), so CORS
# only matters for direct calls during development.
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("FRONTEND_ORIGINS", "http://localhost:3333").split(","),
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
SESSION_COOKIE = "verifai_session"

JOBS = {}
LOCK = threading.Lock()
_engine = None
_engine_ready = threading.Event()
_engine_error = None
_analysis_lock = threading.Lock()  # one analysis at a time: GPU memory + RetinaFace are shared


def _update(job_id, **patch):
    with LOCK:
        JOBS[job_id].update(patch)


def _set_branch(job_id, name, **patch):
    with LOCK:
        JOBS[job_id]["branches"][name].update(patch)


def _add_step(job_id, title, detail, status="done"):
    with LOCK:
        JOBS[job_id]["steps"].append({"title": title, "detail": detail, "status": status,
                                      "t": round(time.time() - JOBS[job_id]["started"], 1)})


# ------------------------------------------------------------------ auth
def current_user(request: Request):
    user = store.user_for_token(request.cookies.get(SESSION_COOKIE))
    if user is None:
        raise HTTPException(401, "Not signed in")
    return user


def _owned_job(job_id, user):
    job = JOBS.get(job_id)
    if not job or job["user_id"] != user["id"]:
        raise HTTPException(404, "Unknown job")  # 404, not 403: don't reveal other users' job ids
    return job


def _start_session(response: Response, user):
    response.set_cookie(SESSION_COOKIE, store.create_session(user["id"]), max_age=store.SESSION_TTL,
                        httponly=True, samesite="lax", secure=os.environ.get("COOKIE_SECURE") == "1")
    return user


# ------------------------------------------------------------------ engine
def _load_engine():
    global _engine, _engine_error
    try:
        cwd = os.getcwd()
        os.chdir(PROJECT_ROOT)  # engine resolves some paths relative to cwd
        try:
            from agent.agentic_reasoning import AgenticReasoner
            from agent.inference_engine import InferenceEngine
            engine = InferenceEngine()
            # The engine's reasoner writes rule-based text only; the streamed Gemma
            # explanation is produced here, from the same report.
            engine.reasoner = AgenticReasoner(llm_client=None)
            _engine = engine
        finally:
            os.chdir(cwd)
    except Exception as e:
        _engine_error = f"{type(e).__name__}: {e}"
    finally:
        _engine_ready.set()


def get_engine():
    _engine_ready.wait()
    if _engine is None:
        raise RuntimeError(f"Inference engine failed to load ({_engine_error})")
    return _engine


# ------------------------------------------------------------------ frames
def _jpeg_b64(img, max_w=360):
    h, w = img.shape[:2]
    if w > max_w:
        img = cv2.resize(img, (max_w, int(h * max_w / w)))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 78])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode() if ok else None


def _grab_frame(path, t_sec):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def _draw_box(frame, bbox, flagged):
    """bbox is the engine's own [x, y, w, h] for that frame -- the face the models scored."""
    out = frame.copy()
    if bbox and len(bbox) == 4:
        x, y, w, h = (int(v) for v in bbox)
        color = (68, 68, 239) if flagged else (94, 197, 34)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, max(2, frame.shape[1] // 300))
    return out


def _sample_seconds(path, media):
    """Seconds per engine frame_id: VideoProcessor keeps every round(fps / FRAME_RATE)-th frame."""
    from config import Config
    fps = media.get("fps")
    if not fps:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
    fps = fps or 25.0
    return max(int(round(fps / Config.FRAME_RATE)), 1) / fps


def _thumbnail(path, media):
    """Small JPEG for the history list, made before the upload is deleted."""
    if media["media_type"] == "video":
        frame = _grab_frame(path, (media.get("duration_s") or 0) * 0.25)
    else:
        frame = cv2.imread(path)
    return _jpeg_b64(frame, max_w=240) if frame is not None else None


def _save_history(job_id):
    """Store the finished analysis in its owner's history (the History page reads these)."""
    with LOCK:
        job = dict(JOBS[job_id])
    v, b, media = job["verdict"] or {}, job["branches"], job["media"] or {}
    score = lambda k: (b.get(k) or {}).get("score")
    record = {
        "job_id": job_id,
        "filename": job["filename"],
        "media_type": media.get("media_type"),
        "classification": v.get("classification"),
        "final_score": v.get("final_score"),
        "confidence_level": v.get("confidence_level"),
        "spatial_score": score("spatial"),
        "frequency_score": score("frequency"),
        "temporal_score": score("temporal"),
        "metadata_flag": media.get("assessment"),
        "override_applied": v.get("override_applied"),
        "no_face": v.get("no_face"),
        "top_contributing_signal": v.get("top_contributing_signal"),
        "source_matches": len((job.get("source") or {}).get("matches") or []),
        "forensic_reasoning": job.get("explanation"),
        "explanation_source": job.get("explanation_source"),
        "media_preview": job.get("thumbnail"),
    }
    try:
        store.save_analysis(job["user_id"], record)
    except Exception:
        pass  # history is best-effort; the result itself is already on screen


# ------------------------------------------------------------------ evidence
def _finding(signal):
    s = r0(signal["score"] * 100)
    text = (f"Model output {s}/100 -- manipulation signal" if s >= 60 else
            f"Model output {s}/100 -- weak signal" if s >= 40 else
            f"Model output {s}/100 -- no manipulation signal")
    if not signal["contributes_to_verdict"]:
        text += f" (not relied on: {signal['reliability']})"
    return text


def build_evidence(path, media, out, track):
    """Branch cards, timeline, suspicious segments and evidence frames from the engine output."""
    is_video = media["media_type"] == "video"
    signals = {s["signal"]: s for s in track["signals"]}
    branches = {b: {"score": round(sig["score"] * 100, 1), "finding": _finding(sig)}
                for b, sig in signals.items() if sig["score"] is not None}

    tid = track["track_id"]
    rows = sorted((r for r in out["per_face_results"] if r["track_id"] == tid), key=lambda r: r["frame_id"])
    timeline, segments, evidence = [], [], []

    if is_video and rows:
        spf = _sample_seconds(path, media)
        for r in rows:
            timeline.append({"t": round(r["frame_id"] * spf, 2), "frame_id": r["frame_id"], "bbox": r["bbox"],
                             "spatial": round(r["spatial_prob_fake"] * 100, 1),
                             "frequency": round(r["frequency_prob_fake"] * 100, 1)})
        run = []
        for p in timeline + [None]:
            if p is not None and (p["spatial"] + p["frequency"]) / 2 >= SEGMENT_THRESHOLD:
                run.append(p)
                continue
            if len(run) >= SEGMENT_MIN_SAMPLES:
                peak = max(run, key=lambda q: q["spatial"] + q["frequency"])
                flagged = [b for b in ("spatial", "frequency")
                           if sum(q[b] for q in run) / len(run) >= SEGMENT_THRESHOLD] or ["spatial"]
                segments.append({"start": run[0]["t"], "end": run[-1]["t"], "peak": peak["t"],
                                 "peak_bbox": peak["bbox"], "branches": flagged,
                                 "cues": [f"sustained high {BRANCH_NAMES[b]} score "
                                          f"(avg {r0(sum(q[b] for q in run) / len(run))}/100 over {len(run)} samples)"
                                          for b in flagged]})
            run = []

        if segments:
            picks = [(s["peak"], s["peak_bbox"], True, s["cues"][0]) for s in segments[:4]]
        else:
            picks = [(p["t"], p["bbox"], False, "per-frame scores stayed below the flag threshold here")
                     for p in (timeline[len(timeline) // 4], timeline[(len(timeline) * 3) // 5])]
        for t, bbox, flagged, note in picks:
            frame = _grab_frame(path, t)
            if frame is not None:
                evidence.append({"t": t, "flagged": flagged, "note": note, "face_found": True,
                                 "image": _jpeg_b64(_draw_box(frame, bbox, flagged))})
        for p in timeline:
            del p["frame_id"], p["bbox"]
        for s in segments:
            del s["peak_bbox"]
    elif rows:
        img = cv2.imread(path)
        if img is not None:
            flagged = track["fusion_prob_fake"] >= 0.5
            evidence.append({"t": None, "flagged": flagged, "note": "face region the models analysed",
                             "face_found": True, "image": _jpeg_b64(_draw_box(img, rows[0]["bbox"], flagged))})

    return {"branches": branches, "fused_score": round(track["fusion_prob_fake"] * 100, 1),
            "timeline": timeline, "segments": segments, "evidence": evidence}


# ------------------------------------------------------------------ verdict
def decide(report, track, evidence, media):
    """
    Verdict = the reasoning layer's call (Fusion score + reliability audit),
    mapped to display labels exactly like ReasoningReport.to_forensic_manager_dict.
    A metadata AI declaration is the only thing allowed to override it.
    """
    notes = []
    verdict, conf = report["overall_verdict"], report["overall_confidence"]
    if verdict == "INCONCLUSIVE":
        classification = "Inconclusive"
    elif verdict == "FAKE":
        classification = "Likely Fake" if conf != "Low" else "Suspicious"
    else:
        classification = "Likely Real" if conf != "Low" else "Inconclusive"

    scores = {b: v["score"] for b, v in evidence["branches"].items()}
    spread = (max(scores.values()) - min(scores.values())) if len(scores) > 1 else 0.0
    used = [s for s in track["signals"] if s["contributes_to_verdict"]]
    top = max(used, key=lambda s: s["separation"])["signal"] if used else None
    final = evidence["fused_score"]

    for s in track["signals"]:
        notes.append(s["note"])  # already starts with the signal name
    for c in track["caveats"]:
        notes.append(c["text"])

    override = media.get("assessment") in ("ai_provenance", "ai_tool_signature")
    if override:
        classification, conf, final = "AI-Generated (metadata)", "High", max(final, 92.0)
        notes.append("Metadata declares AI generation (C2PA / IPTC / generator signature) -> provenance override applied.")
    elif media.get("assessment") in ("edited", "screen_recording"):
        notes.append(f"Metadata shows the file was {'edited in software' if media['assessment'] == 'edited' else 'screen-recorded'}"
                     " -- not a camera original; noted alongside the detectors, not used in the score.")
    elif media.get("assessment") == "camera_original":
        notes.append("Camera-original metadata " + ("is consistent with the low manipulation score." if final < 50 else
                     "is present, but metadata can be copied; the detectors take precedence."))
    else:
        notes.append("Metadata is neutral (stripped or re-encoded); the verdict rests on the detectors.")

    return {
        "final_score": round(final, 1),
        "classification": classification,
        "confidence_level": conf,
        "override_applied": override,
        "branch_spread": round(spread, 1),
        "top_contributing_signal": BRANCH_NAMES[top] if top else "None (no reliable signal)",
        "not_assessed": False,
        "no_face": False,
    }, notes


def decide_no_face(media, localization=None, ai=None, voice=None):
    """No face track: the face detectors have nothing to score. A metadata AI declaration classifies;
    otherwise the pixel-level or audio tools in detectors/ decide."""
    if media.get("media_type") == "audio":
        v, note = backend_hooks.audio_verdict(voice)
        return {**v, "override_applied": False, "branch_spread": 0.0, "not_assessed": False, "no_face": True}, [note]
    notes = [llm.NO_FACE_REASON]
    if media.get("assessment") in ("ai_provenance", "ai_tool_signature"):
        notes.append("Metadata declares AI generation (C2PA / IPTC / generator signature / invisible AI watermark) "
                     "-> classified from that declaration alone.")
        return {"final_score": 92.0, "classification": "AI-Generated (metadata)", "confidence_level": "High",
                "override_applied": True, "branch_spread": 0.0, "top_contributing_signal": "Metadata declaration",
                "not_assessed": False, "no_face": True}, notes
    v, note = backend_hooks.no_face_verdict(localization, ai)
    v, note = backend_hooks.exif_upgrade(media, v, note)
    if v:
        notes.append(note)
        return {**v, "override_applied": False, "branch_spread": 0.0, "not_assessed": False, "no_face": True}, notes
    notes.append("The metadata makes no AI declaration and the pixel-level tools were unavailable, so no verdict is given.")
    return {"final_score": 0.0, "classification": "Cannot Determine", "confidence_level": "n/a",
            "override_applied": False, "branch_spread": 0.0, "top_contributing_signal": "None (no face)",
            "not_assessed": True, "no_face": True}, notes


# ------------------------------------------------------------------ pipeline
def _progress_callback(job_id, names):
    """Maps the engine's real stage reports onto the three branch cards."""
    def cb(stage, frac):
        with LOCK:
            br = JOBS[job_id]["branches"]
            if stage in ("decode", "detect", "track"):
                label = {"decode": "Decoding frames", "detect": "Detecting faces (RetinaFace)",
                         "track": "Tracking & cropping faces"}[stage]
                for n in names:
                    br[n].update(status="running", stage=label, progress=None)
            elif stage == "evidence":
                for n in ("spatial", "frequency"):
                    br[n].update(status="running", stage=BRANCH_MODELS[n], progress=round(frac * 100))
                if "temporal" in names:
                    br["temporal"].update(status="running", stage="Waiting for tracked face crops", progress=0)
            elif stage == "temporal" and "temporal" in names:
                for n in ("spatial", "frequency"):
                    br[n].update(stage="Scored; waiting for fusion", progress=100)
                br["temporal"].update(status="running", stage=BRANCH_MODELS["temporal"], progress=round(frac * 100))
            elif stage in ("fusion", "reasoning"):
                for n in names:
                    br[n].update(stage="Fusing scores", progress=100)
    return cb


def _image_maps(job_id, path, media, face_box):
    """ELA / spectrum / noise maps for still images. Supplementary: never fails the job."""
    if media["media_type"] != "image":
        return None
    try:
        fx = image_forensics.analyze(path, media, tuple(int(v) for v in face_box) if face_box else None)
    except Exception:
        return None
    if not fx:
        return None
    prev = JOBS[job_id].get("forensics") or {}          # pixel tools (TruFor / AI detector) ran first
    fx["maps"] = prev.get("maps", []) + fx["maps"]
    fx["measurements"] = prev.get("measurements", []) + fx["measurements"]
    for k in ("localization", "ai"):
        fx[k] = prev.get(k)
    _update(job_id, forensics=fx)
    _add_step(job_id, "Image forensics", "Computed ELA, frequency-spectrum and noise-residual maps from the uploaded "
              "pixels" + (", comparing the face region with the rest of the image." if face_box else "."))
    return fx["measurements"]


def _pixel_tools(job_id, path, media):
    """New tools (detectors/): TruFor localization + whole-image AI detector, for images and videos.
    Supplementary: never fails the job."""
    try:
        with _analysis_lock:                              # shares the GPU with the face models
            px = backend_hooks.pixel_tools(path, media["media_type"])
    except Exception:
        return None
    fx = {"maps": [px["map_entry"]] if px["map_entry"] else [], "measurements": px["measurements"],
          "localization": px["localization"], "ai": px["ai"], "voice": px.get("voice")}
    _update(job_id, forensics=fx)
    for title, detail in px["steps"]:
        _add_step(job_id, title, detail)
    return px


def _start_source_check(job_id, path, media):
    """Reverse image search in the background, alongside the models."""
    _update(job_id, source={"status": "running", "matches": [], "message": "Searching the web for copies..."})
    holder = {}

    def work():
        try:
            holder["result"] = source_verification.verify(path, media)
        except Exception as e:
            holder["result"] = {"status": "error", "matches": [], "domains": [], "searched": [],
                                "message": f"Reverse image search failed ({type(e).__name__})."}
        _update(job_id, source=holder["result"])

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    return thread, holder


def _source_step(job_id, check):
    """Wait for the search (it has usually finished) and log it as an agent step."""
    if check is None:
        return {"status": "skipped", "matches": [], "domains": [], "message": "Audio input -- reverse image search not applicable."}
    thread, holder = check
    thread.join(timeout=90)
    src = holder.get("result")
    if src is None:
        src = {"status": "error", "matches": [], "domains": [], "message": "Reverse image search timed out."}
        _update(job_id, source=src)
    detail = src["message"]
    if src.get("domains"):
        detail += " Sites: " + ", ".join(src["domains"][:6]) + "."
    if src.get("cached"):
        detail += " (cached result for this exact file)"
    _add_step(job_id, "Source verification", detail + " Context only -- not part of the score.")
    return src


def _explain(job_id, prompt, fallback):
    _update(job_id, stage="explaining")
    _add_step(job_id, "Report", f"Drafting analyst explanation with {llm.OLLAMA_MODEL}...", status="running")
    text = ""
    try:
        for tok in llm.stream(prompt):
            text += tok
            _update(job_id, explanation=text)
        text, source = text.strip(), llm.OLLAMA_MODEL
        if not text:
            raise ValueError("empty response")
    except Exception as e:
        text, source = fallback, f"rule-based (LLM unavailable: {type(e).__name__})"
    with LOCK:
        JOBS[job_id]["steps"][-1].update(status="done", detail=f"Explanation written by {source}.")
    _update(job_id, explanation=text, explanation_source=source)


def run_job(job_id, path, filename):
    source_check = None
    try:
        # 1. metadata + pixel maps (measured from the file)
        _update(job_id, stage="metadata")
        media = metadata_probe.probe(path, filename)
        backend_hooks.exif_checks(path, media)          # detectors/: does the camera metadata hang together?
        _update(job_id, media=media)
        n_find = len([f for f in media["findings"] if f["level"] != "info"])
        c2 = "C2PA manifest found" if media["c2pa"].get("present") else "no C2PA manifest"
        _add_step(job_id, "Metadata forensics", f"{c2}; assessment: {media['assessment'].replace('_', ' ')}; "
                  f"{n_find} notable finding(s).")
        source_check = _start_source_check(job_id, path, media) if media["media_type"] != "audio" else None
        _update(job_id, thumbnail=_thumbnail(path, media))
        # 2. evidence models
        is_video = media["media_type"] == "video"
        is_audio = media["media_type"] == "audio"
        if is_audio:
            with LOCK:
                JOBS[job_id]["branches"] = {n: {"progress": 0, "status": "skipped",
                                                "stage": "Not applicable to audio"}
                                            for n in ("spatial", "frequency", "temporal")}
            _add_step(job_id, "Plan", "Audio input -> running wav2vec2 voice clone detection suite.")
            _update(job_id, stage="branches")
            out = {}
        else:
            names = ["spatial", "frequency"] + (["temporal"] if is_video else [])
            with LOCK:
                JOBS[job_id]["branches"] = {n: {"progress": 0, "status": "queued", "stage": "Queued"} for n in names}
                if not is_video:
                    JOBS[job_id]["branches"]["temporal"] = {"progress": 0, "status": "skipped",
                                                            "stage": "Not applicable to still images"}
            _add_step(job_id, "Plan", f"{media['media_type'].capitalize()} input -> running "
                      f"{', '.join(n.capitalize() for n in names)} branches"
                      + ("" if is_video else "; Temporal skipped (single frame)") + ".")
            _update(job_id, stage="branches")
            engine = get_engine()
            if not _analysis_lock.acquire(blocking=False):
                for n in names:
                    _set_branch(job_id, n, stage="Waiting for the previous analysis to finish")
                _analysis_lock.acquire()
            try:
                out = engine.analyze(path, progress=_progress_callback(job_id, names))
            finally:
                _analysis_lock.release()

        _pixel_tools(job_id, path, media)               # detectors/: TruFor localization + whole-image AI detector
        report = out.get("reasoning") or {}
        tracks = out.get("per_track_summary") or []
        driving = next((t for t in report.get("tracks", []) if t["track_id"] == report.get("driving_track_id")), None)

        # 3a. no usable face -> no detector verdict
        if not tracks or driving is None:
            with LOCK:
                JOBS[job_id]["branches"] = {n: {"progress": 0, "status": "skipped", "stage": "Not applicable to audio" if is_audio else "No face detected -- not applicable"}
                                            for n in ("spatial", "frequency", "temporal")}
            if not is_audio:
                _add_step(job_id, "Face detection", "No usable face track in the " +
                          ("image" if not is_video else "video") + " -> face-based detectors had nothing to score.")
            _update(job_id, evidence={"timeline": [], "segments": [], "evidence": []}, stage="reasoning")
            measurements = _image_maps(job_id, path, media, None) if not is_audio else []
            source = _source_step(job_id, source_check)
            fxj = JOBS[job_id].get("forensics") or {}
            verdict, notes = decide_no_face(media, fxj.get("localization"), fxj.get("ai"), fxj.get("voice"))
            measurements = measurements or fxj.get("measurements")
            for note in notes:
                _add_step(job_id, "Cross-check", note)
            _add_step(job_id, "Verdict", f"{verdict['classification']} · " +
                      (f"{r0(verdict['final_score'])}/100 · {verdict['confidence_level']} confidence."
                       if not verdict["not_assessed"] else "no manipulation score produced."))
            _update(job_id, verdict=verdict)
            if is_audio:   # audio: the voice detector decides; the face detectors are not part of the story
                fallback = " ".join([f"Outcome: {verdict['classification']}."] + notes)
            else:
                fallback = " ".join([f"Outcome: {verdict['classification']}.", llm.NO_FACE_REASON] +
                                    [f["text"] for f in media.get("findings", [])[-1:]])
            _explain(job_id, llm.build_no_face_prompt(media, verdict, measurements, source), fallback)
            _update(job_id, stage="done", status="done")
            _save_history(job_id)
            return

        # 3b. per-branch results, localisation
        track = driving
        _update(job_id, track=track)
        evidence = build_evidence(path, media, out, track)
        for n, v in evidence["branches"].items():
            _set_branch(job_id, n, status="done", progress=100, stage="Complete", score=v["score"], finding=v["finding"])
        if is_video and "temporal" not in evidence["branches"]:
            _set_branch(job_id, "temporal", status="skipped", stage="Too few tracked frames for a sequence")
        _update(job_id, evidence={k: evidence[k] for k in ("timeline", "segments", "evidence")})
        _add_step(job_id, "Face tracking", f"{len(tracks)} person(s) tracked across {out['frame_count']} sampled frame(s); "
                  f"reporting track {track['track_id']}" + (" (the most suspicious)" if len(tracks) > 1 else "") + ".")
        _add_step(job_id, "Evidence models", ", ".join(f"{n.capitalize()} {r0(v['score'])}"
                  for n, v in evidence["branches"].items()) + f" -> Fusion {r0(evidence['fused_score'])}/100.")
        if evidence["segments"]:
            _add_step(job_id, "Localisation", f"{len(evidence['segments'])} suspicious segment(s): " +
                      ", ".join(f"{fmt_t(s['start'])}-{fmt_t(s['end'])}" for s in evidence["segments"]) + ".")

        face_rows = [r for r in out["per_face_results"] if r["track_id"] == track["track_id"]]
        measurements = _image_maps(job_id, path, media, face_rows[0]["bbox"] if face_rows else None)

        # 4. reasoning (deterministic)
        _update(job_id, stage="reasoning")
        source = _source_step(job_id, source_check)
        verdict, notes = decide(report, track, evidence, media)
        fxj = JOBS[job_id].get("forensics") or {}
        notes += backend_hooks.face_notes(fxj.get("localization"), fxj.get("ai"))
        for note in notes:
            _add_step(job_id, "Cross-check", note)
        _add_step(job_id, "Verdict", f"{verdict['classification']} · {r0(verdict['final_score'])}/100 · "
                  f"{verdict['confidence_level']} confidence.")
        _update(job_id, verdict=verdict)

        # 5. explanation
        _explain(job_id, llm.build_explanation_prompt(media, evidence, verdict, track, measurements, source),
                 track["explanation"])
        _update(job_id, stage="done", status="done")
        _save_history(job_id)
    except Exception as e:
        _update(job_id, status="error", error=str(e), stage="error")
    finally:
        if source_check:
            source_check[0].join(timeout=90)  # it still reads the uploaded file
        try:
            os.remove(path)
        except OSError:
            pass


# ------------------------------------------------------------------ routes
@app.post("/auth/register")
def register(response: Response, payload: dict = Body(...)):
    try:
        user = store.create_user(payload.get("email"), payload.get("name"), payload.get("password"))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _start_session(response, user)


@app.post("/auth/login")
def login(response: Response, payload: dict = Body(...)):
    user = store.authenticate(payload.get("email"), payload.get("password"))
    if user is None:
        raise HTTPException(401, "Incorrect email or password.")
    return _start_session(response, user)


@app.post("/auth/logout")
def logout(request: Request, response: Response):
    store.delete_session(request.cookies.get(SESSION_COOKIE))
    response.delete_cookie(SESSION_COOKIE)
    return {"ok": True}


@app.get("/auth/me")
def me(user=Depends(current_user)):
    return user


@app.get("/analyses")
def analyses(user=Depends(current_user)):
    return {"success": True, "data": store.list_analyses(user["id"])}


@app.delete("/analyses/{rid}")
def delete_analysis(rid: str, user=Depends(current_user)):
    if not store.delete_analysis(user["id"], rid):
        raise HTTPException(404, "Unknown analysis")
    return {"success": True}


@app.post("/jobs")
def create_job(file: UploadFile = File(...), user=Depends(current_user)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED:
        raise HTTPException(400, f"Unsupported file type: {ext or 'unknown'}")
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
    job_id = uuid.uuid4().hex[:12]
    with LOCK:
        JOBS[job_id] = {"id": job_id, "user_id": user["id"], "filename": file.filename, "status": "running", "stage": "queued",
                        "started": time.time(), "branches": {}, "steps": [], "explanation": "",
                        "forensics": None, "media": None, "evidence": None, "verdict": None, "track": None,
                        "source": None, "thumbnail": None, "qa": []}
    threading.Thread(target=run_job, args=(job_id, tmp.name, file.filename), daemon=True).start()
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
def get_job(job_id: str, user=Depends(current_user)):
    with LOCK:
        job = _owned_job(job_id, user)
        return {**{k: v for k, v in job.items() if k not in ("user_id", "qa")},
                "elapsed": round(time.time() - job["started"], 1)}


@app.post("/jobs/{job_id}/ask")
def ask(job_id: str, payload: dict = Body(...), user=Depends(current_user)):
    question = (payload.get("question") or "").strip()[:500]
    if not question:
        raise HTTPException(400, "Empty question")
    with LOCK:
        job = _owned_job(job_id, user)
        if job["status"] != "done":
            raise HTTPException(409, "Analysis still running")
        snapshot = {k: job[k] for k in ("media", "evidence", "verdict", "track", "branches", "steps", "explanation",
                                        "forensics", "source")}
        history = list(job.setdefault("qa", []))
    prompt = llm.build_qa_prompt(snapshot, question, history)

    def gen():
        answer = ""
        try:
            for tok in llm.stream(prompt, max_tokens=200, temperature=0.3, timeout=120):
                answer += tok
                yield tok
        except Exception as e:
            msg = f"[The agent's language model is unavailable right now ({type(e).__name__}).]"
            answer += msg
            yield msg
        with LOCK:
            JOBS[job_id].setdefault("qa", []).append((question, answer.strip()))

    return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")


@app.get("/health")
def health():
    return {"status": "ok", "llm": llm.status(), "source_verification": source_verification.status(),
            "engine_loaded": _engine is not None,
            "engine_error": _engine_error, "device": str(_engine.device) if _engine else None,
            "checkpoints": _engine.checkpoints if _engine else None}


@app.on_event("startup")
def startup():
    store.init()
    threading.Thread(target=_load_engine, daemon=True).start()
    threading.Thread(target=llm.warm_up, daemon=True).start()
