"""
Local LLM (Gemma 2 9B via Ollama) for the analyst explanation and follow-up
questions. It only ever phrases facts the pipeline already decided: the
verdict, confidence and caveats come from agentic_reasoning.py, and the
prompts forbid changing them. If Ollama is down, callers fall back to the
reasoning layer's rule-based explanation.

Env: OLLAMA_URL (default http://localhost:11435, the ollama-gpu container),
     OLLAMA_MODEL (default gemma2:latest).
"""

import json
import os

import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11435").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma2:latest")
BRANCH_NAMES = {"spatial": "Spatial", "frequency": "Frequency", "temporal": "Temporal"}
DETECTORS = ("Spatial (EfficientNet-B0 on face crops), Frequency (ResNet18 on FFT high-pass residuals) and, "
             "for video, Temporal (ResNet18+LSTM over 16-frame face sequences), combined by a trained Fusion model")


def r0(x):
    """Round half up, matching the browser's toFixed(0), so log and cards agree."""
    return int(float(x) + 0.5)


def fmt_t(t):
    if t is None:
        return "n/a"
    m, s = divmod(float(t), 60)
    return f"{int(m)}:{s:04.1f}"


def status():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        names = [m["name"] for m in r.json().get("models", [])]
        return {"online": True, "model": OLLAMA_MODEL, "model_available": OLLAMA_MODEL in names}
    except Exception:
        return {"online": False, "model": OLLAMA_MODEL, "model_available": False}


def warm_up():
    """Load the model into memory ahead of the first request."""
    try:
        requests.post(f"{OLLAMA_URL}/api/generate",
                      json={"model": OLLAMA_MODEL, "prompt": "", "keep_alive": "60m"}, timeout=120)
    except Exception:
        pass


def _segments(evidence):
    return [{"from": fmt_t(s["start"]), "to": fmt_t(s["end"]),
             "flagged_by": [BRANCH_NAMES[b] for b in s["branches"]], "observations": s["cues"]}
            for s in (evidence or {}).get("segments", [])]


def _signals(track):
    """The reasoning layer's per-signal assessment, trimmed for the prompt."""
    return {BRANCH_NAMES[s["signal"]]: {
                "score_out_of_100": None if s["score"] is None else r0(s["score"] * 100),
                "reliability": s["reliability"],
                "used_for_verdict": s["contributes_to_verdict"],
                "note": s["note"]}
            for s in track["signals"]}


def _online(source):
    """Reverse-image-search summary for prompts (None when it did not run)."""
    if not source or source.get("status") != "done":
        return None
    return {"summary": source["message"],
            "top_matches": [{"site": m["domain"], "title": m["title"], "match": m["kind"]} for m in source["matches"][:5]]}


def build_explanation_prompt(media, evidence, verdict, track, measurements=None, source=None):
    facts = {
        "media_type": media["media_type"],
        "duration": fmt_t(media.get("duration_s")) if media["media_type"] == "video" else None,
        "verdict": verdict["classification"],
        "manipulation_score_out_of_100": r0(verdict["final_score"]),
        "confidence": verdict["confidence_level"],
        "confidence_reasons": track["confidence_reasons"] if track else [],
        "detectors": _signals(track) if track else {},
        "caveats": [c["text"] for c in track["caveats"]] if track else [],
        "suspicious_segments": _segments(evidence),
        "metadata_findings": [f["text"] for f in media.get("findings", [])],
        "metadata_override": verdict["override_applied"],
        "image_measurements": measurements or None,
        "found_online": _online(source),
    }
    return (
        f"You are the reporting agent of a deepfake forensics system. Detectors: {DETECTORS}. "
        "A metadata/provenance check also ran.\n\n"
        f"FACTS (JSON):\n{json.dumps(facts, indent=1)}\n\n"
        "The verdict, score and confidence are FIXED by the system; never change, soften or contradict them. "
        "Write a concise forensic explanation for an analyst, 4 to 6 sentences, plain text, no markdown, "
        "no bullet points, no headings. Start with the verdict and score. Only detectors with used_for_verdict=true "
        "are the basis of the verdict; if you mention another detector, say it was not relied on and why. "
        "Reflect every caveat in plain words. For video, cite the suspicious time ranges exactly as given. "
        "State what the metadata does or does not tell us. If image_measurements is present you may quote one "
        "or two numbers as descriptive measurements, not as proof. If found_online is present, say in one sentence "
        "where copies were found; this is context about the media's origin, not evidence of manipulation. "
        "Say it is an automated estimate, not proof. "
        "Use only the facts above -- do not invent timestamps, people, sources or manipulation techniques."
    )


NO_FACE_REASON = ("No face was detected in this media, so the three face detectors (Spatial, Frequency, Temporal) "
                  "could not run; the outcome rests on the image-level and metadata checks.")


def build_no_face_prompt(media, verdict, measurements=None, source=None):
    facts = {
        "media_type": media["media_type"],
        "outcome": verdict["classification"],
        "why_no_detector_scores": NO_FACE_REASON,
        "metadata_findings": [f["text"] for f in media.get("findings", [])],
        "metadata_assessment": (media.get("assessment") or "").replace("_", " "),
        "image_measurements": measurements or None,
        "found_online": _online(source),
    }
    if verdict.get("override_applied"):
        ask = ("State that no face was found, so the face-based detectors could not run. Then say the file's own metadata "
               "declares it AI-generated and that this is the sole basis for the classification.")
    elif verdict["classification"] in ("Likely AI-generated", "Possibly AI-generated"):
        facts["why_no_detector_scores"] = "No face was detected, so the face-based deepfake detectors could not run."
        facts["ai_detector_score_out_of_100"] = verdict.get("final_score")
        sure = "strong" if verdict["classification"] == "Likely AI-generated" else "weak"
        ask = ("State that no face was found, so the face-based detectors could not run. Then state that a whole-image "
               f"AI-generation detector, which works from the pixels and not from metadata, found {sure} evidence that the "
               "image was produced by an AI generator, and give its score. Add that such detectors are less reliable on "
               "the newest generators, so this is an estimate, not proof.")
    elif verdict["classification"] == "No traces found":
        facts["why_no_detector_scores"] = "No face was detected, so the face-based deepfake detectors could not run."
        ask = ("State that no face was found, so the face-based detectors could not run. Then state that pixel-level checks "
               "found neither an edited region nor signs of AI generation, and that this lowers suspicion but does NOT prove "
               "the image is authentic, because detectors can miss new generators and subtle edits.")
    elif verdict["classification"] in ("Edited region detected", "Possible edit"):
        facts["why_no_detector_scores"] = "No face was detected, so the face-based deepfake detectors could not run."
        facts["localization_score_out_of_100"] = verdict.get("final_score")
        sure = "strong" if verdict["classification"] == "Edited region detected" else "weak, low-confidence"
        ask = ("State that no face was found, so the face-based detectors could not run. Then state that pixel-level "
               f"manipulation localization found {sure} evidence that part of the image was edited, give its score, and "
               "point the reader to the highlighted region in the localization map. Say this analysis works from the "
               "pixels and does not rely on metadata. Do not claim the whole image is AI-generated.")
    elif media.get("media_type") == "audio":
        facts["voice_detector_probability_ai_cloned"] = verdict.get("final_score")
        ask = ("This is an audio file. State the outcome of the voice-clone check and its probability that the voice is "
               "AI-cloned. Say the detector analyses the sound of the speech itself (not metadata), was tested on two "
               "real-world voice datasets, and can miss voices made with new cloning tools, so the result is an estimate, "
               "not proof. Do not mention faces or video detectors.")
    elif verdict["classification"] == "Edited (metadata evidence)":
        facts["why_no_detector_scores"] = "No face was detected, so the face-based deepfake detectors could not run."
        ask = ("State that no face was found, so the face-based detectors could not run. Then state that the camera's own "
               "metadata contradicts the picture (for example, the small preview image the camera stored at capture no "
               "longer matches the picture), which shows the picture was changed after it was taken while the original "
               "camera metadata was kept. Say that the pixel detectors did not pinpoint the change, and that this shows the "
               "file was altered, not what was altered or whether the change is misleading.")
    else:
        ask = ("State that no face was found, so the face-based detectors could not run and the system cannot determine "
               "whether the media is fake or real. Do not say or imply that it is authentic, and do not say it is fake. "
               "Then say what the metadata does or does not tell us.")
    return (
        "You are the reporting agent of a deepfake forensics system that analyses faces.\n\n"
        f"FACTS (JSON):\n{json.dumps(facts, indent=1)}\n\n"
        f"Write a concise note for an analyst, 3 sentences, plain text, no markdown, no bullet points. {ask} "
        "Use only the facts above -- do not invent or describe the image content."
    )


def build_qa_prompt(job, question, history):
    media, verdict = job["media"], job["verdict"]
    track = job.get("track")
    facts = {
        "file_type": media["media_type"],
        "duration": fmt_t(media.get("duration_s")) if media["media_type"] == "video" else None,
        "verdict": verdict["classification"],
        "manipulation_score_out_of_100": None if verdict.get("not_assessed") else r0(verdict["final_score"]),
        "no_face_detected": bool(verdict.get("no_face")),
        "confidence": verdict["confidence_level"],
        "confidence_reasons": track["confidence_reasons"] if track else [],
        "detectors": _signals(track) if track else {},
        "caveats": [c["text"] for c in track["caveats"]] if track else [],
        "detectors_not_used": [BRANCH_NAMES[b] + ": " + v.get("stage", "")
                               for b, v in job["branches"].items() if v.get("status") == "skipped"],
        "suspicious_segments": _segments(job.get("evidence")),
        "metadata": {
            "device": (media.get("device") or {}).get("label"),
            "platform": media.get("platform"),
            "software": media.get("software"),
            "created": media.get("created"),
            "findings": [f["text"] for f in media.get("findings", [])],
        },
        "decision_rules": {
            "verdict": "a trained Fusion model combines the detector scores; FAKE when its probability > 50",
            "classification": "Likely Fake / Likely Real need Medium or High confidence; otherwise Suspicious or Inconclusive",
            "confidence": "lowered when detectors conflict, a detector is unreliable or missing, or frames are blurry",
            "metadata_override": "A C2PA / IPTC AI declaration or AI-generator signature forces an AI-Generated verdict",
        },
        "image_measurements": (job.get("forensics") or {}).get("measurements"),
        "found_online": _online(job.get("source")),
        "agent_steps": [f"{s['title']}: {s['detail']}" for s in job["steps"]],
        "report_already_given": job.get("explanation", ""),
    }
    convo = "".join(f"\nAnalyst: {q}\nAgent: {a}" for q, a in history[-4:])
    return (
        "You are the forensic agent of a deepfake detection system, answering an analyst's follow-up question "
        f"about ONE analysis you just completed. Detectors: {DETECTORS}, plus a metadata/provenance check.\n\n"
        f"EVIDENCE FOR THIS FILE (JSON):\n{json.dumps(facts, indent=1)}\n"
        f"{('PREVIOUS QUESTIONS:' + convo) if convo else ''}\n\n"
        f"Analyst: {question}\n\n"
        "Answer in 2 to 4 sentences, plain text, no markdown. Use only the evidence above; cite scores and time "
        "ranges exactly as given. Never change the verdict. If the evidence cannot answer the question, say so "
        "plainly and say what additional analysis would be needed. Agent:"
    )


def stream(prompt, max_tokens=280, temperature=0.4, timeout=180):
    """Generator of text chunks from Ollama; raises if the server is unreachable."""
    with requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": True, "keep_alive": "60m",
              "options": {"temperature": temperature, "num_predict": max_tokens}},
        stream=True, timeout=timeout,
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            yield chunk.get("response", "")
            if chunk.get("done"):
                break
