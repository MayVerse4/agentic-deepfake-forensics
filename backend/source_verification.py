"""
Source verification: where else does this media appear online?

Reverse image search through SerpAPI's Google Lens engine. Frames are sent
with SerpAPI's image upload (expires after 10 minutes, never made public),
so no third-party image host is involved.

    image -> the image itself        video -> 2 frames (25% and 60% through)

One Google Lens search per frame; the free SerpAPI plan has 250 searches a
month. Results are cached by the file's SHA-256 (backend/source_cache.json),
so re-uploading the same file costs nothing.

Matches are context for the analyst (earlier copies, original source), not
evidence of manipulation: they never change the detector score.

Key: SERPAPI_KEY in the environment or in the project-root .env file.
Without a key everything reports status "not_configured".
"""

import json
import os
import threading
import time
from urllib.parse import urlparse

import cv2
import requests

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE_FILE = os.path.join(os.path.dirname(__file__), "source_cache.json")
UPLOAD_URL = "https://serpapi.com/image"
SEARCH_URL = "https://serpapi.com/search.json"
ACCOUNT_URL = "https://serpapi.com/account.json"
MAX_UPLOAD_BYTES = 500 * 1024  # SerpAPI image upload limit
MAX_MATCHES = 10

_cache_lock = threading.Lock()
_account = {"t": 0.0, "value": None}


def _api_key():
    key = os.environ.get("SERPAPI_KEY")
    if key:
        return key.strip()
    env_path = os.path.join(PROJECT_ROOT, ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                name, _, value = line.strip().partition("=")
                if name.strip() == "SERPAPI_KEY" and value.strip():
                    return value.strip().strip("'\"")
    return None


def status():
    """For /health: whether a key is set and how many searches are left (cached 5 min)."""
    key = _api_key()
    if not key:
        return {"configured": False, "searches_left": None}
    if time.time() - _account["t"] > 300:
        try:
            r = requests.get(ACCOUNT_URL, params={"api_key": key}, timeout=5)
            r.raise_for_status()
            _account["value"] = r.json().get("total_searches_left")
        except Exception:
            _account["value"] = None
        _account["t"] = time.time()
    return {"configured": True, "searches_left": _account["value"]}


def _load_cache():
    try:
        with open(CACHE_FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_cache(sha, result):
    with _cache_lock:
        cache = _load_cache()
        cache[sha] = result
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f)


def _encode(frame):
    """JPEG under the 500 KB upload limit: shrink / lower quality until it fits."""
    img = frame
    for max_side, quality in ((1280, 88), (1024, 85), (800, 80), (640, 75)):
        h, w = frame.shape[:2]
        if max(h, w) > max_side:
            s = max_side / max(h, w)
            img = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok and len(buf) <= MAX_UPLOAD_BYTES:
            return buf.tobytes()
    raise ValueError("could not encode frame under the upload size limit")


def _frames(path, media):
    if media["media_type"] == "image":
        img = cv2.imread(path)
        return [("image", img)] if img is not None else []
    dur = media.get("duration_s") or 0
    out = []
    for q in (0.25, 0.6):
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_MSEC, dur * q * 1000)
        ok, frame = cap.read()
        cap.release()
        if ok:
            out.append((f"frame at {int(dur * q // 60)}:{dur * q % 60:04.1f}", frame))
    return out


def _search_frame(key, frame):
    up = requests.post(UPLOAD_URL, params={"api_key": key},
                       files={"image": ("frame.jpg", _encode(frame), "image/jpeg")}, timeout=30)
    up.raise_for_status()
    image_id = up.json().get("image_id")
    if not image_id:
        raise ValueError(f"upload returned no image_id: {up.text[:200]}")
    r = requests.get(SEARCH_URL, timeout=60, params={
        "engine": "google_lens", "image_id": image_id, "type": "all", "api_key": key})
    r.raise_for_status()
    data = r.json()
    if data.get("error"):
        raise ValueError(data["error"])
    matches = []
    for kind in ("exact_matches", "visual_matches"):
        for m in data.get(kind) or []:
            link = m.get("link")
            if not link:
                continue
            matches.append({
                "kind": "exact" if kind == "exact_matches" else "visual",
                "title": m.get("title") or "",
                "source": m.get("source") or urlparse(link).hostname or "",
                "domain": (urlparse(link).hostname or "").removeprefix("www."),
                "link": link,
                "thumbnail": m.get("thumbnail"),
                "date": m.get("date"),
            })
    return matches


def verify(path, media):
    """Returns {"status", "matches", "searched", "domains", "message", "cached"}."""
    key = _api_key()
    if not key:
        return {"status": "not_configured", "matches": [], "searched": [], "domains": [], "cached": False,
                "message": "Source verification is not configured (no SERPAPI_KEY)."}
    sha = media.get("sha256")
    if sha:
        cached = _load_cache().get(sha)
        if cached:
            return {**cached, "cached": True}

    matches, searched, errors = [], [], []
    for label, frame in _frames(path, media):
        try:
            matches.extend(_search_frame(key, frame))
            searched.append(label)
        except Exception as e:
            errors.append(f"{label}: {type(e).__name__}: {e}")
    if not searched:
        return {"status": "error", "matches": [], "searched": [], "domains": [], "cached": False,
                "message": "Reverse image search failed. " + "; ".join(errors)[:300]}

    # exact matches first, one entry per link, capped
    seen, unique = set(), []
    for m in sorted(matches, key=lambda m: m["kind"] != "exact"):
        if m["link"] not in seen:
            seen.add(m["link"])
            unique.append(m)
    unique = unique[:MAX_MATCHES]
    domains = sorted({m["domain"] for m in unique if m["domain"]})
    n_exact = sum(m["kind"] == "exact" for m in unique)
    message = (f"No copies found online for the {len(searched)} searched frame(s)." if not unique else
               f"{len(unique)} match(es) online ({n_exact} exact) across {len(domains)} site(s).")
    result = {"status": "done", "matches": unique, "searched": searched, "domains": domains,
              "message": message, "cached": False}
    if sha:
        _save_cache(sha, result)
    return result
