"""
Metadata / provenance forensics.

What it reads, per container:
  MP4 / MOV / M4V -- mvhd creation time, iTunes-style (c)mak/(c)mod/(c)too
      atoms, QuickTime 'keys' metadata (Apple com.apple.quicktime.*, Android
      com.android.*), track handler names, vendor udta boxes (Samsung smta,
      GoPro FIRM/GPMF, DJI, Nikon NCDT, Panasonic PANA, Canon uuid), Sony
      NonRealTimeMeta XML, XMP packets.
  MKV / WebM -- EBML MuxingApp / WritingApp.
  AVI -- RIFF INFO software / date chunks.
  Images -- EXIF (Make/Model/Software/dates/lens), PNG text chunks
      (Stable Diffusion 'parameters', ComfyUI 'prompt'/'workflow'), XMP.
  Any file -- C2PA Content Credentials (c2pa-python) and IPTC
      DigitalSourceType = trainedAlgorithmicMedia (AI-generated marker).

Strength of evidence: a C2PA / IPTC AI declaration or an AI-generator
signature is strong. Device tags are weak (easy to strip or forge). Missing
metadata says nothing -- social platforms strip it on upload.
"""

import hashlib
import os
import re
import struct
from datetime import datetime, timedelta, timezone

import cv2

MP4_EPOCH = datetime(1904, 1, 1, tzinfo=timezone.utc)
CONTAINER_BOXES = {b"moov", b"udta", b"trak", b"mdia", b"minf", b"stbl", b"edts"}
XMP_UUID = bytes.fromhex("BE7ACFCB97A942E89C71999491E3AFAC")
AI_SOURCE_TYPES = ("trainedalgorithmicmedia", "compositewithtrainedalgorithmicmedia", "algorithmicmedia")

# vendor-specific MP4 boxes -> maker
VENDOR_BOXES = {
    b"smta": "Samsung", b"sefd": "Samsung", b"FIRM": "GoPro", b"GPMF": "GoPro", b"CAME": "GoPro",
    b"PANA": "Panasonic", b"NCDT": "Nikon", b"NCTG": "Nikon", b"djmd": "DJI", b"dbgi": "DJI",
    b"CNCV": "Canon", b"CNTH": "Canon", b"INSV": "Insta360",
}
HANDLER_HINTS = {
    "core media": "Apple", "gopro": "GoPro", "dji": "DJI", "ambarella": "Ambarella (action cam / dashcam)",
}

AI_TOOLS = [
    "stable diffusion", "comfyui", "automatic1111", "midjourney", "dall-e", "dall·e", "openai", "sora",
    "firefly", "imagen", "veo", "gemini", "runway", "pika", "kling", "luma", "dream machine", "hailuo",
    "synthesia", "heygen", "d-id", "deepfacelab", "faceswap", "roop", "facefusion", "invokeai", "leonardo",
    "ideogram", "flux", "grok", "meta ai",
]
EDITORS = [
    "premiere", "after effects", "davinci", "resolve", "final cut", "imovie", "capcut", "inshot", "kinemaster",
    "vn video", "powerdirector", "filmora", "vegas", "shotcut", "openshot", "lightroom", "photoshop", "gimp",
    "snapseed", "picsart", "canva", "clipchamp", "handbrake",
]
SCREEN_RECORDERS = ["obs", "screen record", "screenrecorder", "bandicam", "camtasia", "xbox game bar", "quicktime player"]
TRANSCODERS = ["lavf", "ffmpeg", "libav", "gstreamer"]

PHONE_MAKERS = ["apple", "samsung", "google", "xiaomi", "redmi", "poco", "oneplus", "oppo", "vivo", "realme",
                "motorola", "huawei", "honor", "nothing", "asus", "nokia", "hmd", "tecno", "infinix", "iqoo", "lava"]
CAMERA_MAKERS = ["canon", "nikon", "sony", "fujifilm", "panasonic", "olympus", "om digital", "leica", "hasselblad",
                 "pentax", "ricoh", "sigma", "blackmagic", "red digital", "arri"]
ACTION_MAKERS = ["gopro", "insta360", "dji", "akaso", "ambarella", "garmin"]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _clean(b):
    return b.decode("utf-8", "replace").replace("\x00", "").strip()


def _head_tail(path, head=2 << 20, tail=1 << 20):
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        data = f.read(head)
        if size > head:
            f.seek(max(head, size - tail))
            data += f.read(tail)
    return data


# ================================================================ MP4 / MOV
def _iter_boxes(f, start, end):
    pos = start
    while pos + 8 <= end:
        f.seek(pos)
        header = f.read(8)
        if len(header) < 8:
            return
        size, btype = struct.unpack(">I4s", header)
        hdr = 8
        if size == 1:
            size = struct.unpack(">Q", f.read(8))[0]
            hdr = 16
        elif size == 0:
            size = end - pos
        if size < hdr or pos + size > end + 8:
            return
        yield btype, pos + hdr, pos + size
        pos += size


def _read(f, start, end, limit=4096):
    f.seek(start)
    return f.read(max(0, min(end - start, limit)))


def _parse_keys_ilst(f, start, end, out):
    keys = []
    for btype, s, e in _iter_boxes(f, start, end):
        if btype == b"keys":
            f.seek(s + 4)
            count = struct.unpack(">I", f.read(4))[0]
            p = s + 8
            for _ in range(min(count, 128)):
                f.seek(p)
                ksize = struct.unpack(">I", f.read(4))[0]
                if ksize < 8:
                    break
                f.read(4)
                keys.append(_clean(f.read(ksize - 8)))
                p += ksize
        elif btype == b"ilst":
            for idx_box, is_, ie in _iter_boxes(f, s, e):
                idx = struct.unpack(">I", idx_box)[0] - 1
                for dtype, ds, de in _iter_boxes(f, is_, ie):
                    if dtype == b"data" and 0 <= idx < len(keys):
                        out[keys[idx]] = _clean(_read(f, ds + 8, de, 512))


def parse_mp4(path):
    out = {"atoms": {}, "keys": {}, "vendors": set(), "handlers": set(), "xml": "", "top": [], "zero_time": False}
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        def walk(start, end, depth):
            for btype, s, e in _iter_boxes(f, start, end):
                if depth == 0:
                    out["top"].append(btype)
                if btype == b"ftyp":
                    out["atoms"]["major_brand"] = _read(f, s, s + 4).decode("latin-1").strip()
                elif btype == b"mvhd":
                    f.seek(s)
                    version = f.read(1)[0]
                    f.read(3)
                    ctime = struct.unpack(">Q" if version == 1 else ">I", f.read(8 if version == 1 else 4))[0]
                    out["zero_time"] = ctime == 0
                    if 0 < ctime < 2 ** 33:
                        out["atoms"]["creation_time"] = (MP4_EPOCH + timedelta(seconds=ctime)).isoformat()
                elif btype == b"hdlr":
                    name = _clean(_read(f, s + 24, e, 256))
                    if name:
                        out["handlers"].add(name)
                elif btype[:1] == b"\xa9":
                    raw = _read(f, s, e, 512)
                    val = raw[16:] if raw[4:8] == b"data" else raw[4:]
                    key = btype[1:].decode("latin-1")
                    if key != "xyz":  # never keep GPS
                        out["atoms"][key] = _clean(val)
                elif btype in VENDOR_BOXES:
                    out["vendors"].add(VENDOR_BOXES[btype])
                    if btype == b"FIRM":
                        out["atoms"]["gopro_firmware"] = _clean(_read(f, s, e, 64))
                elif btype == b"uuid":
                    uid = _read(f, s, s + 16, 16)
                    body = _read(f, s + 16, e, 65536)
                    if uid == XMP_UUID:
                        out["xml"] += body.decode("utf-8", "replace")
                    else:
                        txt = body.decode("latin-1")
                        for marker, vendor in ((b"CNCV", "Canon"), (b"CNTH", "Canon"), (b"PROF", "Sony")):
                            if marker in body:
                                out["vendors"].add(vendor)
                        if "<?xml" in txt or "NonRealTimeMeta" in txt:
                            out["xml"] += txt
                elif btype == b"xml ":
                    out["xml"] += _read(f, s + 4, e, 65536).decode("utf-8", "replace")
                elif btype == b"meta":
                    f.seek(s)
                    peek = f.read(8)
                    inner = s + 4 if peek[:4] == b"\x00\x00\x00\x00" else s
                    _parse_keys_ilst(f, inner, e, out["keys"])
                    walk(inner, e, depth + 1)
                elif btype in CONTAINER_BOXES and depth < 8:
                    walk(s, e, depth + 1)
        walk(0, size, 0)

    # Sony / Panasonic XDCAM-style XML: <Device manufacturer="Sony" modelName="ILCE-7M3"/>
    m = re.search(r'<Device[^>]*manufacturer="([^"]+)"[^>]*modelName="([^"]+)"', out["xml"])
    if m:
        out["atoms"]["xml_make"], out["atoms"]["xml_model"] = m.group(1), m.group(2)
    return out


# ================================================================ MKV / AVI
def _ebml_string(data, elem_id):
    """Find an EBML string element (e.g. 0x4D80 MuxingApp) and return its value."""
    i = data.find(elem_id)
    while i != -1:
        j = i + len(elem_id)
        if j < len(data):
            first = data[j]
            length_bytes = next((n for n in range(1, 9) if first & (0x80 >> (n - 1))), None)
            if length_bytes:
                size = first & (0xFF >> length_bytes)
                for k in range(1, length_bytes):
                    size = (size << 8) | data[j + k]
                val = data[j + length_bytes: j + length_bytes + size]
                if 0 < size < 256 and all(32 <= c < 127 for c in val):
                    return val.decode()
        i = data.find(elem_id, i + 1)
    return None


def parse_mkv(path):
    data = _head_tail(path, head=256 << 10, tail=0)
    out = {}
    for name, eid in (("muxing_app", b"\x4d\x80"), ("writing_app", b"\x57\x41")):
        v = _ebml_string(data, eid)
        if v:
            out[name] = v
    return out


def parse_avi(path):
    data = _head_tail(path, head=256 << 10, tail=0)
    out = {}
    for tag, name in ((b"ISFT", "software"), (b"IDIT", "creation_time"), (b"ICRD", "creation_time"), (b"IMAK", "make")):
        i = data.find(tag)
        if i != -1:
            size = struct.unpack("<I", data[i + 4:i + 8])[0]
            if 0 < size < 256:
                out.setdefault(name, _clean(data[i + 8:i + 8 + size]))
    return out


# ================================================================ images
def parse_image(path):
    fields, extra_text = {}, ""
    fmt = None
    try:
        from PIL import Image, ExifTags
        img = Image.open(path)
        fmt = img.format
        exif = img.getexif()
        tags = {ExifTags.TAGS.get(k, str(k)): v for k, v in exif.items()}
        for ifd in (0x8769,):
            try:
                tags.update({ExifTags.TAGS.get(k, str(k)): v for k, v in exif.get_ifd(ifd).items()})
            except Exception:
                pass
        for k in ("Make", "Model", "Software", "DateTimeOriginal", "DateTime", "LensModel", "HostComputer",
                  "Artist", "ImageDescription"):
            if k in tags and str(tags[k]).strip():
                fields[k] = str(tags[k]).strip("\x00 ")
        # PNG / WebP text chunks -- where Stable Diffusion & ComfyUI write their settings
        for k, v in (img.info or {}).items():
            if isinstance(v, (str, bytes)) and k.lower() in (
                    "parameters", "prompt", "workflow", "software", "description", "comment", "xmp",
                    "xml:com.adobe.xmp", "generator", "ai_generated", "source"):
                s = v.decode("utf-8", "replace") if isinstance(v, bytes) else v
                extra_text += "\n" + s
                if k.lower() in ("parameters", "prompt", "workflow"):
                    fields[f"png_{k.lower()}"] = s[:80] + ("…" if len(s) > 80 else "")
                elif k.lower() in ("software", "generator"):
                    fields.setdefault("Software", s[:80])
    except Exception:
        pass
    return fields, fmt, extra_text


# ================================================================ C2PA
def read_c2pa(path):
    try:
        import c2pa
        import json
        with c2pa.Reader(path) as reader:
            store = json.loads(reader.json())
    except Exception as e:
        msg = str(e).lower().replace(" ", "")
        if "notfound" in msg or "nomanifest" in msg or "jumbf" in msg:
            return {"present": False}
        return {"present": False, "error": type(e).__name__}

    active = store.get("manifests", {}).get(store.get("active_manifest"), {})
    generator = active.get("claim_generator") or \
        ", ".join(g.get("name", "") for g in active.get("claim_generator_info", []))
    ai_flags = []
    for a in active.get("assertions", []):
        if a.get("label", "").startswith("c2pa.actions"):
            for act in a.get("data", {}).get("actions", []):
                dst = (act.get("digitalSourceType") or "").lower()
                if any(t in dst for t in AI_SOURCE_TYPES):
                    ai_flags.append(f"{act.get('action', 'action')}: {dst.rsplit('/', 1)[-1]}")
    return {
        "present": True,
        "claim_generator": generator,
        "issuer": (active.get("signature_info") or {}).get("issuer"),
        "validation_state": store.get("validation_state"),
        "ai_generated_or_edited": bool(ai_flags),
        "ai_actions": ai_flags,
    }


# ================================================================ classify
def _classify_device(make, model, vendors):
    text = f"{make or ''} {model or ''} {' '.join(vendors)}".lower()
    if any(m in text for m in ACTION_MAKERS):
        if "dji" in text and ("fc" in (model or "").lower() or "mavic" in text or "mini" in text or "air" in text):
            return "drone"
        return "action camera"
    if any(m in text for m in CAMERA_MAKERS) and not re.search(r"xperia|so-\d|xq-", text):
        return "camera"
    if any(m in text for m in PHONE_MAKERS) or re.search(r"iphone|pixel|sm-[a-z]\d|galaxy|xperia", text):
        return "phone"
    return "device" if text.strip() else None


def probe(path, filename):
    ext = os.path.splitext(filename)[1].lower()
    is_video = ext in {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".3gp"}
    is_audio = ext in {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac"}
    info = {
        "filename": filename,
        "size_bytes": os.path.getsize(path),
        "sha256": _sha256(path),
        "media_type": "video" if is_video else ("audio" if is_audio else "image"),
        "fields": {},
    }
    findings = []
    vendors, handlers, xml_text = set(), set(), ""
    platform = None

    if is_video:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        cap.release()
        info.update({
            "fps": round(fps, 2),
            "frame_count": frames,
            "duration_s": round(frames / fps, 2) if fps else None,
            "resolution": f"{w}x{h}",
            "codec": "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)).strip("\x00") or None,
        })
        try:
            if ext in {".mp4", ".mov", ".m4v", ".3gp"}:
                mp4 = parse_mp4(path)
                info["fields"].update(mp4["atoms"])
                info["fields"].update({k.split(".")[-1]: v for k, v in mp4["keys"].items()})
                vendors, handlers, xml_text = mp4["vendors"], mp4["handlers"], mp4["xml"]
                # WhatsApp's re-encoder: 'beam' box after ftyp, zeroed timestamps, blank handler names
                if b"beam" in mp4["top"][:3] and mp4["zero_time"] and not handlers:
                    platform = "WhatsApp"
            elif ext in {".mkv", ".webm"}:
                info["fields"].update(parse_mkv(path))
            elif ext == ".avi":
                info["fields"].update(parse_avi(path))
        except Exception as e:
            findings.append({"level": "info", "text": f"Container parse incomplete ({type(e).__name__})."})
    elif is_audio:
        try:
            import soundfile as sf
            sf_info = sf.info(path)
            info.update({
                "duration_s": round(sf_info.duration, 2),
                "samplerate": sf_info.samplerate,
                "channels": sf_info.channels,
                "format": sf_info.format,
            })
        except Exception:
            pass
    else:
        fields, fmt, xml_text = parse_image(path)
        info["fields"].update(fields)
        info["format"] = fmt
        img = cv2.imread(path)
        if img is not None:
            info["resolution"] = f"{img.shape[1]}x{img.shape[0]}"

    if not platform and re.match(r"^WhatsApp (Video|Image) \d{4}-\d{2}-\d{2}", filename):
        platform = "WhatsApp"
    info["platform"] = platform

    # Never surface location; drop binary / numeric-key junk.
    info["fields"] = {
        k: v for k, v in info["fields"].items()
        if not any(t in k.lower() for t in ("iso6709", "xyz", "gps", "location"))
        and not k.isdigit() and isinstance(v, str) and v and v.isprintable()
    }
    f = info["fields"]

    # ---------------- normalise device / software / date ----------------
    make = f.get("Make") or f.get("make") or f.get("manufacturer") or f.get("mak") or f.get("xml_make")
    model = f.get("Model") or f.get("model") or f.get("mod") or f.get("xml_model")
    for h in handlers:
        for hint, vendor in HANDLER_HINTS.items():
            if hint in h.lower() and vendor not in vendors:
                vendors.add(vendor)
    if not make and vendors:
        make = sorted(vendors)[0]
    if make and "gopro" in make.lower() and not model and f.get("gopro_firmware"):
        model = f"firmware {f['gopro_firmware']}"
    android_ver = f.get("version") if "version" in f and re.match(r"^\d+(\.\d+)*$", f.get("version", "")) else None
    software = f.get("too") or f.get("Software") or f.get("software") or f.get("swr") or \
        f.get("writing_app") or f.get("muxing_app")
    created = f.get("DateTimeOriginal") or f.get("creationdate") or f.get("creation_time") or f.get("DateTime")
    device_class = _classify_device(make, model, vendors)
    if android_ver and not make:
        device_class, make = "phone", "Android device"

    info["device"] = {"label": None, "make": make, "model": model, "class": device_class,
                      "os": (f"Android {android_ver}" if android_ver else
                             f"iOS {f['software']}" if make == "Apple" and f.get("software") else None)}
    info["device"]["label"] = (model if (model and make and model.lower().startswith(make.lower()))
                               else " ".join(dict.fromkeys(x for x in (make, model) if x))) or None
    if make == "Apple" and software and software == f.get("software"):
        software = None  # iOS version, already shown as OS
    info["software"] = software
    info["created"] = created

    # ---------------- provenance: C2PA + IPTC + AI signatures ----------------
    c2pa_info = read_c2pa(path)
    info["c2pa"] = c2pa_info
    blob = (xml_text + " " + " ".join(str(v) for v in f.values()) + " " + " ".join(handlers)).lower()
    if is_video:
        try:
            blob += " " + _head_tail(path, head=1 << 20, tail=512 << 10).decode("latin-1").lower()
        except Exception:
            pass
    iptc_ai = any(t in blob for t in AI_SOURCE_TYPES)
    ai_tools = sorted({t for t in AI_TOOLS if re.search(rf"(?<![a-z]){re.escape(t)}(?![a-z])", (software or "").lower() + " " + xml_text.lower() + " " + " ".join(str(v).lower() for k, v in f.items() if k.startswith("png_") or k in ("Software", "ImageDescription", "Artist")))})
    if any(k.startswith("png_") for k in f):
        ai_tools = sorted(set(ai_tools) | {"Stable Diffusion / ComfyUI (generation parameters embedded)"})

    assessment = "stripped"
    if c2pa_info.get("present"):
        state = c2pa_info.get("validation_state")
        signer = c2pa_info.get("claim_generator") or "unknown generator"
        if c2pa_info.get("ai_generated_or_edited"):
            assessment = "ai_provenance"
            findings.append({"level": "high", "text":
                f"C2PA Content Credentials declare AI involvement ({'; '.join(c2pa_info['ai_actions'])}), signed by {signer}."})
        else:
            assessment = "c2pa_signed"
            findings.append({"level": "low", "text":
                f"Signed C2PA Content Credentials from {signer}"
                f"{f' (signature {state.lower()})' if state else ''}; no AI actions declared."})
    else:
        findings.append({"level": "info", "text": "No C2PA Content Credentials embedded."})

    if iptc_ai and assessment != "ai_provenance":
        assessment = "ai_provenance"
        findings.append({"level": "high", "text":
            "IPTC DigitalSourceType = trainedAlgorithmicMedia found in embedded XMP -- the file declares itself AI-generated."})
    if ai_tools:
        if assessment not in ("ai_provenance",):
            assessment = "ai_tool_signature"
        findings.append({"level": "high", "text": f"AI generator signature in metadata: {', '.join(ai_tools)}."})

    if make or model:
        desc = model if (model and make and model.lower().startswith(make.lower())) else \
            " ".join(dict.fromkeys(x for x in (make, model) if x))
        extra = f" ({info['device']['os']})" if info["device"]["os"] else ""
        findings.append({"level": "low", "text": f"Capture device: {desc}{extra} -- {device_class or 'device'}."})
        if assessment == "stripped":
            assessment = "camera_original"
    elif android_ver:
        findings.append({"level": "low", "text": f"Recorded on an Android {android_ver} device (maker not written by this phone)."})
        if assessment == "stripped":
            assessment = "camera_original"
    if "Samsung" in vendors:
        findings.append({"level": "low", "text": "Samsung camera atom present -- consistent with on-device Samsung capture."})

    sw = (software or "").lower()
    if software:
        if any(t in sw for t in SCREEN_RECORDERS):
            findings.append({"level": "medium", "text": f"Created by screen-recording software ({software}); not a camera original."})
            if assessment in ("stripped", "camera_original"):
                assessment = "screen_recording"
        elif any(t in sw for t in EDITORS):
            findings.append({"level": "medium", "text": f"Processed in editing software ({software}); content may have been altered after capture."})
            if assessment in ("stripped", "camera_original"):
                assessment = "edited"
        elif any(t in sw for t in TRANSCODERS):
            findings.append({"level": "medium", "text":
                f"Encoded by {software} (FFmpeg-based): re-encoded after capture. Common for platform downloads "
                f"and for face-swap tool output, so a weak signal only."})
            if assessment == "stripped":
                assessment = "re_encoded"
        elif not ai_tools and not (make == "Apple" and software == f.get("software")):
            findings.append({"level": "info", "text": f"Software tag: {software}."})

    if platform:
        findings.append({"level": "medium", "text":
            f"Shared via {platform}: the app re-compresses media and removes device, date and location tags, "
            f"so the original capture source cannot be verified. Not evidence of manipulation on its own."})
        if assessment == "stripped":
            assessment = "platform_reencoded"
    elif not make and not android_ver and not created and assessment == "stripped":
        findings.append({"level": "info", "text":
            "No capture device or timestamp -- typical of social-media downloads; not evidence of manipulation on its own."})

    info["assessment"] = assessment
    info["findings"] = findings
    return info
