"""EXIF consistency checks for JPEG photos: does the camera metadata hang together?
backend/metadata_probe.py already READS the tags (device, software, dates, C2PA, AI signatures).
This module checks whether they CONTRADICT each other or the pixels, which is what an edit
followed by copying the original metadata back typically leaves behind:

  1. Embedded EXIF thumbnail vs the main image: the camera writes both at capture. If the picture
     was edited but the old EXIF was kept, the thumbnail still shows the original (Kee & Farid 2010).
  2. Picture size vs the size recorded in EXIF (PixelXDimension/YDimension): cropped or resized.
  3. Dates: modify date later than capture date (re-saved), capture date in the future, or impossible.
  4. JPEG quantization tables: cameras use their own tables; exactly the standard libjpeg (IJG)
     tables mean the file was last saved by software (Photoshop/GIMP/Python/apps). Reported for
     dedicated cameras only, because many phones legitimately use the standard tables.
  5. Camera brands that always write a MakerNote (Canon, Nikon, Sony, ...) but none is present:
     the EXIF was rewritten by software.

check(path, device_class) -> list of {"level", "text", "kind"}  (empty for non-JPEG / no EXIF)
Each finding is evidence about the FILE's history, never proof about the content.
"""
import io
from datetime import datetime, timedelta

import cv2
import numpy as np
from PIL import ExifTags, Image

MAKERNOTE_BRANDS = ("canon", "nikon", "sony", "fujifilm", "olympus", "panasonic", "pentax", "leica")
# standard IJG luminance table (quality 50); other qualities are scaled versions of it
IJG_LUMA = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
                     14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
                     49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99])
ZIGZAG = np.array([0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6,
                   7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31,
                   39, 46, 53, 60, 61, 54, 47, 55, 62, 63])


def _ijg_quality(table):
    """Returns the IJG quality (1-100) if `table` is exactly a standard libjpeg table, else None."""
    t = np.asarray(table)
    candidates = [t, np.empty(64, int)]
    candidates[1][ZIGZAG] = t                                    # PIL may give zigzag or natural order
    for q in range(1, 101):
        s = 5000 / q if q < 50 else 200 - 2 * q
        std = np.clip((IJG_LUMA * s + 50) // 100, 1, 255)
        if any(np.array_equal(c, std) for c in candidates):
            return q
    return None


def _thumbnail(path):
    """JPEG thumbnail bytes inside the EXIF APP1 segment, or None."""
    data = open(path, "rb").read(256 * 1024)
    i = data.find(b"\xff\xe1")
    if i < 0 or data[i + 4:i + 10] != b"Exif\x00\x00":
        return None
    seg = data[i + 4:i + 2 + int.from_bytes(data[i + 2:i + 4], "big")]
    s = seg.find(b"\xff\xd8", 10)
    e = seg.find(b"\xff\xd9", s)
    return seg[s:e + 2] if s > 0 and e > s else None


def _date(v):
    try:
        return datetime.strptime(str(v).strip("\x00 ")[:19], "%Y:%m:%d %H:%M:%S")
    except Exception:
        return None


def check(path, device_class=None):
    out = []
    try:
        img = Image.open(path)
        if img.format != "JPEG":
            return out
        exif = img.getexif()
        tags = {ExifTags.TAGS.get(k, str(k)): v for k, v in exif.items()}
        try:
            tags.update({ExifTags.TAGS.get(k, str(k)): v for k, v in exif.get_ifd(0x8769).items()})
        except Exception:
            pass
    except Exception:
        return out
    if not tags:
        return out
    make = str(tags.get("Make", "")).lower()
    w, h = img.size

    # 1. thumbnail vs main image
    thumb = _thumbnail(path)
    if thumb:
        t = cv2.imdecode(np.frombuffer(thumb, np.uint8), cv2.IMREAD_GRAYSCALE)
        if t is not None and t.size > 0:
            main = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
            th, tw = t.shape
            if abs(tw / th - w / h) > 0.05 and abs(tw / th - h / w) > 0.05:
                out.append({"level": "high", "kind": "thumbnail", "text":
                            f"EXIF thumbnail aspect ratio ({tw}x{th}) does not match the picture ({w}x{h}): "
                            "the picture was cropped or altered after capture while the original EXIF was kept."})
            else:
                # thumbnails are often stored unrotated (EXIF Orientation): compare against every rotation
                # of the picture with a matching aspect ratio and keep the best match
                b = t.astype(np.float32)
                corr, local = -1.0, 1e9
                for k in range(4):
                    m = np.rot90(main, k)
                    if abs(tw / th - m.shape[1] / m.shape[0]) > 0.05:
                        continue
                    a = cv2.resize(np.ascontiguousarray(m), (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32)
                    c = float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) if a.std() > 0 and b.std() > 0 else 1.0
                    if c > corr:
                        corr = c
                        local = float(cv2.blur(np.abs(a - b), (max(3, tw // 10), max(3, th // 10))).max())
                if corr < 0.90:
                    out.append({"level": "high", "kind": "thumbnail", "text":
                                f"EXIF thumbnail shows a different picture (similarity {corr:.2f}): the image content was "
                                "changed after capture while the original EXIF was kept."})
                elif local > 40:
                    out.append({"level": "high", "kind": "thumbnail", "text":
                                "EXIF thumbnail differs from the picture in one area: that region was likely edited "
                                "after capture while the original EXIF was kept."})
                else:
                    out.append({"level": "low", "kind": "thumbnail", "text":
                                f"EXIF thumbnail matches the picture (similarity {corr:.2f})."})

    # 2. recorded size vs actual size
    xd, yd = tags.get("ExifImageWidth"), tags.get("ExifImageHeight")
    try:
        xd, yd = int(xd), int(yd)
        if xd > 0 and yd > 0 and {xd, yd} != {w, h}:
            out.append({"level": "medium", "kind": "size", "text":
                        f"EXIF records {xd}x{yd} but the picture is {w}x{h}: resized or cropped after capture."})
    except (TypeError, ValueError):
        pass

    # 3. dates
    orig, mod = _date(tags.get("DateTimeOriginal")), _date(tags.get("DateTime"))
    if orig and orig > datetime.now() + timedelta(days=1):
        out.append({"level": "medium", "kind": "date", "text": f"Capture date {orig:%Y-%m-%d} is in the future."})
    if orig and mod and mod - orig > timedelta(minutes=2):
        out.append({"level": "low", "kind": "date", "text":
                    f"Modified {mod:%Y-%m-%d %H:%M}, {(mod - orig).days} day(s) after capture ({orig:%Y-%m-%d %H:%M}): "
                    "the file was re-saved later (photo-transfer and rotation tools also do this)."})

    # 4. quantization tables (dedicated cameras only)
    if device_class == "camera":
        try:
            q = _ijg_quality(img.quantization[0])
            if q is not None:
                out.append({"level": "medium", "kind": "quant", "text":
                            f"JPEG compression uses the standard libjpeg tables (quality {q}), not the camera's own: "
                            "last saved by software, not straight from the camera."})
        except Exception:
            pass

    # 5. MakerNote missing for brands that always write one
    if any(b in make for b in MAKERNOTE_BRANDS) and 0x927C not in (exif.get_ifd(0x8769) or {}):
        out.append({"level": "medium", "kind": "makernote", "text":
                    f"{tags.get('Make')} cameras always write a MakerNote, but it is missing: the EXIF was rewritten "
                    "by software."})
    return out
