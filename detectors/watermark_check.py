"""Invisible AI watermarks that have open decoders (invisible-watermark, DWT-DCT):
  - Stable Diffusion v1/v2 reference scripts: the text "StableDiffusionV1" (136 bits)
  - Stable Diffusion XL (diffusers / Stability API): a fixed 48-bit message
Many hosted SD services and apps built on these pipelines keep the watermark on by default.
Decoding works on pixels only, so it survives stripped metadata, but not strong re-compression,
resizing or cropping; a missing watermark proves nothing. Google SynthID and OpenAI watermarks
have no public decoders.

check(path_or_bgr) -> {"found": bool, "which": str|None, "match": float}
"""
import cv2
import numpy as np

SD1_TEXT = b"StableDiffusionV1"
SDXL_BITS = [int(b) for b in bin(0b101100111110110010010000011110111011000110011110)[2:]]   # from diffusers
MATCH = 0.80             # fraction of bits that must agree; clean photos: median 0.51, max 0.69 (n=512)


def _bits(b):
    return [int(x) for byte in b for x in f"{byte:08b}"]


def check(img):
    from imwatermark import WatermarkDecoder
    bgr = cv2.imread(img) if isinstance(img, str) else img
    if bgr is None or min(bgr.shape[:2]) < 256:
        return {"found": False, "which": None, "match": 0.0}
    best = (0.0, None)
    for name, ref in (("Stable Diffusion v1/v2", _bits(SD1_TEXT)), ("Stable Diffusion XL", SDXL_BITS)):
        try:
            dec = WatermarkDecoder("bits", len(ref)).decode(bgr, "dwtDct")
            m = float(np.mean(np.array(dec[:len(ref)], int) == np.array(ref)))
        except Exception:
            m = 0.0
        best = max(best, (m, name))
    return {"found": best[0] >= MATCH, "which": best[1] if best[0] >= MATCH else None, "match": round(best[0], 3)}
