"""
Real, pixel-level image forensics maps. Everything here is measured from the uploaded
file; nothing is simulated and nothing feeds the verdict score.

  ELA       error level analysis: re-save as JPEG and show where the recompression error
            differs. Regions with a different compression history stand out. Most
            informative on JPEG; PNG/screenshots show mostly uniform error.
  Spectrum  log-magnitude 2-D FFT. Periodic upsampling (GAN / diffusion decoders) can leave
            regular off-centre peaks; natural photos fall off smoothly from the centre.
  Noise     high-pass residual (image minus a denoised copy). Sensor noise is normally
            similar across a photo; a pasted or synthesised region can differ.

These are descriptive measurements for an analyst, not proof of manipulation.
"""

import base64

import cv2
import numpy as np


def _jpeg_b64(img, max_w=360):
    h, w = img.shape[:2]
    if w > max_w:
        img = cv2.resize(img, (max_w, max(1, int(h * max_w / w))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode() if ok else None


def _ela(img):
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return None, {}
    re = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    diff = cv2.absdiff(img, re).astype(np.float32).max(axis=2)
    p99 = float(np.percentile(diff, 99))
    scale = 255.0 / max(p99, 1.0)
    vis = cv2.applyColorMap(np.clip(diff * scale, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return vis, {"ela_mean": round(float(diff.mean()), 2), "ela_p99": round(p99, 1)}


def _spectrum(gray):
    g = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32)
    g *= np.outer(np.hanning(256), np.hanning(256))  # window to suppress edge leakage
    mag = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(g))))
    yy, xx = np.indices(mag.shape)
    r = np.hypot(yy - 128, xx - 128) / 128.0
    total = float(mag.sum())
    high = float(mag[r > 0.5].sum()) / total
    band = mag[(r > 0.35) & (r < 0.95)]
    peak = float(band.max() / (np.median(band) + 1e-6))
    vis = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(vis, cv2.COLORMAP_VIRIDIS), {
        "high_freq_share": round(high * 100, 1), "spectral_peak_ratio": round(peak, 2)}


def _noise(gray, face_box):
    g = gray.astype(np.float32)
    resid = g - cv2.GaussianBlur(g, (0, 0), 1.6)
    stats = {"noise_std": round(float(resid.std()), 2)}
    if face_box:
        x, y, w, h = face_box
        x, y = max(0, x), max(0, y)
        mask = np.zeros(g.shape, bool)
        mask[y:y + h, x:x + w] = True
        if mask.sum() > 400 and (~mask).sum() > 400:
            fs, bs = float(resid[mask].std()), float(resid[~mask].std())
            stats.update({"noise_face": round(fs, 2), "noise_background": round(bs, 2),
                          "noise_ratio": round(fs / (bs + 1e-6), 2)})
    vis = np.clip(np.abs(resid) * (255.0 / max(3 * resid.std(), 1e-3)), 0, 255).astype(np.uint8)
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_BONE)
    if face_box:
        x, y, w, h = face_box
        cv2.rectangle(vis, (x, y), (x + w, y + h), (94, 197, 34), max(2, vis.shape[1] // 300))
    return vis, stats


def analyze(path, media, face_box=None):
    """Returns {"maps": [...], "measurements": [plain-language strings]} or None."""
    img = cv2.imread(path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ela_vis, ela = _ela(img)
    spec_vis, spec = _spectrum(gray)
    noise_vis, noise = _noise(gray, face_box)

    is_jpeg = (media.get("mime") or "").lower() in ("image/jpeg", "image/jpg") or path.lower().endswith((".jpg", ".jpeg"))
    maps = [
        {"key": "ela", "title": "Error level analysis", "image": _jpeg_b64(ela_vis) if ela_vis is not None else None,
         "caption": "Brighter = more recompression error. Uneven patches can mean regions with a different editing history."
                    + ("" if is_jpeg else " Less informative here: not a JPEG.")},
        {"key": "spectrum", "title": "Frequency spectrum", "image": _jpeg_b64(spec_vis, 256),
         "caption": "Centre = low frequencies. Regular bright dots away from the centre can indicate synthetic upsampling."},
        {"key": "noise", "title": "Noise residual", "image": _jpeg_b64(noise_vis),
         "caption": "Sensor-like noise is normally similar everywhere." + (" Green box = detected face." if face_box else "")},
    ]

    m = []
    if ela:
        m.append(f"ELA: mean recompression error {ela['ela_mean']}, 99th percentile {ela['ela_p99']}.")
    m.append(f"Spectrum: {spec['high_freq_share']}% of spectral energy is in the outer half of frequencies; "
             f"strongest off-centre peak is {spec['spectral_peak_ratio']}x the band median.")
    if "noise_ratio" in noise:
        m.append(f"Noise: face region {noise['noise_face']} vs rest of image {noise['noise_background']} "
                 f"(ratio {noise['noise_ratio']}).")
    else:
        m.append(f"Noise: overall residual std {noise['noise_std']}.")
    return {"maps": maps, "measurements": m, "stats": {**ela, **spec, **noise}}
