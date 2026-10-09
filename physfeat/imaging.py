"""Image I/O shared by every feature extractor.

* `read_meta`      – format, size, JPEG quality, hash (for the bias audit / dedupe)
* `load_rgb`       – decode an image the same way for every extractor, optionally
                     neutralising the PNG-vs-JPEG shortcut and applying a
                     robustness perturbation (JPEG re-compression, resizing, blur).
"""
from __future__ import annotations

import hashlib
import io

import numpy as np
from PIL import Image, ImageFile, ImageFilter

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".jfif"}
NORMALISE_JPEG_Q = 96  # GenImage de-biasing protocol (Grommelt et al., 2024)

# ----------------------------------------------------------------------------
# JPEG quality estimation (compare the luminance table with Pillow's own tables)
# ----------------------------------------------------------------------------
_QREF = None


def _qref():
    global _QREF
    if _QREF is None:
        tiny = Image.new("RGB", (16, 16), (128, 128, 128))
        ref = {}
        for q in range(1, 101):
            buf = io.BytesIO()
            tiny.save(buf, "JPEG", quality=q)
            ref[q] = np.array(Image.open(io.BytesIO(buf.getvalue())).quantization[0], float)
        _QREF = ref
    return _QREF


def estimate_jpeg_quality(im: Image.Image) -> float:
    q = getattr(im, "quantization", None)
    if not q or 0 not in q:
        return float("nan")
    t = np.array(q[0], float)
    ref = _qref()
    return float(min(ref, key=lambda k: np.abs(ref[k] - t).mean()))


def file_md5(path, chunk=1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def read_meta(path) -> dict:
    out = {"fmt": None, "width": np.nan, "height": np.nan, "jpeg_q": np.nan,
           "bytes": np.nan, "md5": None, "meta_error": None}
    try:
        import os
        out["bytes"] = os.path.getsize(path)
        out["md5"] = file_md5(path)
        with Image.open(path) as im:
            out["fmt"] = im.format
            out["width"], out["height"] = im.size
            if im.format == "JPEG":
                out["jpeg_q"] = estimate_jpeg_quality(im)
    except Exception as e:  # corrupt / unreadable
        out["meta_error"] = repr(e)[:200]
    return out


# ----------------------------------------------------------------------------
# decoding + perturbations
# ----------------------------------------------------------------------------
def jpeg_roundtrip(im: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def _resize(im, s):
    w, h = im.size
    return im.resize((max(8, round(w * s)), max(8, round(h * s))), Image.BICUBIC)


PERTURBATIONS = {
    None: lambda im: im,
    "jpeg90": lambda im: jpeg_roundtrip(im, 90),
    "jpeg75": lambda im: jpeg_roundtrip(im, 75),
    "jpeg50": lambda im: jpeg_roundtrip(im, 50),
    "down0.5": lambda im: _resize(im, 0.5),
    "blur1": lambda im: im.filter(ImageFilter.GaussianBlur(1.0)),
}


JPEGRAND_RANGE = (50, 95)   # training-time augmentation: per-image JPEG quality, deterministic per file


def jpegrand_quality(path) -> int:
    import os
    lo, hi = JPEGRAND_RANGE
    return lo + stable_seed(os.path.basename(str(path))) % (hi - lo + 1)


def effective_quality(perturb, original_q=float("nan")) -> float:
    """JPEG quality of the image the detector actually sees (what a deployed system
    reads from the file header). Resizing/blur keep the original quality."""
    if perturb and perturb.startswith("jpeg") and perturb[4:].isdigit():
        return float(perturb[4:])
    q = original_q
    return float(NORMALISE_JPEG_Q if not np.isfinite(q) else q)


def load_pil(path, normalise_format: bool = True, perturb: str | None = None) -> Image.Image:
    with Image.open(path) as im:
        fmt = im.format
        im = im.convert("RGB")
    if normalise_format and fmt != "JPEG":
        # PNG fakes vs JPEG reals is a known dataset shortcut: give every image a
        # JPEG history so the detector cannot key on the container format.
        im = jpeg_roundtrip(im, NORMALISE_JPEG_Q)
    if perturb == "jpegrand":
        return jpeg_roundtrip(im, jpegrand_quality(path))
    return PERTURBATIONS[perturb](im)


def load_rgb(path, normalise_format: bool = True, perturb: str | None = None) -> np.ndarray:
    return np.asarray(load_pil(path, normalise_format, perturb))


def stable_seed(path) -> int:
    return int(hashlib.md5(str(path).encode()).hexdigest()[:8], 16)