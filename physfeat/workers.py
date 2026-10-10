"""Picklable per-image workers for multiprocessing pools."""
from __future__ import annotations

import numpy as np

from . import geometry
from .imaging import load_rgb, read_meta, stable_seed


def meta_worker(path):
    return read_meta(path)


def geometry_worker(args):
    """args = (path, normalise_format, perturb). Never raises."""
    path, normalise, perturb = args
    try:
        rgb = load_rgb(path, normalise, perturb)
        feats = geometry.extract(rgb, seed=stable_seed(path)).features
        feats["geo_error"] = None
    except Exception as e:
        feats = {k: np.nan for k in geometry.FEATURE_NAMES}
        feats["geo_error"] = repr(e)[:200]
    return feats

# ---- face pipeline (notebook 05) -------------------------------------------------------
def crop_face_worker(args):
    """args = (src_path, out_path, size, margin, jpeg_q). Detects the face, writes a square
    canonical crop as JPEG and returns metadata about the ORIGINAL file. Never raises."""
    src, out, size, margin, q = args
    meta = {"crop_ok": False, "crop_reason": None, "src_fmt": None, "src_w": np.nan, "src_h": np.nan,
            "src_jpeg_q": np.nan, "src_face_px": np.nan}
    try:
        from PIL import Image

        from . import face
        from .imaging import estimate_jpeg_quality
        with Image.open(src) as im:
            meta.update(src_fmt=im.format, src_w=im.size[0], src_h=im.size[1])
            if im.format == "JPEG":
                meta["src_jpeg_q"] = estimate_jpeg_quality(im)
            rgb = np.asarray(im.convert("RGB"))
        L = face.default_mesher()(rgb)
        if L is None:
            meta["crop_reason"] = "no_face"
            return meta
        meta["src_face_px"] = float(np.ptp(L[:, 0]))
        Image.fromarray(face.face_crop(rgb, L, size=size, margin=margin)).save(out, "JPEG", quality=q)
        meta["crop_ok"] = True
    except Exception as e:
        meta["crop_reason"] = repr(e)[:200]
    return meta


def face_features_worker(args):
    """args = (crop_path, perturb). Face-physics features of a (perturbed) crop. Never raises."""
    path, perturb = args
    from . import face
    try:
        return face.extract_features(load_rgb(path, False, perturb))
    except Exception as e:
        out = {k: np.nan for k in face.FEATURE_NAMES}
        out["face_error"] = repr(e)[:200]
        return out


IMAGE_STAT_NAMES = ["st_lum_mean", "st_lum_std", "st_sat_mean", "st_lap_var", "st_hf_ratio"]


def image_stats_worker(path):
    """Low-level image statistics for the shortcut audit (brightness, contrast, saturation,
    sharpness, high-frequency energy). If these alone separate real from fake on a test set,
    that set's real and fake images differ in ways that have nothing to do with fakeness."""
    import cv2
    try:
        rgb = load_rgb(path, False, None).astype(np.float32) / 255
        g = rgb.mean(-1)
        sat = rgb.max(-1) - rgb.min(-1)
        lap = cv2.Laplacian(g, cv2.CV_32F)
        F = np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean())))
        h, w = g.shape
        yy, xx = np.mgrid[0:h, 0:w]
        r = np.hypot(yy - h / 2, xx - w / 2) / (min(h, w) / 2)
        return {"st_lum_mean": float(g.mean()), "st_lum_std": float(g.std()), "st_sat_mean": float(sat.mean()),
                "st_lap_var": float(lap.var()), "st_hf_ratio": float(F[r > 0.5].sum() / (F.sum() + 1e-9))}
    except Exception:
        return {k: np.nan for k in IMAGE_STAT_NAMES}
