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