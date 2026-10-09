"""Controlled lighting violations on real photos (notebook 04).

Every edit replaces the RIGHT half of a square crop, so all edited versions share the
same kind of seam down the middle. They differ only in what that half's lighting implies:

  original : untouched crop
  shift    : CONTROL. Right half replaced by itself shifted vertically by 1/3 of the height
             (wraps around). Creates seams and mismatched content, but every pixel keeps its
             orientation, so the light still comes from the same direction everywhere.
  hflip    : VIOLATION. Right half mirrored left<->right. Its shading now implies a light whose
             horizontal component is reversed relative to the left half.
  rot180   : VIOLATION. Right half rotated 180 deg: light reversed horizontally AND vertically.

A feature that measures lighting consistency should move for hflip/rot180 and not for shift.
A feature that just detects editing seams would move for all three.
"""
from __future__ import annotations

import numpy as np

OPS = ["original", "shift", "hflip", "rot180"]
VIOLATIONS = ["hflip", "rot180"]
CONTROLS = ["shift"]

# expected direction of each lighting feature under a violation (+1 = goes up, -1 = goes down)
EXPECTED = {
    "lit_r2_sh1": -1,          # one global light explains less of the shading
    "lit_cell_ang_med": +1,    # local light directions disagree more with the global one
    "lit_cell_ang_p90": +1,
    "lit_cell_resultant": -1,  # local light directions agree less with each other
    "lit_loo_r2": -1,          # other regions predict a region's shading worse
    "lit_loo_gap": +1,
    "spec_align": -1,          # highlights sit less where the global light says they should
}


def apply_op(rgb: np.ndarray, op: str) -> np.ndarray:
    """rgb: HxWx3 (square crop). Returns a new array."""
    out = rgb.copy()
    h, w = rgb.shape[:2]
    m = w // 2
    right = rgb[:, m:]
    if op == "original":
        return out
    if op == "shift":
        out[:, m:] = np.roll(right, h // 3, axis=0)
    elif op == "hflip":
        out[:, m:] = right[:, ::-1]
    elif op == "rot180":
        out[:, m:] = right[::-1, ::-1]
    else:
        raise ValueError(op)
    return out


def apply_op_normals(n: np.ndarray, op: str) -> np.ndarray:
    """Same edit applied to a ground-truth normal map (x right, y up, z to viewer) — used only
    for the synthetic sanity test; on real photos the normal estimator sees the edited image."""
    out = n.copy()
    w = n.shape[1]
    m = w // 2
    right = n[:, m:].copy()
    if op == "original":
        return out
    if op == "shift":
        out[:, m:] = np.roll(right, n.shape[0] // 3, axis=0)
    elif op == "hflip":
        r = right[:, ::-1]
        r[..., 0] *= -1
        out[:, m:] = r
    elif op == "rot180":
        r = right[::-1, ::-1]
        r[..., 0] *= -1
        r[..., 1] *= -1
        out[:, m:] = r
    return out


def inconsistency_score(df, ref=None):
    """Training-free lighting-inconsistency score: mean of z-scored features, each signed so
    that 'more inconsistent' is larger. z-scores use `ref` (e.g. the original photos)."""
    ref = df if ref is None else ref
    cols = [c for c in EXPECTED if c in df]
    z = []
    for c in cols:
        mu, sd = np.nanmean(ref[c]), np.nanstd(ref[c]) + 1e-9
        z.append(EXPECTED[c] * (df[c].values - mu) / sd)
    z = np.vstack(z)
    return np.nanmean(z, axis=0)