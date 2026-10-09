"""
Lighting / shading physics features for AI-generated image detection.

Idea
----
Under a distant light, the (low-frequency) brightness of a Lambertian surface is
    I(x) ≈ a + l · n(x)            (first-order spherical harmonics)
where n(x) is the surface normal and l points toward the light.  In a real
photograph ONE lighting environment explains the shading of every surface, so a
lighting model fitted on one part of the image predicts the others, and local
light-direction estimates agree.  Generators paint shading that is locally
plausible but globally inconsistent.

Given an RGB image and a per-pixel normal map (from a monocular normal estimator,
camera frame: x right, y up, z toward the viewer), we measure:
  * how well one global lighting model explains the shading (R²),
  * whether locally estimated light directions agree (grid of cells),
  * whether a model fitted on the other cells predicts each cell (leave-one-out R²),
  * whether specular highlights sit where the half-vector says they should,
  * whether the normal field itself is physically plausible (integrability).

All fits run on a small (≤128 px) version of the maps; ~5–10 ms per image.
"""
from __future__ import annotations

import cv2
import numpy as np

WORK_SIDE = 128           # maps are resized to this (longer side) before fitting
GRID = 4                  # GRID x GRID cells for local lighting estimates
MIN_CELL_PIX = 120        # a cell needs this many valid pixels ...
MIN_NORMAL_SPREAD = 0.02  # ... and this much normal variation (2nd eigenvalue) to estimate light
LUM_LO, LUM_HI = 0.03, 0.97
SHADING_BLUR = 1.5        # px; suppresses fine albedo texture, keeps low-frequency shading
VIEW = np.array([0.0, 0.0, 1.0])

FEATURE_NAMES = [
    "lit_r2_sh1",          # R² of global first-order lighting model
    "lit_r2_sh2",          # R² of global second-order (9-coeff SH) model
    "lit_r2_gain",         # sh2 - sh1: how much "complex lighting" is needed
    "lit_strength",        # |l| / (|a| + |l|): directional vs ambient
    "lit_elev",            # z-component of global light direction
    "lit_n_cells",         # number of cells with a usable local estimate
    "lit_cell_ang_med",    # median angle (deg) between local and global light directions
    "lit_cell_ang_p90",    # 90th percentile of that angle
    "lit_cell_resultant",  # mean resultant length of local light directions (1 = all agree)
    "lit_loo_r2",          # leave-one-cell-out R²: global model fitted without a cell predicts it
    "lit_loo_gap",         # in-cell R² minus leave-one-out R² (overfit gap = local inconsistency)
    "nrm_curl",            # normalised curl of the gradient field implied by normals (0 = integrable)
    "nrm_smooth",          # median angular change between neighbouring normals (deg)
    "nrm_frontal",         # mean n_z (how much of the scene faces the camera)
    "spec_frac",           # fraction of pixels that look like specular highlights
    "spec_align",          # mean n·h over highlights minus mean n·h over all pixels
]


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _resize_pair(rgb, normals, side=WORK_SIDE):
    h, w = normals.shape[:2]
    s = side / max(h, w)
    size = (max(8, round(w * s)), max(8, round(h * s)))
    rgb_s = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    n_s = cv2.resize(normals.astype(np.float32), size, interpolation=cv2.INTER_AREA)
    n_s /= np.linalg.norm(n_s, axis=-1, keepdims=True) + 1e-8
    return rgb_s, n_s


def sh1_basis(n):
    return np.concatenate([np.ones((len(n), 1)), n], 1)                       # 4


def sh2_basis(n):
    x, y, z = n[:, 0], n[:, 1], n[:, 2]
    return np.stack([np.ones_like(x), x, y, z, x * y, x * z, y * z,
                     x * x - y * y, 3 * z * z - 1], 1)                        # 9


def _fit(A, b, iters=3, c=2.0):
    """Huber IRLS least squares. Returns coefficients and weights."""
    w = np.ones(len(b))
    coef = np.zeros(A.shape[1])
    for _ in range(iters):
        sw = np.sqrt(w)[:, None]
        coef, *_ = np.linalg.lstsq(A * sw, b * sw[:, 0], rcond=None)
        r = b - A @ coef
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
        u = np.abs(r) / (c * s)
        w = 1.0 / np.maximum(u, 1.0)
    return coef, w


def _r2(A, b, coef, w=None):
    r = b - A @ coef
    w = np.ones(len(b)) if w is None else w
    mu = np.sum(w * b) / w.sum()
    return float(1 - np.sum(w * r * r) / (np.sum(w * (b - mu) ** 2) + 1e-12))


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def _ang(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(_unit(a), _unit(b)), -1, 1))))


# ----------------------------------------------------------------------------
# main entry point
# ----------------------------------------------------------------------------
def extract(rgb: np.ndarray, normals: np.ndarray, return_debug: bool = False):
    """rgb: HxWx3 uint8; normals: hxwx3 float in [-1,1] (any resolution, same aspect)."""
    f = {k: np.nan for k in FEATURE_NAMES}
    rgb_s, n = _resize_pair(rgb, normals)
    h, w = n.shape[:2]
    lin = _srgb_to_linear(rgb_s.astype(np.float32) / 255.0)
    lum = 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]
    shade = cv2.GaussianBlur(lum, (0, 0), SHADING_BLUR)
    chroma = rgb_s.max(-1).astype(np.float32) / 255 - rgb_s.min(-1).astype(np.float32) / 255

    valid = (lum > LUM_LO) & (lum < LUM_HI) & np.isfinite(n).all(-1)
    # ---- normal-field plausibility -------------------------------------------------
    nz = np.clip(n[..., 2], 0.15, None)
    p, q = -n[..., 0] / nz, n[..., 1] / nz          # dz/dx, dz/dy (image y points down, normal y up)
    dp_dy = np.diff(p, axis=0)[:, :-1]
    dq_dx = np.diff(q, axis=1)[:-1, :]
    grad_mag = np.abs(np.diff(p, axis=1)[:-1, :]) + np.abs(np.diff(q, axis=0)[:, :-1])
    f["nrm_curl"] = float(np.median(np.abs(dp_dy - dq_dx)) / (np.median(grad_mag) + 1e-6))
    dots = np.concatenate([(n[1:] * n[:-1]).sum(-1).ravel(), (n[:, 1:] * n[:, :-1]).sum(-1).ravel()])
    f["nrm_smooth"] = float(np.degrees(np.median(np.arccos(np.clip(dots, -1, 1)))))
    f["nrm_frontal"] = float(n[..., 2].mean())

    if valid.sum() < 200:
        return (f, None) if return_debug else f

    N, B = n[valid], shade[valid]
    # ---- global lighting -------------------------------------------------------------
    A1, A2 = sh1_basis(N), sh2_basis(N)
    c1, w1 = _fit(A1, B)
    c2, w2 = _fit(A2, B)
    f["lit_r2_sh1"] = _r2(A1, B, c1, w1)
    f["lit_r2_sh2"] = _r2(A2, B, c2, w2)
    f["lit_r2_gain"] = f["lit_r2_sh2"] - f["lit_r2_sh1"]
    l_glob = c1[1:]
    f["lit_strength"] = float(np.linalg.norm(l_glob) / (abs(c1[0]) + np.linalg.norm(l_glob) + 1e-9))
    f["lit_elev"] = float(_unit(l_glob)[2])

    # ---- local (cell) lighting ------------------------------------------------------
    ys = np.linspace(0, h, GRID + 1).astype(int)
    xs = np.linspace(0, w, GRID + 1).astype(int)
    cell_id = np.full((h, w), -1, int)
    for i in range(GRID):
        for j in range(GRID):
            cell_id[ys[i]:ys[i + 1], xs[j]:xs[j + 1]] = i * GRID + j
    cid = cell_id[valid]
    # weighted normal equations per cell (global Huber weights) -> O(1) leave-one-out fits
    Aw = A1 * w1[:, None]
    G_all, r_all = A1.T @ Aw, Aw.T @ B
    dirs, wts, angs, loo, gaps = [], [], [], [], []
    for k in range(GRID * GRID):
        m = cid == k
        if m.sum() < MIN_CELL_PIX:
            continue
        ev = np.linalg.eigvalsh(np.cov(N[m].T))
        if ev[-2] < MIN_NORMAL_SPREAD:          # normals too uniform to pin down a light direction
            continue
        ck, wk = _fit(A1[m], B[m])
        r2_in = _r2(A1[m], B[m], ck, wk)
        if not np.isfinite(r2_in) or np.linalg.norm(ck[1:]) < 1e-4:
            continue
        # leave-one-cell-out: fit the lighting on every other pixel, predict this cell
        G_k, r_k = A1[m].T @ Aw[m], Aw[m].T @ B[m]
        co = np.linalg.solve(G_all - G_k + 1e-6 * np.eye(4), r_all - r_k)
        # allow a per-cell albedo gain/offset (albedo differs between regions, lighting does not)
        pred = A1[m] @ co
        G = np.c_[pred, np.ones(m.sum())]
        g, *_ = np.linalg.lstsq(G, B[m], rcond=None)
        r2_out = _r2(G, B[m], g)
        dirs.append(_unit(ck[1:]))
        wts.append(max(r2_in, 0) * m.sum())
        angs.append(_ang(ck[1:], l_glob))
        loo.append(r2_out)
        gaps.append(r2_in - r2_out)
    f["lit_n_cells"] = float(len(dirs))
    if len(dirs) >= 2:
        wts = np.asarray(wts) + 1e-9
        f["lit_cell_ang_med"] = float(np.median(angs))
        f["lit_cell_ang_p90"] = float(np.percentile(angs, 90))
        f["lit_cell_resultant"] = float(np.linalg.norm((np.asarray(dirs) * wts[:, None]).sum(0)) / wts.sum())
        f["lit_loo_r2"] = float(np.average(loo, weights=wts))
        f["lit_loo_gap"] = float(np.average(gaps, weights=wts))

    # ---- specular highlights -------------------------------------------------------
    spec = valid & (lum > np.percentile(lum[valid], 99)) & (chroma < 0.12) & (lum > 0.5)
    f["spec_frac"] = float(spec.mean())
    if spec.sum() >= 5:
        hvec = _unit(_unit(l_glob) + VIEW)
        ndh = n @ hvec
        f["spec_align"] = float(ndh[spec].mean() - ndh[valid].mean())

    if return_debug:
        pred = np.full((h, w), np.nan, np.float32)
        pred[valid] = A1 @ c1
        dbg = {"rgb": rgb_s, "normals": n, "shade": shade, "pred": pred, "valid": valid,
               "light": _unit(l_glob), "cell_id": cell_id}
        return f, dbg
    return f


def draw_evidence(dbg) -> np.ndarray:
    """3-panel strip: normals | observed shading | shading predicted by ONE global light
    (with the estimated light direction drawn as an arrow)."""
    n_vis = ((dbg["normals"] * [1, 1, 1] + 1) * 127.5).clip(0, 255).astype(np.uint8)
    def gray(x):
        v = np.nan_to_num(x, nan=0.0)
        lo, hi = np.percentile(v[dbg["valid"]], [1, 99]) if dbg["valid"].any() else (0, 1)
        return (np.clip((v - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)
    obs = cv2.cvtColor(gray(dbg["shade"]), cv2.COLOR_GRAY2RGB)
    pred = cv2.cvtColor(gray(dbg["pred"]), cv2.COLOR_GRAY2RGB)
    h, w = pred.shape[:2]
    l = dbg["light"]
    c = (w // 2, h // 2)
    tip = (int(c[0] + l[0] * 0.4 * w), int(c[1] - l[1] * 0.4 * h))   # y up -> image y down
    cv2.arrowedLine(pred, c, tip, (255, 180, 0), 2, tipLength=0.25)
    return np.concatenate([n_vis, obs, pred], 1)