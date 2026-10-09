"""
Face physics features for deepfake / AI-generated face detection (PLAN.md, phase 2).

Why faces
---------
Notebooks 01-04 showed that lighting features computed from *monocular* normals are too
imprecise to help a CLIP detector on generic scenes. On faces the geometry is known far
better: MediaPipe Face Landmarker regresses a 478-point 3D mesh (468 face points + 10 iris
points), so surface normals come from a face model instead of a per-pixel guess.

Features (NaN when undefined: no face, eyes too small or closed, ...)
--------
Lighting on skin (Lambertian shading, same SH model as lighting.py)
  * how well ONE light explains the skin shading (R²), its strength and elevation;
  * whether the left and right halves of the face agree on the light direction, and
    whether a light fitted on one half predicts the other (a pasted inner face or a
    generator's locally painted shading breaks this).
Eyes (Hu, Li & Lyu, ICASSP 2021; Guo et al., ICASSP 2022)
  * both corneas reflect the same scene, so in iris-normalised coordinates their specular
    highlights should overlap (IoU) and sit at the same offset from the iris centre;
  * real pupils are close to elliptical; GAN pupils are often irregular.
Pose (Yang, Li & Lyu, ICASSP 2019)
  * the inner face (eyes, nose, mouth) and the face outline should point the same way;
    face swaps paste one person's inner face onto another head.

Coordinates: landmarks are (N,3) in pixels of the input image: x right, y down, z = depth
(smaller = closer to the camera, same scale as x). Normals use the camera frame of
lighting.py: x right, y up, z toward the viewer.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from .lighting import LUM_HI, LUM_LO, SHADING_BLUR, VIEW, _ang, _fit, _r2, _srgb_to_linear, _unit, sh1_basis, sh2_basis

# ----------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------
WORK_SIDE = 256          # face crop (longer side, px) for the skin-lighting fit
CROP_MARGIN = 0.05       # margin around the landmark bounding box (fraction of face size)
SKIN_ERODE_FRAC = 0.03   # erode the face outline by this fraction of the crop (hair, silhouette)
FEATURE_DILATE_FRAC = 0.02  # grow the eye / brow / lip masks by this fraction of the crop
MIN_SKIN_PIX = 400       # skin pixels needed for the global fit (a quarter of this per half)
SKIN_CHROMA_TOL = 0.03   # keep pixels whose rg-chromaticity is this close to the median skin colour
SKIN_MIN_REL_LUM = 0.35  # ... and that are not darker than this fraction of the median skin brightness
EYE_GRID = 48            # each iris disc is resampled to EYE_GRID x EYE_GRID
MIN_IRIS_PX = 5.0        # iris radius (px, input image) below which eye features are NaN
HL_MAX_R = 0.8           # look for highlights only inside this radius (iris radii): avoids the sclera
MIN_OPEN_FRAC = 0.3      # share of the iris disc that must be visible (not under the lids)
HL_MIN_LUM = 0.6         # a corneal highlight is brighter than this (sRGB luminance) ...
HL_REL = 0.25            # ... and brighter than the median iris luminance by this much
PUPIL_MIN_CONTRAST = 0.08  # pupil vs iris luminance gap needed to segment the pupil at all
PUPIL_MAX_SHARE = 0.6      # the dark class must be at most this share of the visible iris
MIN_PUPIL_PIX = 12

FACE_LANDMARKER_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                       "face_landmarker/float16/1/face_landmarker.task")

# MediaPipe face-mesh landmark indices ("left"/"right" are the subject's own sides)
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
             152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LEFT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_BROW = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
RIGHT_BROW = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
MIDLINE = [10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 164, 0, 17, 18, 200, 199, 175, 152]
IRISES = [(468, [469, 470, 471, 472]), (473, [474, 475, 476, 477])]   # (centre, ring)
INNER_FACE = [33, 133, 362, 263, 168, 6, 4, 1, 61, 291, 13, 14]
N_MESH = 468

LIGHT_NAMES = [
    "face_r2_sh1",          # R² of one first-order light on the skin
    "face_r2_sh2",          # R² of the second-order (9-coefficient) model
    "face_light_strength",  # |l| / (|a| + |l|): directional vs ambient
    "face_light_elev",      # z-component of the light direction
    "face_lr_light_deg",    # angle between lights fitted on the left and right halves
    "face_lr_cross_r2",     # R² when one half's light predicts the other half (mean of both ways)
    "face_lr_gap",          # in-half R² minus cross R² (local-vs-global inconsistency)
    "face_spec_align",      # skin highlights: mean n·h over highlights minus mean over skin
]
EYE_NAMES = [
    "face_iris_px",         # iris radius in input pixels (tells the model how reliable eye cues are)
    "face_eye_n_hl",        # usable eyes that show a corneal highlight (0, 1, 2)
    "face_eye_hl_iou",      # IoU of the two highlights in iris-normalised coordinates
    "face_eye_hl_offset",   # distance between the two highlight centroids (iris radii)
    "face_eye_hl_area_ratio",  # smaller / larger highlight area
    "face_pupil_irreg_mean",   # 1 - IoU(pupil, fitted ellipse), mean over usable eyes
    "face_pupil_irreg_max",
]
POSE_NAMES = [
    "face_pose_inner_outer_deg",  # angle between inner-face and outline plane normals
    "face_yaw_deg", "face_pitch_deg",
]
FEATURE_NAMES = ["face_found", "face_px"] + LIGHT_NAMES + EYE_NAMES + POSE_NAMES


# ----------------------------------------------------------------------------
# landmark detection (MediaPipe Face Landmarker)
# ----------------------------------------------------------------------------
def ensure_model(path=None) -> Path:
    """Path to face_landmarker.task; downloads it (~4 MB) to ~/.cache/physfeat on first use."""
    p = Path(path or os.environ.get("PHYSFAKE_FACE_MODEL", "")
             or Path.home() / ".cache" / "physfeat" / "face_landmarker.task")
    if not p.exists():
        import urllib.request
        p.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(FACE_LANDMARKER_URL, p)
    return p


class FaceMesher:
    """rgb (HxWx3 uint8) -> (478,3) landmarks in input pixels, or None if no face is found.
    With several faces, the largest one is returned."""

    def __init__(self, model_path=None, max_faces=4):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self._mp = mp
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=ensure_model(model_path).read_bytes()),
            running_mode=vision.RunningMode.IMAGE, num_faces=max_faces)
        self.landmarker = vision.FaceLandmarker.create_from_options(opts)

    def __call__(self, rgb: np.ndarray):
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb[..., :3]))
        faces = self.landmarker.detect(img).face_landmarks
        if not faces:
            return None
        h, w = rgb.shape[:2]
        cands = [np.array([[p.x * w, p.y * h, p.z * w] for p in f], float) for f in faces]
        return max(cands, key=lambda L: np.ptp(L[:, 0]) * np.ptp(L[:, 1]))


_MESHER = None


def default_mesher() -> FaceMesher:
    global _MESHER
    if _MESHER is None:  # one landmarker per process
        _MESHER = FaceMesher()
    return _MESHER


# ----------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------
def crop_box(L, shape, margin=CROP_MARGIN):
    h, w = shape[:2]
    (x0, y0), (x1, y1) = L[:, :2].min(0), L[:, :2].max(0)
    pad = margin * max(x1 - x0, y1 - y0)
    return (int(max(0, np.floor(x0 - pad))), int(max(0, np.floor(y0 - pad))),
            int(min(w, np.ceil(x1 + pad))), int(min(h, np.ceil(y1 + pad))))


def face_crop(rgb, L, size=224, margin=0.3) -> np.ndarray:
    """Square crop centred on the face (for the CLIP branch), resized to size x size."""
    (x0, y0), (x1, y1) = L[:, :2].min(0), L[:, :2].max(0)
    c, half = (np.array([x0 + x1, y0 + y1]) / 2), (1 + margin) * max(x1 - x0, y1 - y0) / 2
    h, w = rgb.shape[:2]
    a = np.clip(np.round(c - half), 0, [w, h]).astype(int)
    b = np.clip(np.round(c + half), 0, [w, h]).astype(int)
    crop = rgb[a[1]:b[1], a[0]:b[0]]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA if crop.shape[0] > size else cv2.INTER_CUBIC)


def to_camera(P):
    """Image frame (x right, y down, z away) -> camera frame (x right, y up, z toward viewer)."""
    return np.asarray(P, float) * [1.0, -1.0, -1.0]


def mesh_normal_map(L, shape):
    """Per-pixel unit normals (HxWx3, camera frame) interpolated from the face mesh;
    NaN outside the mesh. The triangulation is a 2-D Delaunay of the projected landmarks,
    which is independent of MediaPipe's version-specific topology tables."""
    from scipy.interpolate import LinearNDInterpolator
    from scipy.spatial import Delaunay
    pts = np.asarray(L, float)[:N_MESH]
    tri = Delaunay(pts[:, :2])
    P = to_camera(pts)
    s = tri.simplices
    fn = np.cross(P[s[:, 1]] - P[s[:, 0]], P[s[:, 2]] - P[s[:, 0]])   # area-weighted face normals
    fn *= np.where(fn[:, 2] < 0, -1.0, 1.0)[:, None]                  # orient toward the viewer
    vn = np.zeros_like(P)
    for k in range(3):
        np.add.at(vn, s[:, k], fn)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12
    h, w = shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    n = LinearNDInterpolator(tri, vn)(xx + 0.5, yy + 0.5).astype(np.float32)
    return n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-8)


def _hull_mask(pts, shape):
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(np.round(pts[:, :2]).astype(np.int32)), 1)
    return m


def skin_mask(L, shape):
    """Face outline minus eyes, brows and lips, eroded away from the silhouette."""
    side = max(shape[:2])
    er = max(1, int(round(SKIN_ERODE_FRAC * side)))
    dl = max(1, int(round(FEATURE_DILATE_FRAC * side)))
    skin = cv2.erode(_hull_mask(L[FACE_OVAL], shape), np.ones((2 * er + 1,) * 2, np.uint8))
    feats = np.zeros_like(skin)
    for idx in (LEFT_EYE, RIGHT_EYE, LEFT_BROW, RIGHT_BROW, LIPS):
        feats |= _hull_mask(L[idx], shape)
    feats = cv2.dilate(feats, np.ones((2 * dl + 1,) * 2, np.uint8))
    return (skin > 0) & (feats == 0)


def skin_colour_gate(rgb, skin):
    """Drop pixels inside the face outline that are not skin-coloured (hair over the forehead,
    glasses, beards). Uses rg-chromaticity, which shading does not change for a Lambertian
    surface, so shadowed skin is kept; a brightness floor removes near-black pixels."""
    if skin.sum() < MIN_SKIN_PIX:
        return skin
    c = rgb.astype(np.float32) + 1.0
    chrom = c[..., :2] / c.sum(-1, keepdims=True)
    dist = np.linalg.norm(chrom - np.median(chrom[skin], 0), axis=-1)
    lum = c.mean(-1)
    return skin & (dist < SKIN_CHROMA_TOL) & (lum > SKIN_MIN_REL_LUM * np.median(lum[skin]))


def midline(L):
    """Facial midline as x = a*y + b (pixels): robust to head roll."""
    a, b = np.polyfit(L[MIDLINE, 1], L[MIDLINE, 0], 1)
    return float(a), float(b)


def _plane_normal(P):
    v = np.linalg.svd(P - P.mean(0), full_matrices=False)[2][-1]
    return v if v[2] >= 0 else -v


# ----------------------------------------------------------------------------
# feature groups (array-level, so they can be tested on synthetic data)
# ----------------------------------------------------------------------------
def skin_lighting(rgb, normals, skin, mid, return_debug=False):
    """rgb HxWx3 uint8, normals HxWx3 (camera frame, NaN where unknown), skin HxW bool,
    mid = (a, b): the line x = a*y + b separating the two halves of the face."""
    f = {k: np.nan for k in LIGHT_NAMES}
    lin = _srgb_to_linear(rgb.astype(np.float32) / 255.0)
    lum = 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]
    shade = cv2.GaussianBlur(lum, (0, 0), SHADING_BLUR)
    chroma = rgb.max(-1).astype(np.float32) / 255 - rgb.min(-1).astype(np.float32) / 255
    valid = skin & np.isfinite(normals).all(-1) & (lum > LUM_LO) & (lum < LUM_HI)
    dbg = None
    if valid.sum() < MIN_SKIN_PIX:
        return (f, dbg) if return_debug else f

    N, B = normals[valid], shade[valid]
    A1, A2 = sh1_basis(N), sh2_basis(N)
    c1, w1 = _fit(A1, B)
    c2, w2 = _fit(A2, B)
    l_glob = c1[1:]
    f["face_r2_sh1"] = _r2(A1, B, c1, w1)
    f["face_r2_sh2"] = _r2(A2, B, c2, w2)
    f["face_light_strength"] = float(np.linalg.norm(l_glob) / (abs(c1[0]) + np.linalg.norm(l_glob) + 1e-9))
    f["face_light_elev"] = float(_unit(l_glob)[2])

    # ---- left half vs right half --------------------------------------------------
    yy, xx = np.nonzero(valid)
    left = xx < mid[0] * yy + mid[1]
    halves = {}
    for name, m in (("left", left), ("right", ~left)):
        if m.sum() >= MIN_SKIN_PIX // 4:
            ck, wk = _fit(A1[m], B[m])
            if np.linalg.norm(ck[1:]) > 1e-4:
                halves[name] = (m, ck, _r2(A1[m], B[m], ck, wk))
    if len(halves) == 2:
        (mL, cL, rL), (mR, cR, rR) = halves["left"], halves["right"]
        f["face_lr_light_deg"] = _ang(cL[1:], cR[1:])
        cross = []
        for c_src, m_dst in ((cL, mR), (cR, mL)):
            # the other half's light, with a free gain/offset (albedo may differ, lighting may not)
            G = np.c_[A1[m_dst] @ c_src, np.ones(m_dst.sum())]
            g, *_ = np.linalg.lstsq(G, B[m_dst], rcond=None)
            cross.append(_r2(G, B[m_dst], g))
        f["face_lr_cross_r2"] = float(np.mean(cross))
        f["face_lr_gap"] = float(np.mean([rL, rR]) - np.mean(cross))

    # ---- specular highlights on skin ------------------------------------------------
    spec = valid & (lum > np.percentile(lum[valid], 99)) & (chroma < 0.15) & (lum > 0.4)
    if spec.sum() >= 5:
        ndh = normals @ _unit(_unit(l_glob) + VIEW)
        f["face_spec_align"] = float(np.nanmean(ndh[spec]) - np.nanmean(ndh[valid]))

    if return_debug:
        pred = np.full(lum.shape, np.nan, np.float32)
        pred[valid] = A1 @ c1
        dbg = {"shade": shade, "pred": pred, "valid": valid, "light": _unit(l_glob), "mid": mid,
               "light_halves": {k: _unit(v[1][1:]) for k, v in halves.items()}}
        return f, dbg
    return f


def _iris_patch(rgb, centre, radius, eye_poly=None, grid=EYE_GRID):
    """Resample the iris disc onto a grid in iris-normalised coordinates (radius = 1)."""
    u = np.linspace(-1, 1, grid, dtype=np.float32)
    U, V = np.meshgrid(u, u)
    mx = (centre[0] + radius * U).astype(np.float32)   # cv2.remap needs float32 maps
    my = (centre[1] + radius * V).astype(np.float32)
    patch = cv2.remap(rgb, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32) / 255
    disc = U ** 2 + V ** 2 <= 1.0
    visible = disc.copy()
    if eye_poly is not None:   # keep the part of the iris that is not under the eyelids
        hull = cv2.convexHull(np.asarray(eye_poly, np.float32)[:, :2])
        inside = np.array([cv2.pointPolygonTest(hull, (float(x), float(y)), False) >= 0
                           for x, y in zip(mx.ravel(), my.ravel())]).reshape(mx.shape)
        visible &= inside
    lum = 0.2126 * patch[..., 0] + 0.7152 * patch[..., 1] + 0.0722 * patch[..., 2]   # sRGB luminance
    return lum, disc, visible, (U, V)


def _pupil_irregularity(lum, visible, hl, UV):
    """1 - IoU between the pupil (dark blob, Otsu threshold inside the visible iris) and its
    best-fit ellipse. NaN if the pupil cannot be isolated: too little contrast with the iris
    (common for dark-brown eyes in RGB), too small, or cut by the eyelids."""
    iris = visible & ~hl
    vals = lum[iris]
    if len(vals) < 4 * MIN_PUPIL_PIX:
        return np.nan, None
    thr, _ = cv2.threshold(np.round(vals * 255).astype(np.uint8).reshape(-1, 1), 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr /= 255.0
    lo, hi = vals[vals <= thr], vals[vals > thr]
    if len(lo) == 0 or len(hi) == 0 or hi.mean() - lo.mean() < PUPIL_MIN_CONTRAST \
            or len(lo) > PUPIL_MAX_SHARE * len(vals):
        return np.nan, None
    dark = (iris & (lum <= thr)).astype(np.uint8)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(dark, connectivity=8)
    if n < 2:
        return np.nan, None
    g = lum.shape[0]
    k = 1 + int(np.argmin(np.linalg.norm(cents[1:] - [g / 2, g / 2], axis=1)))   # blob nearest the iris centre
    cnts = cv2.findContours((lab == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    cnt = max(cnts, key=cv2.contourArea)
    if len(cnt) < 5:
        return np.nan, None
    comp = np.zeros((g, g), np.uint8)
    cv2.drawContours(comp, [cnt], -1, 1, -1)   # filled: a highlight on the pupil leaves no hole
    if comp.sum() < MIN_PUPIL_PIX:
        return np.nan, None
    occluded = (UV[0] ** 2 + UV[1] ** 2 <= 1.0) & ~visible
    if (cv2.dilate(comp, np.ones((3, 3), np.uint8)).astype(bool) & occluded).any():
        return np.nan, None    # truncated by a lid: its outline says nothing about the pupil
    ell = cv2.fitEllipse(cnt)
    em = np.zeros_like(comp)
    cv2.ellipse(em, ell, 1, -1)
    inter, union = (comp & em).sum(), (comp | em).sum()
    return float(1 - inter / max(union, 1)), ell


def eye_features(rgb, eyes, return_debug=False):
    """eyes: two dicts {"centre": (x, y), "radius": r, "poly": (k,2) eye-opening points or None}."""
    f = {k: np.nan for k in EYE_NAMES}
    f["face_iris_px"] = float(np.mean([e["radius"] for e in eyes]))
    usable, dbg = [], []
    for e in eyes:
        if e["radius"] < MIN_IRIS_PX:
            continue
        lum, disc, vis, UV = _iris_patch(rgb, e["centre"], e["radius"], e.get("poly"))
        if vis.sum() < MIN_OPEN_FRAC * disc.sum():
            continue
        inner = UV[0] ** 2 + UV[1] ** 2 <= HL_MAX_R ** 2
        hl = vis & inner & (lum >= max(HL_MIN_LUM, float(np.median(lum[vis])) + HL_REL))
        irr, ell = _pupil_irregularity(lum, vis, hl, UV)
        usable.append({"hl": hl, "irr": irr, "UV": UV})
        dbg.append({"lum": lum, "visible": vis, "hl": hl, "ellipse": ell})
    if usable:
        f["face_eye_n_hl"] = float(sum(u["hl"].any() for u in usable))
        irr = [u["irr"] for u in usable if np.isfinite(u["irr"])]
        if irr:
            f["face_pupil_irreg_mean"], f["face_pupil_irreg_max"] = float(np.mean(irr)), float(np.max(irr))
    if len(usable) == 2:
        a, b = usable[0]["hl"], usable[1]["hl"]
        if a.any() and b.any():
            f["face_eye_hl_iou"] = float((a & b).sum() / (a | b).sum())
            U, V = usable[0]["UV"]
            ca, cb = np.array([U[a].mean(), V[a].mean()]), np.array([U[b].mean(), V[b].mean()])
            f["face_eye_hl_offset"] = float(np.linalg.norm(ca - cb))
            f["face_eye_hl_area_ratio"] = float(min(a.sum(), b.sum()) / max(a.sum(), b.sum()))
        elif a.any() or b.any():   # one eye reflects the light, the other does not
            f["face_eye_hl_iou"], f["face_eye_hl_area_ratio"] = 0.0, 0.0
    return (f, dbg) if return_debug else f


def pose_features(L):
    f = {k: np.nan for k in POSE_NAMES}
    P = to_camera(L)
    n_in, n_out = _plane_normal(P[INNER_FACE]), _plane_normal(P[FACE_OVAL])
    f["face_pose_inner_outer_deg"] = _ang(n_in, n_out)
    f["face_yaw_deg"] = float(np.degrees(np.arctan2(n_out[0], n_out[2])))
    f["face_pitch_deg"] = float(np.degrees(np.arctan2(n_out[1], n_out[2])))
    return f


def _eyes_from_landmarks(L):
    eyes = []
    polys = [L[LEFT_EYE, :2], L[RIGHT_EYE, :2]]
    for c_idx, ring in IRISES:
        c = L[c_idx, :2]
        r = float(np.mean(np.linalg.norm(L[ring, :2] - c, axis=1)))
        poly = min(polys, key=lambda p: np.linalg.norm(p.mean(0) - c))   # the eye this iris sits in
        eyes.append({"centre": c, "radius": r, "poly": poly})
    return eyes


# ----------------------------------------------------------------------------
# main entry point
# ----------------------------------------------------------------------------
def extract(rgb: np.ndarray, landmarks=None, mesher=None, return_debug=False):
    """rgb: HxWx3 uint8. landmarks: (478,3) in input pixels, or None to run MediaPipe."""
    f = {k: np.nan for k in FEATURE_NAMES}
    f["face_found"] = 0.0
    if landmarks is None:
        landmarks = (mesher or default_mesher())(rgb)
    if landmarks is None:
        return (f, None) if return_debug else f
    L = np.asarray(landmarks, float)
    f["face_found"] = 1.0
    f.update(pose_features(L))

    eye_f, eye_dbg = eye_features(rgb, _eyes_from_landmarks(L), return_debug=True)   # full resolution
    f.update(eye_f)

    x0, y0, x1, y1 = crop_box(L, rgb.shape)
    crop = rgb[y0:y1, x0:x1]
    f["face_px"] = float(max(crop.shape[:2]))
    s = WORK_SIDE / max(crop.shape[:2])
    crop = cv2.resize(crop, (max(8, round(crop.shape[1] * s)), max(8, round(crop.shape[0] * s))),
                      interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    Lc = (L - [x0, y0, 0.0]) * s
    normals = mesh_normal_map(Lc, crop.shape)
    skin = skin_colour_gate(crop, skin_mask(Lc, crop.shape))
    light_f, light_dbg = skin_lighting(crop, normals, skin, midline(Lc), return_debug=True)
    f.update(light_f)

    if return_debug:
        return f, {"crop": crop, "landmarks": Lc, "normals": normals, "skin": skin,
                   "lighting": light_dbg, "eyes": eye_dbg}
    return f


def extract_features(rgb, landmarks=None, mesher=None) -> dict:
    """Never raises (for batch jobs)."""
    try:
        return extract(rgb, landmarks, mesher)
    except Exception as e:
        out = {k: np.nan for k in FEATURE_NAMES}
        out["face_error"] = repr(e)[:200]
        return out


# ----------------------------------------------------------------------------
# visualisation (notebooks and the demo)
# ----------------------------------------------------------------------------
def _gray(x, valid):
    v = np.nan_to_num(x, nan=0.0)
    lo, hi = np.percentile(v[valid], [1, 99]) if valid.any() else (0.0, 1.0)
    g = (np.clip((v - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)
    g[~valid] = 0
    return cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)


def _arrow(img, l, colour, origin=None, scale=0.3):
    h, w = img.shape[:2]
    c = origin or (w // 2, h // 2)
    tip = (int(c[0] + l[0] * scale * w), int(c[1] - l[1] * scale * h))   # y up -> image y down
    cv2.arrowedLine(img, c, tip, colour, 2, cv2.LINE_AA, tipLength=0.25)


def draw_evidence(dbg) -> np.ndarray:
    """Strip: face + skin mask + midline | observed skin shading | shading predicted by ONE
    light (yellow = global light, cyan/magenta = lights of the two halves) | both irises
    (red = corneal highlight, green = fitted pupil ellipse)."""
    crop = dbg["crop"].copy()
    h, w = crop.shape[:2]
    over = crop.copy()
    over[dbg["skin"]] = (0.6 * over[dbg["skin"]] + 0.4 * np.array([80, 160, 255])).astype(np.uint8)
    lt = dbg["lighting"]
    panels = [over]
    if lt is not None:
        a, b = lt["mid"]
        cv2.line(over, (int(b), 0), (int(a * h + b), h), (255, 255, 255), 1, cv2.LINE_AA)
        obs, pred = _gray(lt["shade"], lt["valid"]), _gray(lt["pred"], lt["valid"])
        _arrow(pred, lt["light"], (255, 200, 0))
        for name, col, x in (("left", (0, 220, 255), w // 4), ("right", (255, 0, 220), 3 * w // 4)):
            if name in lt["light_halves"]:
                _arrow(pred, lt["light_halves"][name], col, origin=(x, h // 2), scale=0.2)
        panels += [obs, pred]
    s = h // 2
    col = np.zeros((h, s, 3), np.uint8)   # up to two irises stacked, same height as the other panels
    for i, e in enumerate(dbg["eyes"][:2]):
        g = cv2.cvtColor((np.clip(e["lum"], 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        g[~e["visible"]] //= 3
        g[e["hl"]] = (255, 40, 40)
        if e["ellipse"] is not None:
            cv2.ellipse(g, e["ellipse"], (40, 220, 40), 1, cv2.LINE_AA)
        col[i * s:(i + 1) * s] = cv2.resize(g, (s, s), interpolation=cv2.INTER_NEAREST)
    panels.append(col)
    return np.concatenate(panels, 1)
