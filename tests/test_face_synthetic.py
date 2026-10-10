"""Synthetic sanity tests for physfeat.face: renders where the right answer is known.

Run:  python -m pytest tests        (or)        python tests/test_face_synthetic.py
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physfeat import face as F  # noqa: E402

S, R = 256, 110.0          # image size, sphere radius (px)
C = (S - 1) / 2


def _linear_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def _sphere():
    yy, xx = np.mgrid[0:S, 0:S].astype(float)
    x, y = (xx + 0.5 - C) / R, -(yy + 0.5 - C) / R           # camera frame: y up
    inside = x ** 2 + y ** 2 < 1
    n = np.stack([x, y, np.sqrt(np.clip(1 - x ** 2 - y ** 2, 0, None))], -1)
    n[~inside] = np.nan
    return n, inside


def _render(n, light_left, light_right=None):
    """Lambertian sphere, skin-coloured; optionally a different light on the right half."""
    l1 = np.asarray(light_left, float) / np.linalg.norm(light_left)
    shade = np.clip(np.nan_to_num(n) @ l1, 0, None)
    if light_right is not None:
        l2 = np.asarray(light_right, float) / np.linalg.norm(light_right)
        right = np.arange(S)[None, :] >= S / 2
        shade = np.where(right, np.clip(np.nan_to_num(n) @ l2, 0, None), shade)
    lin = 0.12 + 0.7 * shade
    rgb = _linear_to_srgb(lin[..., None] * np.array([1.0, 0.8, 0.65]))
    return (rgb * 255).round().astype(np.uint8)


LIGHT = [0.5, 0.4, 0.75]
MIRRORED = [-0.5, 0.4, 0.75]   # what mirroring the right half does to its light (cf. notebook 04)


def test_mesh_normals_match_sphere():
    """Landmarks on a sphere (MediaPipe convention: z smaller = closer) -> interpolated
    normals must match the analytic sphere normals; checks every coordinate-frame flip."""
    rng = np.random.default_rng(0)
    r, t = R * np.sqrt(rng.random(F.N_MESH)) * 0.98, 2 * np.pi * rng.random(F.N_MESH)
    u, v = C + r * np.cos(t), C + r * np.sin(t)
    z = -np.sqrt(R ** 2 - r ** 2)                              # front of the sphere = closer = smaller z
    L = np.c_[u + 0.5, v + 0.5, z]
    nmap = F.mesh_normal_map(L, (S, S))
    gt, inside = _sphere()
    inner = inside & (np.hypot(*np.mgrid[0:S, 0:S] - C) < 0.8 * R) & np.isfinite(nmap).all(-1)
    err = np.degrees(np.arccos(np.clip((nmap[inner] * gt[inner]).sum(-1), -1, 1)))
    assert np.median(err) < 3.0, np.median(err)


def test_lighting_consistent_vs_violation():
    n, inside = _sphere()
    skin = inside & (np.hypot(*np.mgrid[0:S, 0:S] - C) < 0.9 * R)
    mid = (0.0, S / 2)
    ok = F.skin_lighting(_render(n, LIGHT), n, skin, mid)
    bad = F.skin_lighting(_render(n, LIGHT, MIRRORED), n, skin, mid)
    print("consistent:", {k: round(v, 3) for k, v in ok.items()})
    print("violation: ", {k: round(v, 3) for k, v in bad.items()})
    assert ok["face_r2_sh1"] > 0.9
    assert ok["face_lr_light_deg"] < 10
    assert bad["face_lr_light_deg"] > ok["face_lr_light_deg"] + 20
    assert bad["face_lr_cross_r2"] < ok["face_lr_cross_r2"] - 0.1
    assert bad["face_r2_sh1"] < ok["face_r2_sh1"]


def _eye_image(hl_offsets, pupil="round"):
    """Two irises on skin; hl_offsets = per-eye highlight offset in iris radii (or None)."""
    img = np.full((200, 400, 3), (200, 160, 140), np.uint8)
    eyes = []
    for (cx, cy), off in zip([(100, 100), (300, 100)], hl_offsets):
        cv2.circle(img, (cx, cy), 20, (90, 60, 40), -1, cv2.LINE_AA)               # iris
        if pupil == "round":
            cv2.circle(img, (cx, cy), 7, (10, 10, 10), -1, cv2.LINE_AA)
        else:                                                                       # irregular pupil
            pts = [(cx + int((7 + 4 * (k % 2)) * np.cos(a)), cy + int((7 + 4 * (k % 2)) * np.sin(a)))
                   for k, a in enumerate(np.linspace(0, 2 * np.pi, 10, endpoint=False))]
            cv2.fillPoly(img, [np.array(pts, np.int32)], (10, 10, 10), cv2.LINE_AA)
        if off is not None:
            cv2.circle(img, (int(cx + off[0] * 20), int(cy + off[1] * 20)), 3, (255, 255, 255), -1, cv2.LINE_AA)
        eyes.append({"centre": np.array([cx, cy], float), "radius": 20.0, "poly": None})
    return img, eyes


def test_corneal_highlights():
    same = F.eye_features(*_eye_image([(0.4, -0.4), (0.4, -0.4)]))
    diff = F.eye_features(*_eye_image([(0.4, -0.4), (-0.4, 0.4)]))
    one = F.eye_features(*_eye_image([(0.4, -0.4), None]))
    print("same:", same, "\ndiff:", diff, "\none: ", one)
    assert same["face_eye_n_hl"] == 2 and same["face_eye_hl_iou"] > 0.6 and same["face_eye_hl_offset"] < 0.1
    assert diff["face_eye_hl_iou"] < 0.1 and diff["face_eye_hl_offset"] > 0.8
    assert one["face_eye_n_hl"] == 1 and one["face_eye_hl_iou"] == 0.0


def test_pupil_shape():
    rnd = F.eye_features(*_eye_image([(0.4, -0.4)] * 2, "round"))
    irr = F.eye_features(*_eye_image([(0.4, -0.4)] * 2, "irregular"))
    print("round:", rnd["face_pupil_irreg_mean"], "irregular:", irr["face_pupil_irreg_mean"])
    assert rnd["face_pupil_irreg_mean"] < 0.08
    assert irr["face_pupil_irreg_mean"] > rnd["face_pupil_irreg_mean"] + 0.05


def test_skin_gate_drops_hair_keeps_shadow():
    n, inside = _sphere()
    rgb = _render(n, [0.9, 0.1, 0.4])                 # strong side light: one side in deep shadow
    rgb[:60] = (70, 45, 25)                           # dark-brown "hair" band over the top
    skin = inside.copy()
    gated = F.skin_colour_gate(rgb, skin)
    hair = skin & (np.arange(S)[:, None] < 60)
    lit = rgb.astype(float).mean(-1)
    shadow = skin & ~hair & (lit <= np.percentile(lit[skin & ~hair], 25))
    assert shadow.sum() > 500
    assert gated[hair].mean() < 0.05                  # hair removed
    assert gated[shadow].mean() > 0.9                 # shadowed skin kept (chromaticity ignores shading)


def test_face_crop_is_never_stretched():
    """A face against the image border: the crop must stay square (margin shrinks or the
    border is replicated) instead of being clipped and stretched."""
    img = np.zeros((300, 400, 3), np.uint8)
    cv2.circle(img, (42, 150), 40, (255, 255, 255), -1)       # round "face" 2 px from the left edge
    t = np.linspace(0, 2 * np.pi, F.N_MESH + 10)
    L = np.c_[42 + 40 * np.cos(t), 150 + 40 * np.sin(t), np.zeros_like(t)]
    for margin in (0.0, 0.3, 1.0):
        crop = F.face_crop(img, L, size=128, margin=margin)
        assert crop.shape == (128, 128, 3)
        ys, xs = np.nonzero(crop[..., 0] > 128)
        w, h = np.ptp(xs) + 1, np.ptp(ys) + 1
        assert abs(w / h - 1) < 0.08, (margin, w, h)          # still a circle, not an ellipse


def test_small_iris_is_nan():
    img, eyes = _eye_image([(0.4, -0.4)] * 2)
    for e in eyes:
        e["radius"] = 2.0
    f = F.eye_features(img, eyes)
    assert np.isnan(f["face_eye_hl_iou"]) and np.isnan(f["face_eye_n_hl"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}\n")
