"""
Projective-geometry (perspective) physics features for AI-generated image detection.

Idea
----
In a real photograph taken with a (roughly) pinhole camera, images of parallel 3-D
lines converge *exactly* to a common vanishing point (VP), and the VPs of mutually
orthogonal directions obey camera constraints (principal point near the image
centre, positive focal length). Generators learn perspective only statistically,
so their lines tend to *almost* converge: scattered intersections, one VP split
into several nearby ones, and VP triplets that imply an impossible camera.

Pipeline
--------
1. Resize (max side = 512 by default) -> grayscale -> LSD line segments.
2. Sequential, length-weighted RANSAC to find up to 3 dominant VPs
   (VPs may be finite or at infinity; everything is done in homogeneous coords).
3. Least-squares refinement of each VP, then measurement of how well its
   inlier segments actually converge (angular residuals).
4. Camera-consistency checks between VPs (implied focal length, principal point).

All coordinates are normalised: origin at the image centre, unit = half diagonal.
Features are NaN when they are undefined (e.g. no lines in a photo of a cat);
downstream models must handle NaN (HistGradientBoosting does natively).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------
MAX_SIDE = 512            # images are resized so the longer side is this many px
MIN_LEN_FRAC = 0.025      # keep segments longer than this fraction of the diagonal
LONG_LEN_FRAC = 0.08      # "long" segments (used for the unexplained-lines feature)
INLIER_DEG = 1.5          # a segment supports a VP if its angular error is below this
TIGHT_DEG = 0.5           # threshold for the "tight convergence" fraction
N_HYP = 400               # RANSAC hypotheses per VP
MAX_VPS = 3
MIN_VP_SUPPORT = 4        # minimum number of inlier segments for a VP
MIN_VP_LEN_FRAC = 0.04    # minimum share of total line length for a VP
MAX_SEGMENTS = 1500       # cap (longest kept) to bound runtime on very busy images
FINITE_LIMIT = 8.0        # |VP| beyond this many half-diagonals is treated as "at infinity"

FEATURE_NAMES = [
    "geo_log_nseg",          # log(1 + number of segments kept)
    "geo_line_density",      # total segment length / image diagonal
    "geo_n_vps",             # number of VPs found (0..3)
    "geo_frac_vp1", "geo_frac_vp2", "geo_frac_vp3",   # share of line length supporting each VP
    "geo_frac_long_unexpl",  # share of long-segment length explained by no VP
    "geo_res_vp1", "geo_res_vp2", "geo_res_vp3",      # length-weighted mean angular residual (deg)
    "geo_tight_vp1", "geo_tight_vp2", "geo_tight_vp3",  # share of inlier length with residual < TIGHT_DEG
    "geo_cv_res_vp1",        # split-half cross-validated residual of VP1 (deg)
    "geo_min_vp_sep_deg",    # smallest angle between VP directions (split VPs are suspicious)
    "geo_ortho_cos_min",     # best |cos| between VP directions under the assumed camera
    "geo_f2_12",             # signed log focal^2 implied by VP1-VP2 orthogonality
    "geo_pp_offset",         # principal point (orthocentre of 3 finite VPs) distance from centre
]


@dataclass
class GeometryResult:
    features: dict
    segments: np.ndarray = field(default=None, repr=False)   # (N,4) pixel coords in resized image
    labels: np.ndarray = field(default=None, repr=False)     # (N,) VP index or -1
    residuals: np.ndarray = field(default=None, repr=False)  # (N,) angular error to own VP (deg)
    vps_px: list = field(default_factory=list)                # VPs in resized-pixel homogeneous coords
    image_rgb: np.ndarray = field(default=None, repr=False)  # resized RGB image (for plotting)


# ----------------------------------------------------------------------------
# image loading
# ----------------------------------------------------------------------------
def load_image(path_or_array, max_side: int = MAX_SIDE) -> np.ndarray:
    """Return an RGB uint8 image whose longer side is `max_side`."""
    if isinstance(path_or_array, np.ndarray):
        rgb = path_or_array
    else:
        bgr = cv2.imread(str(path_or_array), cv2.IMREAD_COLOR)
        if bgr is None:  # some formats (e.g. certain webp/gif) -> fall back to PIL
            from PIL import Image
            rgb = np.array(Image.open(path_or_array).convert("RGB"))
        else:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, -1)
    h, w = rgb.shape[:2]
    s = max_side / max(h, w)
    if abs(s - 1) > 1e-3:
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
        rgb = cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=interp)
    return rgb


# ----------------------------------------------------------------------------
# core geometry helpers
# ----------------------------------------------------------------------------
_LSD = None


def _lsd():
    global _LSD
    if _LSD is None:  # one detector per process
        _LSD = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    return _LSD


def detect_segments(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    diag = np.hypot(h, w)
    lines = _lsd().detect(gray)[0]
    if lines is None:
        return np.zeros((0, 4), np.float64)
    segs = lines.reshape(-1, 4).astype(np.float64)
    L = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    keep = L >= MIN_LEN_FRAC * diag
    segs, L = segs[keep], L[keep]
    if len(segs) > MAX_SEGMENTS:
        segs = segs[np.argsort(-L)[:MAX_SEGMENTS]]
    return segs


def _normalise(segs, h, w):
    c = np.array([w / 2.0, h / 2.0])
    s = np.hypot(h, w) / 2.0
    p1 = (segs[:, :2] - c) / s
    p2 = (segs[:, 2:] - c) / s
    return p1, p2, c, s


def _angles_to_vps(mid, d, vps):
    """Angular error (deg, 0..90) between each segment and the line joining its
    midpoint to each VP.  vps: (K,3) homogeneous.  Returns (K,N)."""
    # direction from midpoint towards VP; valid for finite and infinite VPs
    dv = vps[:, None, :2] - mid[None] * vps[:, None, 2:3]          # (K,N,2)
    num = np.abs(dv[..., 0] * d[None, :, 0] + dv[..., 1] * d[None, :, 1])
    den = np.linalg.norm(dv, axis=-1) * np.linalg.norm(d, axis=-1)[None] + 1e-12
    return np.degrees(np.arccos(np.clip(num / den, 0.0, 1.0)))


def _refine_vp(lines_n, weights):
    """Weighted least squares VP: argmin_{|v|=1} sum w_i (l_i . v)^2."""
    A = lines_n * np.sqrt(weights)[:, None]
    _, _, vt = np.linalg.svd(A, full_matrices=False)
    v = vt[-1]
    return v / (np.linalg.norm(v) + 1e-12)


def _wmean(x, w):
    return float(np.sum(x * w) / (np.sum(w) + 1e-12))


def _is_finite(v):
    return abs(v[2]) > 1e-9 and np.linalg.norm(v[:2] / v[2]) < FINITE_LIMIT


def _ray(v):
    """3-D direction of a VP under a camera with principal point at the centre and
    focal length = 1 half-diagonal (~53 deg diagonal FOV); used only for
    separation angles, not for the camera-consistency tests."""
    r = np.array([v[0], v[1], v[2]], float)
    return r / (np.linalg.norm(r) + 1e-12)


# ----------------------------------------------------------------------------
# main entry point
# ----------------------------------------------------------------------------
def extract(path_or_array, seed: int | None = None, keep_debug: bool = False) -> GeometryResult:
    rgb = load_image(path_or_array)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    diag = np.hypot(h, w)
    feats = {k: np.nan for k in FEATURE_NAMES}

    segs = detect_segments(gray)
    n = len(segs)
    feats["geo_log_nseg"] = float(np.log1p(n))
    L = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]) if n else np.zeros(0)
    feats["geo_line_density"] = float(L.sum() / diag)
    feats["geo_n_vps"] = 0.0
    labels = -np.ones(n, int)
    resid = np.full(n, np.nan)
    vps = []

    if n >= 2 * MIN_VP_SUPPORT:
        if seed is None:
            seed = int(hashlib.md5(str(path_or_array if not isinstance(path_or_array, np.ndarray)
                                        else path_or_array.shape).encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        p1, p2, _, _ = _normalise(segs, h, w)
        mid, d = (p1 + p2) / 2, p2 - p1
        Ln = np.linalg.norm(d, axis=1)
        lines = np.cross(np.c_[p1, np.ones(n)], np.c_[p2, np.ones(n)])
        lines /= np.linalg.norm(lines[:, :2], axis=1, keepdims=True) + 1e-12  # (a,b) unit
        total_len = Ln.sum()
        free = np.ones(n, bool)

        for k in range(MAX_VPS):
            idx = np.flatnonzero(free)
            if len(idx) < MIN_VP_SUPPORT:
                break
            prob = Ln[idx] / Ln[idx].sum()
            i = rng.choice(idx, N_HYP, p=prob)
            j = rng.choice(idx, N_HYP, p=prob)
            ok = i != j
            hyp = np.cross(lines[i[ok]], lines[j[ok]])
            nrm = np.linalg.norm(hyp, axis=1)
            hyp = hyp[nrm > 1e-9] / nrm[nrm > 1e-9, None]
            if len(hyp) == 0:
                break
            ang = _angles_to_vps(mid[idx], d[idx], hyp)          # (H, |idx|)
            score = ((ang < INLIER_DEG) * Ln[idx][None]).sum(1)
            v = hyp[np.argmax(score)]
            # refine twice (inliers -> LS fit -> inliers)
            for _ in range(2):
                a = _angles_to_vps(mid[idx], d[idx], v[None])[0]
                inl = idx[a < INLIER_DEG]
                if len(inl) < 2:
                    break
                v = _refine_vp(lines[inl], Ln[inl])
            a_all = _angles_to_vps(mid, d, v[None])[0]
            inl = idx[a_all[idx] < INLIER_DEG]
            if len(inl) < MIN_VP_SUPPORT or Ln[inl].sum() < MIN_VP_LEN_FRAC * total_len:
                break
            labels[inl] = k
            resid[inl] = a_all[inl]
            free[inl] = False
            vps.append(v)

            kk = k + 1
            feats[f"geo_frac_vp{kk}"] = float(Ln[inl].sum() / total_len)
            feats[f"geo_res_vp{kk}"] = _wmean(a_all[inl], Ln[inl])
            feats[f"geo_tight_vp{kk}"] = float(Ln[inl][a_all[inl] < TIGHT_DEG].sum() / Ln[inl].sum())
            if k == 0 and len(inl) >= 2 * MIN_VP_SUPPORT:
                # split-half cross-validation: fit on one half, measure the other
                perm = rng.permutation(inl)
                A, B = perm[: len(perm) // 2], perm[len(perm) // 2:]
                vA, vB = _refine_vp(lines[A], Ln[A]), _refine_vp(lines[B], Ln[B])
                eAB = _wmean(_angles_to_vps(mid[B], d[B], vA[None])[0], Ln[B])
                eBA = _wmean(_angles_to_vps(mid[A], d[A], vB[None])[0], Ln[A])
                feats["geo_cv_res_vp1"] = 0.5 * (eAB + eBA)

        feats["geo_n_vps"] = float(len(vps))
        long_ = Ln >= (LONG_LEN_FRAC * diag) / (diag / 2)
        if long_.any():
            feats["geo_frac_long_unexpl"] = float(Ln[long_ & (labels < 0)].sum() / Ln[long_].sum())

        # ---- relations between VPs --------------------------------------------------
        if len(vps) >= 2:
            rays = [_ray(v) for v in vps]
            seps, coss = [], []
            for a in range(len(vps)):
                for b in range(a + 1, len(vps)):
                    c = abs(float(np.dot(rays[a], rays[b])))
                    seps.append(np.degrees(np.arccos(min(1.0, c))))
                    coss.append(c)
            feats["geo_min_vp_sep_deg"] = float(min(seps))
            feats["geo_ortho_cos_min"] = float(min(coss))

            v1, v2 = vps[0], vps[1]
            if _is_finite(v1) and _is_finite(v2):
                # orthogonality with principal point at centre:
                # f^2 = -(x1 x2 + y1 y2) for VPs in normalised inhomogeneous coords
                q1, q2 = v1[:2] / v1[2], v2[:2] / v2[2]
                f2 = -float(np.dot(q1, q2))
                feats["geo_f2_12"] = float(np.sign(f2) * np.log1p(abs(f2)))

        if len(vps) == 3 and all(_is_finite(v) for v in vps):
            # principal point = orthocentre of the VP triangle (3 orthogonal directions)
            q = [v[:2] / v[2] for v in vps]
            A = np.array([q[2] - q[1], q[2] - q[0]])
            b = np.array([np.dot(q[0], q[2] - q[1]), np.dot(q[1], q[2] - q[0])])
            if abs(np.linalg.det(A)) > 1e-9:
                pp = np.linalg.solve(A, b)
                feats["geo_pp_offset"] = float(min(np.linalg.norm(pp), 10.0))

    res = GeometryResult(features=feats)
    if keep_debug:
        c = np.array([w / 2.0, h / 2.0]); s = diag / 2.0
        T = np.array([[s, 0, c[0]], [0, s, c[1]], [0, 0, 1.0]])   # normalised -> pixel
        res.segments, res.labels, res.residuals = segs, labels, resid
        res.vps_px = [T @ v for v in vps]
        res.image_rgb = rgb
    return res


def extract_features(path) -> dict:
    """Picklable wrapper for multiprocessing pools; never raises."""
    try:
        return extract(path).features
    except Exception as e:  # corrupt file etc.
        out = {k: np.nan for k in FEATURE_NAMES}
        out["geo_error"] = repr(e)[:200]
        return out


# ----------------------------------------------------------------------------
# visualisation (used by notebooks and the demo)
# ----------------------------------------------------------------------------
VP_COLOURS = [(230, 57, 70), (29, 120, 220), (46, 170, 90)]


def draw_evidence(res: GeometryResult, show_vp_rays: bool = True) -> np.ndarray:
    """Overlay: segments coloured by VP, thickness/alpha by convergence error;
    unexplained long lines in yellow; short rays towards each finite VP."""
    img = res.image_rgb.copy()
    over = img.copy()
    h, w = img.shape[:2]
    diag = np.hypot(h, w)
    for (x1, y1, x2, y2), lab, r in zip(res.segments, res.labels, res.residuals):
        p, q = (int(x1), int(y1)), (int(x2), int(y2))
        if lab >= 0:
            col = VP_COLOURS[lab % 3]
            cv2.line(over, p, q, col, 2, cv2.LINE_AA)
        elif np.hypot(x2 - x1, y2 - y1) >= LONG_LEN_FRAC * diag:
            cv2.line(over, p, q, (255, 200, 0), 2, cv2.LINE_AA)
    if show_vp_rays:
        for k, v in enumerate(res.vps_px):
            if abs(v[2]) < 1e-9:
                continue
            vx, vy = v[0] / v[2], v[1] / v[2]
            sel = np.flatnonzero(res.labels == k)
            for s in sel[np.argsort(-np.hypot(res.segments[sel, 2] - res.segments[sel, 0],
                                             res.segments[sel, 3] - res.segments[sel, 1]))[:12]]:
                mx, my = (res.segments[s, :2] + res.segments[s, 2:]) / 2
                # extend the segment's midpoint toward the VP (clipped to a long ray)
                dx, dy = vx - mx, vy - my
                t = min(1.0, 0.6 * diag / (np.hypot(dx, dy) + 1e-9))
                cv2.line(over, (int(mx), int(my)), (int(mx + t * dx), int(my + t * dy)),
                         VP_COLOURS[k % 3], 1, cv2.LINE_AA)
            if 0 <= vx < w and 0 <= vy < h:
                cv2.circle(over, (int(vx), int(vy)), 6, VP_COLOURS[k % 3], -1, cv2.LINE_AA)
    return cv2.addWeighted(over, 0.85, img, 0.15, 0)