"""Reference statistics for the demo's physics explanations: percentiles of each physics
measurement over REAL face crops from notebook 05 (140k/FFHQ, DiffusionFace and FF++ reals).

    python tools/build_demo_assets.py <notebook-05 output dir> demo/assets/real_face_reference.json
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from physfeat import face as F  # noqa: E402

nb05, out = Path(sys.argv[1]), Path(sys.argv[2])


def chunks(prefix, ext):
    files = sorted(glob.glob(str(nb05 / "**" / f"{prefix}_[0-9][0-9][0-9][0-9].{ext}"), recursive=True))
    paths, parts = [], []
    for f in files:
        paths += Path(f[: -len(ext) - 1] + ".paths.txt").read_text().splitlines()
        parts.append(pd.read_parquet(f))
    return paths, pd.concat(parts, ignore_index=True)


# original paths -> real/fake (same folder conventions as physfeat.facedata)
orig, meta = chunks("crop", "parquet")
import hashlib  # noqa: E402

real_crops = set()
for p, ok in zip(orig, meta.crop_ok.values):
    q = p.replace("\\", "/")
    is_real = ("/real-vs-fake/" in q and "/real/" in q) or "/diffface/Real/" in q or \
              ("faceforencispp" in q and "/real/" in q)
    if ok and is_real:
        real_crops.add(hashlib.md5(p.encode()).hexdigest()[:20] + ".jpg")

paths, feats = chunks("face_clean", "parquet")
mask = np.array([Path(p).name in real_crops for p in paths])
real = feats.loc[mask, F.FEATURE_NAMES].astype(float)
print(f"{mask.sum():,} real face crops")

ref = {"n_real": int(mask.sum()), "source": "notebook-05 real face crops (FFHQ via 140k, DiffusionFace Real, FF++ real)",
       "features": {}}
for f in F.FEATURE_NAMES:
    v = real[f].dropna().values
    if len(v) < 50:
        continue
    ref["features"][f] = {"defined_share": round(float(len(v) / len(real)), 4),
                          "percentiles": np.percentile(v, np.arange(101)).round(6).tolist()}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(ref, indent=1))
print("wrote", out, "with", len(ref["features"]), "features")
