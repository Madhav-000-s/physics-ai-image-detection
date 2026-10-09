"""Chunked, resumable feature computation (a chunk is reused only if its path list matches)."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd


def chunked(paths, name, fn, ext, feat_dir, chunk=2000, log=print):
    feat_dir = Path(feat_dir)
    feat_dir.mkdir(parents=True, exist_ok=True)
    paths = list(paths)
    outs = []
    for ci, s in enumerate(range(0, len(paths), chunk)):
        ps = paths[s:s + chunk]
        f, fp = feat_dir / f"{name}_{ci:04d}.{ext}", feat_dir / f"{name}_{ci:04d}.paths.txt"
        if f.exists() and fp.exists() and fp.read_text().splitlines() == ps:
            outs.append(pd.read_parquet(f) if ext == "parquet" else np.load(f))
            continue
        t0 = time.time()
        out = fn(ps)
        if ext == "parquet":
            out.to_parquet(f)
        else:
            np.save(f, out)
        fp.write_text("\n".join(ps))
        outs.append(out)
        log(f"{name} chunk {ci}: {len(ps)} imgs in {time.time() - t0:.0f}s")
    if not outs:
        return None
    return pd.concat(outs, ignore_index=True) if ext == "parquet" else np.concatenate(outs)