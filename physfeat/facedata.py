"""Face datasets for notebook 05: find or fetch images and build one manifest.

Manifest columns
----------------
path      image file (original, before cropping)
source    dataset: "140k", "sfhq", "ffpp", "diffface"
method    generator / manipulation ("real" for pristine images)
label     0 real, 1 fake
split     "train" or "test"
group     id for grouped sampling (e.g. FF++ video), so test frames come from many videos

Test sets are built by `test_sets()`: each pairs fakes of one method with reals of the
matching source, so the detector cannot win by telling one real-image domain from another.

Sources
-------
* 140k Real and Fake Faces (Kaggle xhlulu/140k-real-and-fake-faces): FFHQ reals vs StyleGAN
  fakes, 256 px JPEG, real-vs-fake/{train,valid,test}/{real,fake}/  -> training data + seen test
* SFHQ part 1 (Kaggle selfishgene/synthetic-faces-high-quality-sfhq-part-1): StyleGAN2 faces
  "brought to life" from paintings / 3D models, 1024 px. Fake only -> paired with 140k test reals
* FF++ frames (Kaggle adham7elmy/faceforencispp-extracted-frames): fake/<method>/*.png, real/*.png
* DiffusionFace (Zenodo 10865300 / 10865065, CC BY 4.0): one tar per method, streamed so only
  the first few hundred images of each ~3 GB archive are downloaded
"""
from __future__ import annotations

import os
import re
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .imaging import IMG_EXTS

ZENODO_P1 = "https://zenodo.org/records/10865300/files/{}.tar?download=1"
ZENODO_P2 = "https://zenodo.org/records/10865065/files/{}.tar?download=1"
DIFFFACE = {   # method name -> archive URL
    **{m: ZENODO_P1.format(m) for m in [
        "Real", "ADM", "DDIM", "DDPM", "LDM", "PNDM", "DiffSwap", "Inpaint", "MM_CelebA",
        "SDv15_DS0.3", "SDv15_DS0.5", "SDv15_DS0.7", "SDv21_DS0.3", "SDv21_DS0.5", "SDv21_DS0.7"]},
    **{m: ZENODO_P2.format(m) for m in [
        "stable_diffusion_v_1_5_text2img_p3g7", "stable_diffusion_v_1_5_text2img_p4g5",
        "stable_diffusion_v_1_5_text2img_p5g3", "stable_diffusion_v_2_1_text2img_p0g5",
        "stable_diffusion_v_2_1_text2img_p1g7", "stable_diffusion_v_2_1_text2img_p2g3"]},
}
# how each DiffusionFace method is reported
DIFFFACE_FAMILY = {"DiffSwap": "face swap", "Inpaint": "inpainting"}

COLUMNS = ["path", "source", "method", "label", "split", "group"]


def _images(root):
    for dp, _, fns in os.walk(root, followlinks=True):
        for fn in fns:
            if os.path.splitext(fn)[1].lower() in IMG_EXTS:
                yield os.path.join(dp, fn)


def _parts(path, root):
    return [p.lower() for p in Path(os.path.relpath(path, root)).parts[:-1]]


# ----------------------------------------------------------------------------
# scanners for attached (Kaggle) datasets
# ----------------------------------------------------------------------------
def scan_140k(root) -> pd.DataFrame:
    rows = []
    split_map = {"train": "train", "test": "test"}          # "valid" is left out (unused)
    for p in _images(root):
        parts = _parts(p, root)
        lab = 0 if "real" in parts else 1 if "fake" in parts else None
        split = next((split_map[s] for s in parts if s in split_map), None)
        if lab is None or split is None:
            continue
        rows.append({"path": p, "source": "140k", "method": "real" if lab == 0 else "stylegan",
                     "label": lab, "split": split, "group": Path(p).stem})
    return pd.DataFrame(rows, columns=COLUMNS)


def scan_sfhq(root) -> pd.DataFrame:
    rows = [{"path": p, "source": "sfhq", "method": "sfhq_stylegan2", "label": 1, "split": "test",
             "group": Path(p).stem}
            for p in _images(root) if "images" in _parts(p, root) or not _parts(p, root)]
    return pd.DataFrame(rows, columns=COLUMNS)


def scan_ffpp(root) -> pd.DataFrame:
    rows = []
    for p in _images(root):
        parts = _parts(p, root)
        stem = Path(p).stem
        if "real" in parts:
            method, lab, video = "real", 0, stem.split("_")[0]
        elif "fake" in parts:
            i = parts.index("fake")
            if i + 1 >= len(parts):
                continue
            method, lab = parts[i + 1], 1
            video = "_".join(stem.split("_")[:2])            # target_source video pair
        else:
            continue
        rows.append({"path": p, "source": "ffpp", "method": method, "label": lab, "split": "test",
                     "group": video})
    return pd.DataFrame(rows, columns=COLUMNS)


def find_root(input_root, marker, max_depth=5):
    """First directory under input_root whose path (lower-case) contains `marker`."""
    marker = marker.lower()
    for dp, dns, _ in os.walk(input_root, followlinks=True):
        if dp.count(os.sep) - str(input_root).count(os.sep) >= max_depth:
            dns[:] = []
        if marker in dp.lower().replace("\\", "/"):
            return dp
    return None


# ----------------------------------------------------------------------------
# DiffusionFace: stream the start of each tar
# ----------------------------------------------------------------------------
def stream_tar(url, out_dir, n, stride=1, timeout=120, log=print) -> list:
    """Save up to n image members of a remote tar to out_dir, keeping every `stride`-th image,
    without downloading the rest of the archive. Re-running reuses what is on disk."""
    out_dir = Path(out_dir)
    done = out_dir / f"_done_n{n}_s{stride}.txt"
    if done.exists():
        return [str(out_dir / p) for p in done.read_text().splitlines() if p]
    out_dir.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "physfeat/0.2 (research; python urllib)"})
    saved, seen = [], 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        with tarfile.open(fileobj=r, mode="r|*") as tf:
            for m in tf:
                if not m.isfile() or os.path.splitext(m.name)[1].lower() not in IMG_EXTS:
                    continue
                seen += 1
                if (seen - 1) % stride:
                    continue
                # keep the archive's folders (sanitised; no absolute paths or "..")
                parts = [re.sub(r"[^A-Za-z0-9._-]", "_", p) for p in Path(m.name).parts if p not in ("", ".", "..", "/")]
                parts = [p for p in parts if p.strip("_")] or [f"img{seen}"]
                name = "/".join(parts)
                (out_dir / name).parent.mkdir(parents=True, exist_ok=True)
                (out_dir / name).write_bytes(tf.extractfile(m).read())
                saved.append(name)
                if len(saved) >= n:
                    break
    done.write_text("\n".join(saved))
    log(f"  {url.split('/files/')[-1].split('?')[0]}: {len(saved)} images (scanned {seen})")
    return [str(out_dir / p) for p in saved]


def fetch_diffface(methods, out_root, n_per_method, n_real, stride=1, urls=None, log=print) -> pd.DataFrame:
    urls = urls or DIFFFACE
    rows = []
    for m in ["Real"] + [x for x in methods if x != "Real"]:
        try:
            paths = stream_tar(urls[m], Path(out_root) / m, n_real if m == "Real" else n_per_method,
                               stride=stride, log=log)
        except Exception as e:   # one unreachable archive should not kill the run
            log(f"  {m}: FAILED ({type(e).__name__}: {str(e)[:120]}) -> skipped")
            continue
        lab = 0 if m == "Real" else 1
        rows += [{"path": p, "source": "diffface", "method": "real" if lab == 0 else m, "label": lab,
                  "split": "test", "group": Path(p).stem} for p in paths]
    return pd.DataFrame(rows, columns=COLUMNS)


# ----------------------------------------------------------------------------
# sampling and test sets
# ----------------------------------------------------------------------------
def sample(df, n_train_per_class, n_test_per_method, max_per_group=None, seed=0) -> pd.DataFrame:
    """Deterministic caps: n_train_per_class per label in train; n_test_per_method per
    (source, method) in test, spread over groups (videos) first."""
    out = []
    tr = df[df.split == "train"]
    for _, g in tr.groupby("label"):
        out.append(g.sample(min(len(g), n_train_per_class), random_state=seed))
    te = df[df.split == "test"]
    for _, g in te.groupby(["source", "method"]):
        g = g.sample(frac=1.0, random_state=seed)
        if max_per_group:
            g = g[g.groupby("group").cumcount() < max_per_group]
        # round-robin over groups so a cap keeps as many distinct videos as possible
        g = g.assign(_r=g.groupby("group").cumcount()).sort_values("_r", kind="stable").drop(columns="_r")
        out.append(g.head(n_test_per_method))
    return pd.concat(out, ignore_index=True)


def family_of(source, method):
    if source == "140k":
        return "GAN (seen)"
    if source == "sfhq":
        return "GAN"
    if source == "ffpp":
        return "face swap / reenactment"
    return DIFFFACE_FAMILY.get(method, "diffusion")


def test_sets(df, min_n=20) -> dict:
    """name -> {"real": idx, "fake": idx, "family": str, "seen": bool} over df's positional index.
    Fakes are paired with reals of the same source (SFHQ, which has none, uses the 140k reals)."""
    assert (df.index.values == np.arange(len(df))).all(), "test_sets needs df.reset_index(drop=True)"
    te = df.split.values == "test"
    lab = df.label.values
    src = df.source.values
    out = {}
    real_src = {"140k": "140k", "sfhq": "140k", "ffpp": "ffpp", "diffface": "diffface"}
    for (s, m), g in df[te & (lab == 1)].groupby(["source", "method"]):
        reals = np.flatnonzero(te & (lab == 0) & (src == real_src[s]))
        fakes = g.index.values
        if len(reals) < min_n or len(fakes) < min_n:
            continue
        name = f"{s}:{m}"
        out[name] = {"real": reals, "fake": fakes, "family": family_of(s, m), "seen": s == "140k"}
    return out
