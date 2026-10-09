"""Find GenImage-style images on disk and build a labelled manifest.

Works with the original GenImage layout
    imagenet_ai_0419_sdv4/train/ai/*.png, .../val/nature/*.JPEG
and with the common Kaggle re-uploads (tiny-genimage, unbiased-tiny-genimage),
which use folder names like  midjourney/ai, sd14/nature, real/, fake/ ...
Labels, generator and split are inferred from path components.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .imaging import IMG_EXTS

REAL_TOKENS = {"nature", "real", "0real", "natural", "imagenet", "authentic", "genuine", "0"}
FAKE_TOKENS = {"ai", "fake", "1fake", "generated", "synthetic", "aigenerated", "1"}
SPLIT_TOKENS = {"train": "train", "training": "train", "val": "val", "valid": "val",
                "validation": "val", "test": "test", "testing": "test"}

# normalised alias -> canonical generator name (checked as substrings, longest first)
GEN_ALIASES = {
    "midjourney": "midjourney", "imagenetmidjourney": "midjourney", "mj": "midjourney",
    "stablediffusionv14": "sd14", "stablediffusionv4": "sd14", "sdv14": "sd14", "sd14": "sd14",
    "sdv4": "sd14",
    "stablediffusionv15": "sd15", "stablediffusionv5": "sd15", "sdv15": "sd15", "sd15": "sd15",
    "sdv5": "sd15",
    "adm": "adm", "glide": "glide", "wukong": "wukong", "vqdm": "vqdm", "biggan": "biggan",
    "sdxl": "sdxl", "flux": "flux", "dalle": "dalle",
    "stablediffusion": "sd", "sd": "sd",   # version-less fallbacks (checked after the specific ones)
}
_ALIAS_ORDER = sorted(GEN_ALIASES, key=len, reverse=True)
_SHORT = {"mj", "adm", "sd"}  # short aliases must match a whole component


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def parse_path(path: str) -> dict:
    parts = [_norm(p) for p in Path(path).parts[:-1]]
    label, gen, split = None, None, None
    for p in reversed(parts):               # innermost folder wins
        if label is None and p in REAL_TOKENS:
            label = 0
        elif label is None and p in FAKE_TOKENS:
            label = 1
        if split is None and p in SPLIT_TOKENS:
            split = SPLIT_TOKENS[p]
        if gen is None:
            for a in _ALIAS_ORDER:
                if (a in _SHORT and (p == a or p.endswith(a) and p.startswith("imagenet"))) or \
                   (a not in _SHORT and a in p):
                    gen = GEN_ALIASES[a]
                    break
    if label is None and gen is not None:
        # flat layouts (e.g. unbiased-tiny-genimage: BigGAN/, ADM/, ..., plus a real folder):
        # a generator-named folder with no real/fake sub-folder holds that generator's fakes
        label = 1
    return {"label": label, "subset": gen, "split": split}


def scan(roots, exts=IMG_EXTS) -> pd.DataFrame:
    rows = []
    for root in roots:
        for dp, _, fns in os.walk(root, followlinks=True):
            for fn in fns:
                if os.path.splitext(fn)[1].lower() in exts:
                    p = os.path.join(dp, fn)
                    rows.append({"path": p, **parse_path(os.path.relpath(p, root))})
    df = pd.DataFrame(rows, columns=["path", "label", "subset", "split"])
    # fakes are attributed to the generator of their subset; reals are just "real"
    df["generator"] = np.where(df["label"] == 0, "real", df["subset"])
    return df


def folder_tree(root, max_depth=4, max_children=12) -> str:
    """Compact tree with image counts, to show what the scanner is looking at."""
    lines = []

    def count(d):
        n = 0
        for _, _, fns in os.walk(d):
            n += sum(os.path.splitext(f)[1].lower() in IMG_EXTS for f in fns)
        return n

    def rec(d, depth):
        subs = sorted([e for e in os.scandir(d) if e.is_dir()], key=lambda e: e.name)
        for e in subs[:max_children]:
            lines.append("  " * depth + f"{e.name}/  ({count(e.path)} imgs)")
            if depth + 1 < max_depth:
                rec(e.path, depth + 1)
        if len(subs) > max_children:
            lines.append("  " * depth + f"... {len(subs) - max_children} more")
    lines.append(f"{root}/  ({count(root)} imgs)")
    rec(root, 1)
    return "\n".join(lines)