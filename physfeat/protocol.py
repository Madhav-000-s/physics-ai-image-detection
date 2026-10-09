"""Cross-generator evaluation protocol.

Train:  real images (train split)  vs  fakes from ONE generator (train split)
Test:   for every generator g:  real images (val split)  vs  fakes from g (val split)

The generator used for training is "seen"; every other generator is "unseen".
Generalisation = performance on the unseen ones.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

PREFERRED_TRAIN_GENERATORS = ["sd14", "sd15", "sd", "adm", "glide", "vqdm", "wukong", "midjourney", "biggan"]


def pick_train_generator(df: pd.DataFrame, requested: str = "auto") -> str:
    gens = df.loc[(df.label == 1) & (df.split == "train"), "generator"].value_counts()
    if gens.empty:
        raise ValueError("No fake images in the train split — the folder scanner labelled nothing as fake. "
                         "Check the folder tree printed above and extend GEN_ALIASES / FAKE_TOKENS "
                         "in physfeat/dataset.py. Counts per generator/split:\n"
                         + df.groupby(["generator", "split"]).size().to_string())
    if requested != "auto":
        if requested not in gens.index:
            raise ValueError(f"{requested!r} has no training fakes; available: {list(gens.index)}")
        return requested
    for g in PREFERRED_TRAIN_GENERATORS:
        if gens.get(g, 0) >= 200:
            return g
    return gens.index[0]


def train_mask(df, train_gen):
    return (df.split == "train").values & ((df.label == 0) | (df.generator == train_gen)).values


def test_masks(df, min_fakes=5):
    """dict generator -> boolean mask over df (val reals + val fakes of that generator)."""
    val = (df.split == "val").values
    real = val & (df.label == 0).values
    out = {}
    for g in sorted(df.loc[df.label == 1, "generator"].dropna().unique()):
        m = real | (val & (df.generator == g).values)
        if (m & (df.label == 1).values).sum() >= min_fakes:
            out[g] = m
    return out


def evaluate(scores: dict, df: pd.DataFrame, train_gen: str, extra_mask=None) -> pd.DataFrame:
    """scores: model name -> array of P(fake) over the whole df. Returns a tidy table."""
    rows = []
    for g, m in test_masks(df).items():
        if extra_mask is not None:
            m = m & extra_mask
        y = df.label.values[m]
        if len(np.unique(y)) < 2:
            continue
        for name, s in scores.items():
            p = s[m]
            rows.append({"generator": g, "seen": g == train_gen, "model": name,
                         "auc": roc_auc_score(y, p),
                         "bal_acc": balanced_accuracy_score(y, p > 0.5),
                         "n_fake": int(y.sum()), "n_real": int((1 - y).sum())})
    return pd.DataFrame(rows)


def summary_table(res: pd.DataFrame, metric="auc") -> pd.DataFrame:
    if res.empty:
        return pd.DataFrame({"note": ["no generator had enough validation fakes to evaluate"]})
    t = res.pivot_table(index=["generator", "seen"], columns="model", values=metric).reset_index()
    t = t.sort_values(["seen", "generator"], ascending=[False, True]).set_index("generator")
    unseen = t[~t.seen].drop(columns="seen")
    t = t.drop(columns="seen")
    t.loc["MEAN (unseen)"] = unseen.mean()
    return t


def bootstrap_mean_auc(scores: dict, df: pd.DataFrame, train_gen: str, pairs, n_boot=500, seed=0):
    """Paired bootstrap of the mean AUC over unseen generators.

    Resamples real and fake test images (stratified by generator, reals shared across
    generators exactly as in `evaluate`) and recomputes every model's mean unseen AUC on
    the same resample. Returns per-model [mean, lo, hi] and per-pair difference [mean, lo, hi, P(diff<=0)].
    """
    rng = np.random.default_rng(seed)
    val = (df.split == "val").values
    real_idx = np.flatnonzero(val & (df.label == 0).values)
    gens = [g for g in test_masks(df) if g != train_gen]
    fake_idx = {g: np.flatnonzero(val & (df.generator == g).values) for g in gens}
    names = list(scores)
    S = {k: np.asarray(v) for k, v in scores.items()}

    def mean_auc(ri, fi):
        out = {k: [] for k in names}
        for g in gens:
            idx = np.r_[ri, fi[g]]
            yy = np.r_[np.zeros(len(ri)), np.ones(len(fi[g]))]
            for k in names:
                out[k].append(roc_auc_score(yy, S[k][idx]))
        return {k: float(np.mean(v)) for k, v in out.items()}

    point = mean_auc(real_idx, fake_idx)
    boots = []
    for _ in range(n_boot):
        ri = rng.choice(real_idx, len(real_idx))
        fi = {g: rng.choice(v, len(v)) for g, v in fake_idx.items()}
        boots.append(mean_auc(ri, fi))
    B = pd.DataFrame(boots)
    models = {k: [point[k], float(B[k].quantile(.025)), float(B[k].quantile(.975))] for k in names}
    diffs = {}
    for a, b in pairs:
        d = B[a] - B[b]
        diffs[f"{a}  minus  {b}"] = [point[a] - point[b], float(d.quantile(.025)), float(d.quantile(.975)),
                                    float((d <= 0).mean())]
    return models, diffs