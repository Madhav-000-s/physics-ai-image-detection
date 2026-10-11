# %% [markdown]
# # Physics-Augmented Deepfake Detection — Notebook 06: leave-one-family-out
#
# Notebook 05 trained on one generator (StyleGAN) and met its pre-registered criterion by a negligible margin
# (fused − CLIP = +0.0016 AUC). Two things limited it:
#
# 1. **The fusion could not learn when to trust physics.** Trained where CLIP is near-perfect (StyleGAN), it puts almost
#    all its weight on CLIP, even on sets where physics beats CLIP (SD text-to-image faces).
# 2. **The physics model learned StyleGAN-specific feature directions**, some of which flip on other generators
#    (e.g. corneal-highlight offset: 0.40 on face swaps, i.e. ≈ 0.60 in the physically expected direction).
#
# This notebook reuses notebook 05's cached features (no new feature extraction) and changes the training regime:
#
# * **Leave-one-family-out (LOFO):** train on real images + fakes from every family *except one*, test on the held-out
#   family. Families: GAN, diffusion, face swap (DiffSwap), face swap / reenactment (FF++), inpainting.
#   Every test set is split in half (by video for FF++); the halves used for testing are never trained on.
# * **Physics prior score (training-free):** each feature's direction is fixed by physics, as stated in the
#   `physfeat/face.py` docstring *before* notebook 05 ran (one light explains real skin shading; both eyes reflect the same
#   scene; real pupils are elliptical; the inner face and the outline face the same way). Score = mean of signed z-scores
#   (z relative to the training reals). It is fused with CLIP by the same logistic stacking.
#
# **Models compared in every fold:** CLIP probe · physics GBM (learned) · fused (CLIP + learned physics) ·
# physics prior (no training) · fused (CLIP + physics prior). All trained on clean + one JPEG copy (q50–95) per image.
#
# **Criterion (same as notebook 05, plus an effect size):** mean AUC over held-out test sets, fused − CLIP, paired bootstrap
# 95% CI excluding 0 **and** a gain of at least +0.01 AUC to call it a practically useful improvement.
#
# Finally, the **demo model** is trained on the training halves of *all* families and checked on the test halves.
#
# Inputs: notebook 05's `features/` folder (env `PHYSFAKE_NB05`, or attach the notebook-05 output on Kaggle).

# %%
import os, sys, json, glob, time, hashlib, re, warnings
from pathlib import Path
import numpy as np, pandas as pd

IN_KAGGLE = os.path.exists("/kaggle/input")
NB05 = Path(os.environ.get("PHYSFAKE_NB05", "/kaggle/input" if IN_KAGGLE else "../physfake_data/nb05"))
OUT_DIR = Path(os.environ.get("PHYSFAKE_OUT", "/kaggle/working" if IN_KAGGLE else "./work06")).resolve()
N_BOOT = int(os.environ.get("PHYSFAKE_N_BOOT", "500"))
MIN_GAIN = 0.01          # practically useful improvement (AUC)
SEED = 0
(OUT_DIR / "models").mkdir(parents=True, exist_ok=True)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
try:
    from IPython.display import display
except ImportError:
    display = print
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 100)

from physfeat import face as F, models as M, protocol as P
print("notebook-05 features from:", NB05, "· output:", OUT_DIR)

# %% [markdown]
# ## 1 · Rebuild notebook 05's table of images from its cached feature chunks
# Each chunk file `<name>_NNNN.<ext>` has a `.paths.txt` listing the images it covers. Labels come from the original
# image paths (dataset folder layout); crops are named by the MD5 of the original path.

# %%
def find_chunks(prefix, ext):
    files = sorted(f for f in glob.glob(str(NB05 / "**" / f"{prefix}_[0-9][0-9][0-9][0-9].{ext}"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no {prefix}_NNNN.{ext} under {NB05}")
    return files

def load(prefix, ext):
    paths, parts = [], []
    for f in find_chunks(prefix, ext):
        paths += Path(f[: -len(ext) - 1] + ".paths.txt").read_text().splitlines()
        parts.append(pd.read_parquet(f) if ext == "parquet" else np.load(f))
    data = pd.concat(parts, ignore_index=True) if ext == "parquet" else np.concatenate(parts)
    return [Path(p).name if ext != "parquet" or prefix != "crop" else p for p in paths], data

def parse(path):
    p = path.replace("\\", "/")
    name = Path(p).name
    if "/real-vs-fake/" in p:
        lab = 0 if "/real/" in p else 1
        split = "train" if "/train/" in p else "test"
        return dict(source="140k", method="real" if lab == 0 else "stylegan", label=lab, split=split, group=Path(p).stem)
    if "/diffface/" in p:
        m = p.split("/diffface/")[1].split("/")[0]
        return dict(source="diffface", method="real" if m == "Real" else m, label=int(m != "Real"), split="test",
                    group=Path(p).stem)
    if "faceforencispp" in p:
        if "/real/" in p:
            return dict(source="ffpp", method="real", label=0, split="test", group=name.split("_")[0])
        m = p.split("/fake/")[1].split("/")[0].lower()
        return dict(source="ffpp", method=m, label=1, split="test", group="_".join(name.split("_")[:2]))
    if "synthetic-faces-high-quality" in p:
        return dict(source="sfhq", method="sfhq_stylegan2", label=1, split="test", group=Path(p).stem)
    return None

orig_paths, crop_meta = load("crop", "parquet")
rows = []
for p, ok in zip(orig_paths, crop_meta.crop_ok.values):
    info = parse(p)
    if info is None or not ok:
        continue
    rows.append({"orig": p, "crop": hashlib.md5(p.encode()).hexdigest()[:20] + ".jpg", **info})
df = pd.DataFrame(rows).drop_duplicates("crop").set_index("crop")
print(f"{len(df):,} images with a face crop")
display(pd.crosstab([df.source, df.method], df.split))

# %%
def aligned(prefix, ext, crops):
    names, data = load(prefix, ext)
    pos = pd.Series(np.arange(len(names)), index=names)
    pos = pos[~pos.index.duplicated()]
    have = pd.Index(crops).isin(pos.index)
    idx = pos.reindex(crops).values
    if ext == "parquet":
        out = data.iloc[np.nan_to_num(idx, nan=0).astype(int)].reset_index(drop=True)
        out.loc[~have] = np.nan
        return out.reindex(columns=F.FEATURE_NAMES).astype(float)
    out = data[np.nan_to_num(idx, nan=0).astype(int)].astype(np.float32)
    out[~have] = np.nan
    return out

crops = df.index.tolist()
feat = {"clean": (aligned("clip_clean", "npy", crops), aligned("face_clean", "parquet", crops))}
for c in ("jpeg75", "jpeg50", "jpegrand"):
    feat[c] = (aligned(f"clip_{c}", "npy", crops), aligned(f"face_{c}", "parquet", crops))
cover = {c: float(np.isfinite(v[0][:, 0]).mean()) for c, v in feat.items()}
print("share of images with features per condition:", {k: round(v, 3) for k, v in cover.items()})
df = df.reset_index()

# %% [markdown]
# ## 2 · Families, splits, and the physics prior

# %%
def family(src, m):
    if src in ("140k", "sfhq"):
        return "GAN"
    if src == "ffpp":
        return "face swap / reenactment"
    return {"DiffSwap": "face swap", "Inpaint": "inpainting"}.get(m, "diffusion")

df["family"] = [family(s, m) if l == 1 else "real" for s, m, l in zip(df.source, df.method, df.label)]
# every non-140k image (and the 140k test split) is assigned to half A (trainable) or B (test), by group
h = df.group.map(lambda g: int(hashlib.md5(str(g).encode()).hexdigest()[:8], 16) % 2)
df["half"] = np.where(df.split == "train", "A", np.where(h == 0, "A", "B"))
df.loc[(df.source == "140k") & (df.split == "test"), "half"] = "B"     # keep notebook 05's 140k test set for testing
FAMILIES = ["GAN", "diffusion", "face swap", "face swap / reenactment", "inpainting"]
display(pd.crosstab([df.family, df.source], df.half))

# physics prior: directions fixed by physics (physfeat/face.py docstring, written before notebook 05 ran)
PRIOR = {
    "face_r2_sh1": -1,               # one light explains real skin shading better
    "face_lr_light_deg": +1,         # the two halves of a real face agree on the light
    "face_lr_cross_r2": -1,          # ... so one half's light predicts the other
    "face_lr_gap": +1,
    "face_spec_align": -1,           # skin highlights sit where the light says
    "face_eye_hl_iou": -1,           # both corneas reflect the same scene
    "face_eye_hl_offset": +1,
    "face_eye_hl_area_ratio": -1,
    "face_pupil_irreg_mean": +1,     # real pupils are elliptical
    "face_pose_inner_outer_deg": +1, # inner face and outline face the same way
}

def prior_score(X, ref):
    """Mean of physics-signed z-scores; z relative to `ref` (training reals). NaN features are skipped."""
    z = []
    for f, s in PRIOR.items():
        mu, sd = np.nanmean(ref[f]), np.nanstd(ref[f]) + 1e-9
        z.append(s * (X[f].values - mu) / sd)
    z = np.vstack(z)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nan_to_num(np.nanmean(z, axis=0), nan=0.0)

# %% [markdown]
# ## 3 · Training and evaluation helpers

# %%
from sklearn.linear_model import LogisticRegression

def train_rows(mask):
    """Clean copy + one JPEG copy per training image: jpegrand for 140k train, q75 / q50 otherwise
    (alternating by image), with the quality the fusion sees."""
    idx = np.flatnonzero(mask)
    clip_parts, phys_parts, q_parts, y_parts, g_parts = [], [], [], [], []
    clip_parts.append(feat["clean"][0][idx]); phys_parts.append(feat["clean"][1].iloc[idx])
    q_parts.append(np.full(len(idx), 95.0)); y_parts.append(df.label.values[idx]); g_parts.append(idx)
    is140 = (df.source.values[idx] == "140k") & (df.split.values[idx] == "train")
    for cond, sel, q in (("jpegrand", is140, None), ("jpeg75", ~is140 & (idx % 2 == 0), 75.0),
                         ("jpeg50", ~is140 & (idx % 2 == 1), 50.0)):
        j = idx[sel]
        c = feat[cond][0][j]
        ok = np.isfinite(c[:, 0])
        j = j[ok]
        clip_parts.append(feat[cond][0][j]); phys_parts.append(feat[cond][1].iloc[j])
        if q is None:
            from physfeat.imaging import jpegrand_quality
            q_parts.append(np.array([jpegrand_quality("/kaggle/working/crops/" + df.crop.values[k]) for k in j], float))
        else:
            q_parts.append(np.full(len(j), q))
        y_parts.append(df.label.values[j]); g_parts.append(j)
    return (np.concatenate(clip_parts), pd.concat(phys_parts, ignore_index=True), np.concatenate(q_parts),
            np.concatenate(y_parts), np.concatenate(g_parts))

PHYS_COLS = [c for c in F.FEATURE_NAMES if c != "face_found"]
NAMES = ["CLIP", "Physics (learned)", "Fused (CLIP + learned physics)", "Physics prior (no training)",
         "Fused (CLIP + physics prior)"]

def fit_fold(train_mask, log=lambda *a: None):
    Xc, Xp, q, y, g = train_rows(train_mask)
    cols = [c for c in PHYS_COLS if Xp[c].notna().mean() >= 0.01 and Xp[c].nunique() > 1]
    det = M.RobustDetector(cols, seed=SEED).fit(Xc, Xp, y, q, g, log=log)
    ref = Xp[y == 0]
    # out-of-fold CLIP logits for the prior fusion, from the same grouped folds the detector used
    from sklearn.model_selection import GroupKFold, cross_val_predict
    splits = list(GroupKFold(n_splits=5).split(Xc, y, g))
    oof_c = cross_val_predict(LogisticRegression(C=det.C, class_weight="balanced", max_iter=4000), Xc, y,
                              cv=splits, method="predict_proba")[:, 1]
    stack_prior = LogisticRegression(class_weight="balanced").fit(np.c_[M.logit(oof_c), prior_score(Xp, ref)], y)
    return {"det": det, "ref": ref, "stack_prior": stack_prior, "cols": cols}

def predict(model, idx, cond="clean", q=95.0):
    c, ph = feat[cond][0][idx], feat[cond][1].iloc[idx].reset_index(drop=True)
    s = model["det"].predict(c, ph, np.full(len(idx), q))
    N = M.RobustDetector.NAMES
    pr = prior_score(ph, model["ref"])
    fused_prior = model["stack_prior"].predict_proba(np.c_[M.logit(s[N["clip"]]), pr])[:, 1]
    return {"CLIP": s[N["clip"]], "Physics (learned)": s[N["phys"]], "Fused (CLIP + learned physics)": s[N["fused_q"]],
            "Physics prior (no training)": pr, "Fused (CLIP + physics prior)": fused_prior}

def sets_for(test_mask, fams=None):
    """Test sets (each fake method vs the B-half reals of its source) among rows in test_mask."""
    out = {}
    real_src = {"140k": "140k", "sfhq": "140k", "ffpp": "ffpp", "diffface": "diffface"}
    for (s, m), g in df[test_mask & (df.label == 1)].groupby(["source", "method"]):
        fam = family(s, m)
        if fams and fam not in fams:
            continue
        reals = np.flatnonzero(test_mask & (df.label == 0) & (df.source == real_src[s]))
        if len(g) >= 20 and len(reals) >= 20:
            out[f"{s}:{m}"] = {"real": reals, "fake": g.index.values, "family": fam, "seen": False}
    return out

# %% [markdown]
# ## 4 · Leave-one-family-out

# %%
CONDS = {"clean": 95.0, "jpeg75": 75.0, "jpeg50": 50.0}
lofo_scores = {c: [] for c in CONDS}   # per fold: (scores dict over the fold's test rows, sets with local indices)
t_all = time.time()
for held in FAMILIES:
    t0 = time.time()
    train_mask = (df.half == "A") & (df.family != held)
    test_mask = (df.half == "B").values
    sets = sets_for(test_mask, fams=[held])
    if not sets:
        print(f"{held}: no test sets, skipped"); continue
    model = fit_fold(train_mask.values)
    idx = np.unique(np.concatenate([np.r_[s["real"], s["fake"]] for s in sets.values()]))
    local = {k: {**v, "real": np.searchsorted(idx, v["real"]), "fake": np.searchsorted(idx, v["fake"])}
             for k, v in sets.items()}
    for cond, q in CONDS.items():
        ok = np.isfinite(feat[cond][0][idx, 0])
        assert ok.all(), f"missing {cond} features for held-out test images"
        lofo_scores[cond].append((predict(model, idx, cond, q), local))
    print(f"held out {held:25s}: {len(sets)} test sets, train {int(train_mask.sum()):,} images, "
          f"{(time.time() - t0) / 60:.1f} min")
print(f"LOFO total {(time.time() - t_all) / 60:.1f} min")

def combine(fold_list):
    """Concatenate folds into one score array per model (a real image scored by several folds appears once per fold)."""
    scores = {n: [] for n in NAMES}
    sets, off = {}, 0
    for s, local in fold_list:
        n = len(next(iter(s.values())))
        for k in NAMES:
            scores[k].append(s[k])
        for name, v in local.items():
            sets[name] = {**v, "real": v["real"] + off, "fake": v["fake"] + off}
        off += n
    return {k: np.concatenate(v) for k, v in scores.items()}, sets

PAIRS = [("Fused (CLIP + learned physics)", "CLIP"), ("Fused (CLIP + physics prior)", "CLIP"),
         ("Physics prior (no training)", "Physics (learned)")]
lofo_res, lofo_boot = [], {}
for cond in CONDS:
    s, sets = combine(lofo_scores[cond])
    lofo_res.append(P.evaluate_sets(s, sets).assign(condition=cond))
    lofo_boot[cond] = P.bootstrap_sets(s, sets, PAIRS, n_boot=N_BOOT, seed=SEED)
lofo_res = pd.concat(lofo_res, ignore_index=True)
print("LOFO: AUC per held-out test set (clean)")
display(P.summary_sets(lofo_res[lofo_res.condition == "clean"], models=NAMES).round(4))
print("LOFO: mean AUC over held-out sets [95% CI]")
display(pd.DataFrame({c: {m: "{:.4f} [{:.4f}, {:.4f}]".format(*lofo_boot[c][0][m]) for m in NAMES} for c in CONDS}))

# %% [markdown]
# ### Verdict (criterion fixed above: CI excludes 0 and gain ≥ +0.01 AUC)

# %%
rows = []
for cond, (_, diffs) in lofo_boot.items():
    for name, (d, lo, hi, p) in diffs.items():
        useful = lo > 0 and d >= MIN_GAIN
        rows.append({"condition": cond, "comparison": name, "diff": d, "ci_lo": lo, "ci_hi": hi, "P(diff<=0)": p,
                     "verdict": "HELPS (useful)" if useful else "helps (negligible)" if lo > 0 else
                                "HURTS" if hi < 0 else "no measurable difference"})
lofo_diff = pd.DataFrame(rows)
display(lofo_diff.round(4))

# %% [markdown]
# ## 5 · Demo model: trained on the A-halves of every family, checked on the B-halves

# %%
t0 = time.time()
demo = fit_fold((df.half == "A").values, log=print)
test_mask = (df.half == "B").values
sets_all = sets_for(test_mask)
idx = np.unique(np.concatenate([np.r_[s["real"], s["fake"]] for s in sets_all.values()]))
local = {k: {**v, "real": np.searchsorted(idx, v["real"]), "fake": np.searchsorted(idx, v["fake"])}
         for k, v in sets_all.items()}
demo_res, demo_boot = [], {}
for cond, q in CONDS.items():
    s = predict(demo, idx, cond, q)
    demo_res.append(P.evaluate_sets(s, local).assign(condition=cond))
    demo_boot[cond] = P.bootstrap_sets(s, local, PAIRS, n_boot=N_BOOT, seed=SEED)
demo_res = pd.concat(demo_res, ignore_index=True)
print(f"demo model trained in {(time.time() - t0) / 60:.1f} min")
print("Demo model: AUC per test set (clean; all families seen in training)")
display(P.summary_sets(demo_res[demo_res.condition == "clean"], models=NAMES).round(4))
print("Demo model: balanced accuracy at threshold 0.5 (clean)")
display(P.summary_sets(demo_res[demo_res.condition == "clean"], "bal_acc", models=NAMES).round(4))
print("Demo model: fused − CLIP [95% CI]")
display(pd.DataFrame({c: {k: "{:+.4f} [{:+.4f}, {:+.4f}]".format(*v[:3]) for k, v in demo_boot[c][1].items()}
                      for c in CONDS}))

# %%
import joblib
joblib.dump({"detector": demo["det"], "stack_prior": demo["stack_prior"], "prior": PRIOR,
             "prior_ref_mean": demo["ref"][list(PRIOR)].mean().to_dict(),
             "prior_ref_std": demo["ref"][list(PRIOR)].std(ddof=0).to_dict(),
             "phys_cols": demo["cols"], "clip_model": "openai/clip-vit-large-patch14",
             "crop": {"size": 256, "margin": 0.3, "jpeg_q": 95},
             "train_data": "notebook-05 features, A-halves of all families"},
            OUT_DIR / "models" / "face_detector_all.joblib")
out = {
    "lofo_auc_clean": P.summary_sets(lofo_res[lofo_res.condition == "clean"], models=NAMES).round(4).to_dict(),
    "lofo_mean_auc": {c: {m: np.round(v, 4).tolist() for m, v in b[0].items()} for c, b in lofo_boot.items()},
    "lofo_diffs": {c: {k: np.round(v, 4).tolist() for k, v in b[1].items()} for c, b in lofo_boot.items()},
    "demo_auc_clean": P.summary_sets(demo_res[demo_res.condition == "clean"], models=NAMES).round(4).to_dict(),
    "demo_bal_acc_clean": P.summary_sets(demo_res[demo_res.condition == "clean"], "bal_acc", models=NAMES).round(4).to_dict(),
    "demo_diffs": {c: {k: np.round(v, 4).tolist() for k, v in b[1].items()} for c, b in demo_boot.items()},
    "demo_stack_prior_coef[clip_logit, prior]": demo["stack_prior"].coef_[0].round(4).tolist(),
    "demo_stack_quality_aware_coef": demo["det"].stack_q.coef_[0].round(4).tolist(),
}
(OUT_DIR / "results_lofo.json").write_text(json.dumps(out, indent=2, default=str))
lofo_res.to_csv(OUT_DIR / "results_lofo.csv", index=False)
demo_res.to_csv(OUT_DIR / "results_demo_model.csv", index=False)
lofo_diff.to_csv(OUT_DIR / "lofo_diffs.csv", index=False)
print("saved", OUT_DIR / "results_lofo.json")
