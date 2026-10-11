# %% [markdown]
# # Physics-Augmented Deepfake Detection — Notebook 05: faces
#
# Notebooks 01–04 found that physics features computed from *guessed* geometry (monocular normals, vanishing
# points on ImageNet scenes) add nothing to a CLIP detector. Faces come with precise geometry: a 3D face mesh
# from MediaPipe. This notebook asks the same question on faces, with the protocol fixed in advance.
#
# **Train:** real (FFHQ) vs StyleGAN faces — *140k Real and Fake Faces*. Every model gets clean + JPEG-augmented copies.
# **Test on unseen fakes**, each paired with real images from the same source:
#
# | family | test sets |
# |---|---|
# | GAN | SFHQ part 1 (StyleGAN2, faces "brought to life" from paintings / 3D models) |
# | face swap / reenactment | FF++ Deepfakes, Face2Face, FaceSwap, FaceShifter, NeuralTextures |
# | diffusion | DiffusionFace: ADM, DDIM, LDM, PNDM, SD img2img, SD text2img, text-to-face |
# | face swap / inpainting (diffusion) | DiffusionFace DiffSwap, Inpaint |
#
# **Models:** CLIP ViT-L/14 linear probe (baseline) · face-physics gradient boosting (`physfeat.face`) · stacked fusion,
# with and without the JPEG quality (`models.RobustDetector`, as in notebook 03).
#
# **Pre-registered success criterion:** mean AUC over unseen test sets, *fused (quality-aware) − CLIP*, paired bootstrap
# 95% CI excludes 0. Checked on clean, JPEG q75 and JPEG q50 test images.
#
# **Shortcut guard:** every image, real or fake, is reduced to the same canonical form before any model sees it:
# MediaPipe face detection → square face crop (margin 0.3) → 256×256 → JPEG q95.
#
# **Kaggle setup**
# * *Add Input* → **140k Real and Fake Faces** (`xhlulu/140k-real-and-fake-faces`, required).
#   Optional: **Synthetic Faces High Quality (SFHQ) part 1** (`selfishgene/synthetic-faces-high-quality-sfhq-part-1`)
#   and **FaceForencis++ extracted frames** (`adham7elmy/faceforencispp-extracted-frames`).
# * *Settings* → **GPU T4**, **Internet On** (installs MediaPipe + this repo's `physfeat`, streams DiffusionFace from Zenodo).
# * **Save Version → Save & Run All**. Every heavy step is chunked and resumable. Expected ≈ 1.5–2.5 h with the default caps.
#
# DiffusionFace (CC BY 4.0) is streamed: only the first images of each ~3 GB archive are downloaded.
# FF++ frames are an unofficial mirror with unclear licensing: report it as such, and do not redistribute.

# %%
import os, sys, json, time, hashlib, subprocess, warnings
from pathlib import Path

IN_KAGGLE = os.path.exists("/kaggle/input")
SMOKE = os.environ.get("PHYSFAKE_SMOKE", "") == "1"        # local dry run: tiny data, random CLIP features
OUT_DIR = Path(os.environ.get("PHYSFAKE_OUT", "/kaggle/working" if IN_KAGGLE else "./work05")).resolve()
RAW_DIR = Path(os.environ.get("PHYSFAKE_RAW", "/tmp/physfake_raw" if IN_KAGGLE else str(OUT_DIR / "raw")))
INPUT_ROOT = os.environ.get("PHYSFAKE_INPUT_ROOT", "/kaggle/input")
PKG = os.environ.get("PHYSFAKE_PKG", "git+https://github.com/Madhav-000-s/physics-ai-image-detection")
CLIP_MODEL = os.environ.get("PHYSFAKE_CLIP_MODEL", "openai/clip-vit-large-patch14")

N_TRAIN_PER_CLASS = int(os.environ.get("PHYSFAKE_N_TRAIN", "10000"))   # 140k train: per class
N_TEST_PER_METHOD = int(os.environ.get("PHYSFAKE_N_TEST", "1000"))     # per (source, method), incl. reals
FFPP_MAX_PER_VIDEO = 3                                                  # frames per FF++ video
DIFFFACE_METHODS = os.environ.get("PHYSFAKE_DIFFFACE", ",".join([
    "ADM", "DDIM", "LDM", "PNDM", "DiffSwap", "Inpaint", "MM_CelebA", "SDv15_DS0.5", "SDv21_DS0.5",
    "stable_diffusion_v_1_5_text2img_p4g5", "stable_diffusion_v_2_1_text2img_p1g7"])).split(",")
DIFFFACE_METHODS = [m for m in DIFFFACE_METHODS if m]
DIFFFACE_N, DIFFFACE_N_REAL = int(os.environ.get("PHYSFAKE_DIFFFACE_N", "800")), 2 * N_TEST_PER_METHOD
DIFFFACE_URLS = json.loads(os.environ.get("PHYSFAKE_DIFFFACE_URLS", "{}")) or None   # override (smoke test)
ROOT_OVERRIDES = json.loads(os.environ.get("PHYSFAKE_ROOTS", "{}"))                  # {"140k": path, ...}

CROP_SIZE, CROP_MARGIN, CROP_JPEG_Q = 256, 0.3, 95
PERTURBATIONS = ["jpeg75", "jpeg50"]
N_BOOT = int(os.environ.get("PHYSFAKE_N_BOOT", "500"))
N_CPU = os.cpu_count() or 2
CHUNK = 2000
SEED = 0

FEAT_DIR, CROP_DIR, MODEL_DIR = OUT_DIR / "features", OUT_DIR / "crops", OUT_DIR / "models"
for d in (OUT_DIR, FEAT_DIR, CROP_DIR, MODEL_DIR, RAW_DIR):
    d.mkdir(parents=True, exist_ok=True)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
print(f"output: {OUT_DIR} · raw: {RAW_DIR} · CPUs: {N_CPU} · smoke: {SMOKE}")

# %% [markdown]
# ## 1 · Install MediaPipe and `physfeat` (this project's feature package, from GitHub)

# %%
def pip(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check=True)

if os.environ.get("PHYSFAKE_SKIP_INSTALL", "") != "1":
    try:
        import mediapipe  # noqa: F401
    except ImportError:
        pip("mediapipe")
    pip("--no-deps", "--force-reinstall", PKG)
try:
    PKG_COMMIT = subprocess.run(["git", "ls-remote", PKG.replace("git+", ""), "HEAD"], capture_output=True,
                                text=True, timeout=30).stdout.split()[0][:12] if PKG.startswith("git+") else "local"
except Exception:
    PKG_COMMIT = "unknown"
print("physfeat commit:", PKG_COMMIT)

# %%
import importlib
import numpy as np, pandas as pd
from multiprocessing import Pool
from tqdm.auto import tqdm
from physfeat import face as F, facedata as FD, models as M, protocol as P, workers as W, imaging as I
from physfeat.cache import chunked
for m in (F, FD, M, P, W, I):
    importlib.reload(m)
try:
    from IPython.display import display
except ImportError:
    display = print
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 80)

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "")
if IN_KAGGLE and DEVICE == "cpu":
    raise RuntimeError("No GPU: sidebar -> Session options -> Accelerator -> GPU T4, then Save Version again.")
F.ensure_model()   # download MediaPipe's face model once, before worker processes start

# %% [markdown]
# ## 2 · Data: attached datasets + DiffusionFace (streamed)

# %%
roots = {
    "140k": ROOT_OVERRIDES.get("140k") or FD.find_root(INPUT_ROOT, "real-vs-fake"),
    "sfhq": ROOT_OVERRIDES.get("sfhq") or FD.find_root(INPUT_ROOT, "synthetic-faces-high-quality"),
    "ffpp": ROOT_OVERRIDES.get("ffpp") or FD.find_root(INPUT_ROOT, "faceforencispp"),
}
print("attached:", roots)
if roots["140k"] is None:
    raise FileNotFoundError("140k Real and Fake Faces not found: Add Input -> Datasets -> "
                            "'140k Real and Fake Faces' (xhlulu/140k-real-and-fake-faces)")
parts = [FD.scan_140k(roots["140k"])]
if roots["sfhq"]:
    parts.append(FD.scan_sfhq(roots["sfhq"]))
if roots["ffpp"]:
    parts.append(FD.scan_ffpp(roots["ffpp"]))
if DIFFFACE_METHODS:
    print("DiffusionFace (streaming the start of each archive):")
    parts.append(FD.fetch_diffface(DIFFFACE_METHODS, RAW_DIR / "diffface", DIFFFACE_N, DIFFFACE_N_REAL,
                                   urls=DIFFFACE_URLS))
raw = pd.concat(parts, ignore_index=True)
manifest = FD.sample(raw, N_TRAIN_PER_CLASS, max(N_TEST_PER_METHOD, DIFFFACE_N_REAL),
                     max_per_group=FFPP_MAX_PER_VIDEO, seed=SEED)
# FF++ / SFHQ / 140k test sets are capped at N_TEST_PER_METHOD; only the DiffusionFace reals may use more
cap = np.where((manifest.source == "diffface") & (manifest.label == 0), DIFFFACE_N_REAL, N_TEST_PER_METHOD)
keep = (manifest.split == "train") | (manifest.groupby(["source", "method", "split"]).cumcount().values < cap)
manifest = manifest[keep].reset_index(drop=True)
print(f"{len(raw):,} images found · {len(manifest):,} sampled")
display(pd.crosstab([manifest.source, manifest.method], manifest.split))

# DiffusionFace archives are streamed from the start: show which sub-folders the sampled images came from
dfx = manifest[manifest.source == "diffface"]
if len(dfx):
    base = RAW_DIR / "diffface"
    sub = [str(Path(p).relative_to(base).parent.relative_to(Path(p).relative_to(base).parts[0]))
           for p in dfx.path]
    print("DiffusionFace sampled files by folder inside each archive "
          "(check the start of an archive is not one narrow subset):")
    display(pd.crosstab(dfx.method.values, np.array(sub)).iloc[:, :12])

# %% [markdown]
# ## 3 · Canonical face crops
# The same pipeline for every image: face mesh → square crop → 256 px → JPEG q95. Images where no face is found are dropped
# (the table shows how often that happens per source; a large difference between real and fake is itself worth reporting).

# %%
manifest["crop"] = [str(CROP_DIR / (hashlib.md5(p.encode()).hexdigest()[:20] + ".jpg")) for p in manifest.path]

def crop_fn(ps):
    crops = manifest.set_index("path").loc[ps, "crop"].tolist()
    with Pool(N_CPU) as pool:
        rows = list(tqdm(pool.imap(W.crop_face_worker, [(p, c, CROP_SIZE, CROP_MARGIN, CROP_JPEG_Q)
                                                        for p, c in zip(ps, crops)], chunksize=16),
                         total=len(ps), desc="crops", leave=False))
    out = pd.DataFrame(rows)
    for c in ("crop_reason", "src_fmt"):
        out[c] = out[c].astype("string")
    return out

t0 = time.time()
meta = chunked(manifest.path.tolist(), "crop", crop_fn, "parquet", FEAT_DIR, CHUNK)
manifest = pd.concat([manifest, meta.reset_index(drop=True)], axis=1)
manifest["crop_ok"] = manifest.crop_ok & manifest.crop.map(os.path.exists)
print(f"crops: {time.time() - t0:.0f}s")
det_rate = manifest.groupby(["source", "method"]).crop_ok.mean().rename("face_found").to_frame()
det_rate["n"] = manifest.groupby(["source", "method"]).size()
display(det_rate.round(3))

df = manifest[manifest.crop_ok].reset_index(drop=True)
tr_idx, te_idx = np.flatnonzero(df.split == "train"), np.flatnonzero(df.split == "test")
df_te = df.iloc[te_idx].reset_index(drop=True)
SETS = FD.test_sets(df_te)
print(f"train {len(tr_idx):,} · test {len(te_idx):,} · test sets: {len(SETS)}")
display(pd.DataFrame({k: {"family": v["family"], "seen": v["seen"], "n_real": len(v["real"]),
                          "n_fake": len(v["fake"])} for k, v in SETS.items()}).T)

# %% [markdown]
# ## 4 · Shortcut audit
# How easily can real and fake be told apart *within each test set* by things that are not fakeness?
# (a) metadata of the original files (format, size, JPEG quality, face size); (b) low-level statistics of the canonical
# crops (brightness, contrast, saturation, sharpness, high-frequency energy). 5-fold cross-validated AUC within the set.
# High values do not invalidate a result, but they mean that set's real and fake images also differ in other ways.

# %%
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score

def stats_fn(ps):
    with Pool(N_CPU) as pool:
        return pd.DataFrame(list(tqdm(pool.imap(W.image_stats_worker, ps, chunksize=32), total=len(ps),
                                      desc="stats", leave=False)))
stats = chunked(df.crop.tolist(), "stats", stats_fn, "parquet", FEAT_DIR, CHUNK)
meta_cols = pd.DataFrame({"is_jpeg": (df.src_fmt == "JPEG").astype(float), "jpeg_q": df.src_jpeg_q,
                          "w": df.src_w, "h": df.src_h, "face_px": df.src_face_px})

def within_set_auc(X):
    out = {}
    for name, s in SETS.items():
        idx = te_idx[np.r_[s["real"], s["fake"]]]
        y = df.label.values[idx]
        Xs = pd.DataFrame(X[idx])
        Xs = Xs.loc[:, Xs.nunique(dropna=True) > 1]          # drop empty / constant columns (e.g. JPEG quality of PNGs)
        if Xs.shape[1] == 0:
            out[name] = 0.5                                   # nothing to go on: no shortcut
            continue
        p = cross_val_predict(HistGradientBoostingClassifier(max_iter=100, random_state=SEED), Xs.values, y,
                              cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]
        out[name] = roc_auc_score(y, p)
    return pd.Series(out)

audit = pd.DataFrame({"metadata (original files)": within_set_auc(meta_cols.values),
                      "low-level stats (crops)": within_set_auc(stats[W.IMAGE_STAT_NAMES].values)})
display(audit.round(3))

# %% [markdown]
# ## 5 · Features: CLIP (baseline) and face physics
# Clean crops for everything; a JPEG-augmented copy (quality 50–95, fixed per image) of every training crop;
# JPEG q75 / q50 copies of every test crop.

# %%
crop_paths = df.crop.tolist()
tr_paths, te_paths = [crop_paths[i] for i in tr_idx], [crop_paths[i] for i in te_idx]

if SMOKE:   # no CLIP download in a dry run: deterministic random "embeddings"
    def clip_fn(perturb):
        return lambda ps: np.stack([np.random.default_rng(I.stable_seed(p + str(perturb))).normal(size=768)
                                    for p in ps]).astype(np.float16)
else:
    from physfeat import clipfeat as C
    clip_model = C.load_clip(CLIP_MODEL, DEVICE)
    def clip_fn(perturb):
        return lambda ps: C.embed(clip_model, ps, DEVICE, batch_size=128 if DEVICE == "cuda" else 16,
                                  num_workers=N_CPU, normalise_format=False, perturb=perturb)[0]

def phys_fn(perturb):
    def fn(ps):
        with Pool(N_CPU) as pool:
            rows = list(tqdm(pool.imap(W.face_features_worker, [(p, perturb) for p in ps], chunksize=16),
                             total=len(ps), desc=f"face physics/{perturb}", leave=False))
        out = pd.DataFrame(rows).reindex(columns=F.FEATURE_NAMES)
        return out.astype(float)
    return fn

t0 = time.time()
clip_clean = chunked(crop_paths, "clip_clean", clip_fn(None), "npy", FEAT_DIR, CHUNK).astype(np.float32)
clip_aug = chunked(tr_paths, "clip_jpegrand", clip_fn("jpegrand"), "npy", FEAT_DIR, CHUNK).astype(np.float32)
clip_pert = {p: chunked(te_paths, f"clip_{p}", clip_fn(p), "npy", FEAT_DIR, CHUNK).astype(np.float32)
             for p in PERTURBATIONS}
print(f"CLIP: {(time.time() - t0) / 60:.1f} min")
if not SMOKE:
    del clip_model
    torch.cuda.empty_cache() if DEVICE == "cuda" else None

t0 = time.time()
phys_clean = chunked(crop_paths, "face_clean", phys_fn(None), "parquet", FEAT_DIR, CHUNK)
phys_aug = chunked(tr_paths, "face_jpegrand", phys_fn("jpegrand"), "parquet", FEAT_DIR, CHUNK)
phys_pert = {p: chunked(te_paths, f"face_{p}", phys_fn(p), "parquet", FEAT_DIR, CHUNK) for p in PERTURBATIONS}
print(f"face physics: {(time.time() - t0) / 60:.1f} min")
cov = phys_clean.notna().groupby(df.source.values + ":" + df.method.values).mean()
print("Share of crops where each feature group is defined:")
display(pd.DataFrame({"skin lighting": cov["face_r2_sh1"], "left/right": cov["face_lr_light_deg"],
                      "eyes (highlight IoU)": cov["face_eye_hl_iou"], "pupil": cov["face_pupil_irreg_mean"]}).round(2))

# %% [markdown]
# ## 6 · Train (real vs StyleGAN, clean + JPEG-augmented)

# %%
q_aug = np.array([I.jpegrand_quality(p) for p in tr_paths], float)
y_tr = df.label.values[tr_idx]
X_clip = np.r_[clip_clean[tr_idx], clip_aug]
X_phys = pd.concat([phys_clean.iloc[tr_idx].reset_index(drop=True), phys_aug.reset_index(drop=True)], ignore_index=True)
y2, q2 = np.r_[y_tr, y_tr], np.r_[np.full(len(tr_idx), float(CROP_JPEG_Q)), q_aug]
groups = np.r_[np.arange(len(tr_idx)), np.arange(len(tr_idx))]
# physics columns the model can use: defined for >= 1% of training crops and not constant
PHYS_COLS = [c for c in F.FEATURE_NAMES if X_phys[c].notna().mean() >= 0.01 and X_phys[c].nunique() > 1]
dropped = sorted(set(F.FEATURE_NAMES) - set(PHYS_COLS))
print(f"physics features used: {len(PHYS_COLS)}/{len(F.FEATURE_NAMES)}" + (f" · dropped: {dropped}" if dropped else ""))
t0 = time.time()
det = M.RobustDetector(PHYS_COLS, seed=SEED).fit(X_clip, X_phys, y2, q2, groups)
print(f"trained in {(time.time() - t0) / 60:.1f} min")
N = M.RobustDetector.NAMES
MODELS = [N["clip"], N["phys"], N["fused"], N["fused_q"]]
PAIRS = [(N["fused_q"], N["clip"]), (N["fused"], N["clip"])]

# %% [markdown]
# ## 7 · Evaluate on unseen test sets

# %%
Q_TEST = {"clean": float(CROP_JPEG_Q), "jpeg75": 75.0, "jpeg50": 50.0}
conds = {"clean": (clip_clean[te_idx], phys_clean.iloc[te_idx].reset_index(drop=True))}
conds.update({p: (clip_pert[p], phys_pert[p].reset_index(drop=True)) for p in PERTURBATIONS})
all_res, boots, all_scores = [], {}, {}
for cond, (c, ph) in conds.items():
    s = det.predict(c, ph, np.full(len(c), Q_TEST[cond]))
    all_scores[cond] = s
    all_res.append(P.evaluate_sets(s, SETS).assign(condition=cond))
    t0 = time.time()
    boots[cond] = P.bootstrap_sets(s, SETS, PAIRS, n_boot=N_BOOT, seed=SEED)
    print(f"{cond}: bootstrap {time.time() - t0:.0f}s")
res = pd.concat(all_res, ignore_index=True)

print("AUC per test set (clean)")
display(P.summary_sets(res[res.condition == "clean"], models=MODELS).round(4))
print("Mean AUC over unseen test sets, by test-time JPEG")
display(pd.DataFrame({c: {m: boots[c][0][m][0] for m in MODELS} for c in conds}).round(4))

# headroom check (PLAN.md phase 1): is there anything left for physics to add?
clip_fam = P.summary_sets(res[res.condition == "clean"], models=MODELS)[N["clip"]]
clip_fam = clip_fam[clip_fam.index.str.startswith("MEAN")]
print("\nCLIP baseline, clean, by family:\n" + clip_fam.round(4).to_string())
if (clip_fam > 0.95).all():
    print("-> CLIP is above 0.95 on every unseen family: little headroom; look at JPEG conditions and per-set results.")
else:
    print("-> CLIP leaves headroom on: " + ", ".join(clip_fam.index[clip_fam <= 0.95].str.replace("MEAN ", "")))

# %% [markdown]
# ### Pre-registered test: fused − CLIP (mean AUC over unseen sets, paired bootstrap 95% CI)

# %%
rows = []
for cond, (_, diffs) in boots.items():
    for name, (d, lo, hi, p) in diffs.items():
        rows.append({"condition": cond, "comparison": name, "diff": d, "ci_lo": lo, "ci_hi": hi,
                     "P(diff<=0)": p, "CI excludes 0": lo > 0 or hi < 0})
diff_tab = pd.DataFrame(rows)
display(diff_tab.round(4))
key = diff_tab[diff_tab.comparison.str.startswith(N["fused_q"])].set_index("condition")
for cond, r in key.iterrows():
    verdict = ("PHYSICS HELPS" if r.ci_lo > 0 else "PHYSICS HURTS" if r.ci_hi < 0 else "no measurable difference")
    print(f"{cond:7s}: fused(q-aware) − CLIP = {r['diff']:+.4f} [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}]  ->  {verdict}")

print("\nPer family (clean): fused(q-aware) − CLIP")
fam = P.summary_sets(res[res.condition == "clean"], models=MODELS)
display((fam[N["fused_q"]] - fam[N["clip"]]).round(4).to_frame("diff"))

# %% [markdown]
# ## 8 · Which physics features carry signal?
# Each feature's direction is fixed on the *training* data (does it go up or down for StyleGAN fakes?), then its AUC is
# measured on every unseen set in that direction. 0.5 = no signal; below 0.5 = the direction flipped on that set.

# %%
Xtr = phys_clean.iloc[tr_idx]
sign = {}
for f in F.FEATURE_NAMES:
    ok = Xtr[f].notna().values
    if ok.sum() > 50 and Xtr[f][ok].nunique() > 1:
        sign[f] = 1.0 if roc_auc_score(y_tr[ok], Xtr[f].values[ok]) >= 0.5 else -1.0
feat_rows = {}
Xte = phys_clean.iloc[te_idx].reset_index(drop=True)
for name, s in SETS.items():
    idx = np.r_[s["real"], s["fake"]]
    y = np.r_[np.zeros(len(s["real"])), np.ones(len(s["fake"]))]
    r = {}
    for f, sg in sign.items():
        v = Xte[f].values[idx]
        ok = np.isfinite(v)
        r[f] = roc_auc_score(y[ok], sg * v[ok]) if ok.sum() > 20 and len(np.unique(y[ok])) == 2 else np.nan
    feat_rows[name] = r
feat_auc = pd.DataFrame(feat_rows).T
feat_auc.insert(0, "family", [SETS[k]["family"] for k in feat_auc.index])
fam_feat = feat_auc.groupby("family").mean(numeric_only=True).T
display(fam_feat.round(3))

# %%
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(17, 5.5), gridspec_kw={"width_ratios": [1.6, 1]})
ax = axes[0]
x = np.arange(len(conds)); w = 0.2
for i, m in enumerate(MODELS):
    pts = np.array([boots[c][0][m] for c in conds])
    bars = ax.bar(x + (i - 1.5) * w, pts[:, 0], w, yerr=[pts[:, 0] - pts[:, 1], pts[:, 2] - pts[:, 0]], capsize=3, label=m)
    ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
ax.set_xticks(x, list(conds)); ax.set_ylim(0.4, 1.02); ax.grid(axis="y", alpha=.3); ax.legend(fontsize=8, loc="lower left")
ax.set_title("Mean AUC over unseen face test sets (95% bootstrap CI)")
ax = axes[1]
im = ax.imshow(fam_feat.values, cmap="RdBu_r", vmin=0.3, vmax=0.7, aspect="auto")
ax.set_yticks(range(len(fam_feat)), fam_feat.index, fontsize=7); ax.set_xticks(range(fam_feat.shape[1]), fam_feat.columns, rotation=30, ha="right", fontsize=8)
ax.set_title("Physics feature AUC by family (direction fixed on training data)"); fig.colorbar(im, ax=ax, fraction=0.04)
fig.tight_layout(); fig.savefig(OUT_DIR / "face_results.png", dpi=120); plt.show()

# %%
# Evidence panels: one real and one fake crop per family, as the demo will show them
picks = []
for fam_name in sorted({v["family"] for v in SETS.values()}):
    name = next(k for k, v in SETS.items() if v["family"] == fam_name)
    picks += [(f"{name} real", df_te.crop[SETS[name]["real"][0]]), (f"{name} fake", df_te.crop[SETS[name]["fake"][0]])]
fig, axes = plt.subplots(len(picks), 1, figsize=(12, 3.0 * len(picks)))
axes = np.atleast_1d(axes)
for ax, (title, path) in zip(axes, picks):
    f, dbg = F.extract(I.load_rgb(path, False), return_debug=True)
    ax.axis("off")
    if dbg is None:
        ax.set_title(f"{title}: no face found", fontsize=9); continue
    ax.imshow(F.draw_evidence(dbg))
    ax.set_title(f"{title} | one-light R²={f['face_r2_sh1']:.2f}  L/R light angle={f['face_lr_light_deg']:.0f}°  "
                 f"eye-highlight IoU={f['face_eye_hl_iou']:.2f}  pupil irregularity={f['face_pupil_irreg_mean']:.2f}", fontsize=8)
fig.tight_layout(); fig.savefig(OUT_DIR / "face_evidence.png", dpi=100); plt.show()

# %% [markdown]
# ## 9 · Save (the demo loads `models/face_detector.joblib`)

# %%
import joblib, shutil
joblib.dump({"detector": det, "phys_cols": PHYS_COLS, "clip_model": CLIP_MODEL,
             "crop": {"size": CROP_SIZE, "margin": CROP_MARGIN, "jpeg_q": CROP_JPEG_Q},
             "train_data": "140k Real and Fake Faces (FFHQ vs StyleGAN)", "physfeat_commit": PKG_COMMIT},
            MODEL_DIR / "face_detector.joblib")
res.to_csv(OUT_DIR / "results_faces.csv", index=False)
df[["path", "source", "method", "label", "split", "group", "crop"]].to_csv(OUT_DIR / "manifest_faces.csv", index=False)
diff_tab.to_csv(OUT_DIR / "bootstrap_diffs_faces.csv", index=False)
feat_auc.to_csv(OUT_DIR / "feature_auc_faces.csv")
audit.to_csv(OUT_DIR / "shortcut_audit_faces.csv")
ex_dir = OUT_DIR / "demo_examples"; ex_dir.mkdir(exist_ok=True)
for name, s in SETS.items():   # a few crops per test set for the demo's example gallery
    for kind in ("real", "fake"):
        for j, i in enumerate(s[kind][:2]):
            shutil.copy(df_te.crop[i], ex_dir / f"{name.replace(':', '__')}__{kind}{j}.jpg")
out = {
    "physfeat_commit": PKG_COMMIT, "n_train": int(len(tr_idx)), "n_test": int(len(te_idx)),
    "test_sets": {k: {"family": v["family"], "seen": v["seen"], "n_real": len(v["real"]), "n_fake": len(v["fake"])}
                  for k, v in SETS.items()},
    "face_detection_rate": {f"{s}:{m}": round(float(v), 4) for (s, m), v in det_rate.face_found.items()},
    "clip_C": det.C,
    "stack_quality_aware_coef[clip,phys,t,t*clip,t*phys]": det.stack_q.coef_[0].round(4).tolist(),
    "auc_clean": P.summary_sets(res[res.condition == "clean"], models=MODELS).round(4).to_dict(),
    "bootstrap_mean_auc_unseen": {c: {k: np.round(v, 4).tolist() for k, v in b[0].items()} for c, b in boots.items()},
    "bootstrap_diffs": {c: {k: np.round(v, 4).tolist() for k, v in b[1].items()} for c, b in boots.items()},
    "feature_auc_by_family": fam_feat.round(4).to_dict(),
    "shortcut_audit": audit.round(4).to_dict(),
}
(OUT_DIR / "results_faces.json").write_text(json.dumps(out, indent=2, default=str))
print(json.dumps({"bootstrap_diffs": out["bootstrap_diffs"]}, indent=1))
