# Physics-Based AI-Generated Image Detection — Results

CV course project · results of notebooks 01–04 (Kaggle, Tesla T4)

## TL;DR

| Question | Answer |
|---|---|
| Do physics features (perspective geometry + lighting consistency) improve detection of images from **unseen generators**? | **No.** On clean images the fused model gains +0.0002 AUC over CLIP alone. |
| Are physics features **robust to compression**? | **Yes.** JPEG q50 costs them 0.008 AUC, against 0.186 for CLIP. |
| Does that robustness make the **fused detector** better under compression? | **No.** Training CLIP with JPEG augmentation recovers +0.171 AUC at q50. After that, physics adds +0.0002 (95% CI includes 0). |
| Do the lighting features **measure lighting** on real photos? | **Yes, weakly.** They respond to introduced lighting violations beyond what editing seams cause (AUC ≈ 0.56–0.57 vs a seam-only control, p < 0.005). CLIP does not respond to lighting specifically. |

**Bottom line:** a rigorous negative result for fully generated images on GenImage, with one positive controlled finding. The lighting-consistency features capture a real physical signal that CLIP lacks, but monocular surface-normal estimation is too imprecise for it to help detect generated images.

---

## Setup (common to all notebooks)

- **Dataset:** Unbiased Tiny GenImage (Kaggle `cartografia/unbiased-tiny-genimage`). The fakes are re-encoded as JPEG q96 to remove the known PNG-vs-JPEG shortcut.
- **Protocol (cross-generator):** train on real vs **one** generator (SD 1.5); test on held-out validation images from **every** generator. *Unseen* = the six generators not used in training. The headline metric is **mean AUC over unseen generators**.
- **Baseline:** CLIP ViT-L/14 image embeddings with a logistic-regression probe (UnivFD-style). Regularisation C = 16, chosen by cross-validation.
- **Physics branch:**
  - *Geometry* (18 features): LSD line segments → RANSAC vanishing points → convergence residuals, VP splitting, camera-consistency checks.
  - *Lighting* (16 features): Marigold surface normals (512 px, ≈0.66 s/image on a T4) → first/second-order spherical-harmonic shading fit; agreement of local light directions over a 4×4 grid; leave-one-region-out shading prediction; specular-highlight alignment; integrability of the normal field.
  - Each branch is fed to a gradient-boosted classifier.
- **Fusion:** logistic stacking on out-of-fold logits of the CLIP and physics models.

---

## Notebook 01 — Data, bias audit, geometry + CLIP features

### Dataset after scanning, de-duplication and splitting

| Source | Train | Val | Total |
|---|---:|---:|---:|
| real | 4685 | 1130 | 5815 |
| adm | 1973 | 527 | 2500 |
| biggan | 1980 | 520 | 2500 |
| glide | 1998 | 502 | 2500 |
| midjourney | 1988 | 512 | 2500 |
| sd15 | 2056 | 444 | 2500 |
| vqdm | 2022 | 478 | 2500 |
| wukong | 1979 | 521 | 2500 |
| **All** | 18681 | 4634 | **23,315** |

13 exact-duplicate files were removed, and no files were unreadable.

### Bias audit

A classifier trained on **file metadata only** (format, JPEG quality, width, height, aspect, bits per pixel) reaches **AUC 0.667** on unseen generators. This is above chance (0.5), so a small residual shortcut exists. It is far below the detectors, but should be stated as a limitation.

### Preview: CLIP vs geometry (AUC, trained on SD 1.5)

| Generator | CLIP (baseline) | Physics: geometry | Fused |
|---|---:|---:|---:|
| SD 1.5 (seen) | 0.997 | 0.698 | 0.997 |
| ADM | 0.914 | 0.626 | 0.914 |
| BigGAN | 0.941 | 0.605 | 0.941 |
| Glide | 0.978 | 0.672 | 0.978 |
| Midjourney | 0.979 | 0.679 | 0.979 |
| VQDM | 0.945 | 0.592 | 0.945 |
| Wukong | 0.993 | 0.725 | 0.993 |
| **Mean (unseen)** | 0.959 | 0.650 | 0.959 |

**Findings**
- CLIP alone is already strong across generators.
- Geometry features carry weak signal (0.65) that CLIP already covers: the fused model equals CLIP.
- 94% of images report ≥ 2 vanishing points. On ImageNet-style content (animals, objects), these "vanishing points" are mostly fits to texture edges, not real perspective, which explains why geometry adds nothing here.

*(Notebook 02 re-tunes the CLIP probe, so its CLIP numbers are slightly higher than this preview.)*

---

## Notebook 02 — Lighting physics, robustness, full evaluation

### Clean test images — AUC per generator

| Generator | CLIP | Phys: geometry | Phys: lighting | Phys: geo+light | Fused: CLIP+geo+light |
|---|---:|---:|---:|---:|---:|
| SD 1.5 (seen) | 0.9988 | 0.6944 | 0.7235 | 0.7976 | 0.9988 |
| ADM | 0.9273 | 0.6279 | 0.6071 | 0.6360 | 0.9283 |
| BigGAN | 0.9710 | 0.6104 | 0.5770 | 0.6111 | 0.9704 |
| Glide | 0.9853 | 0.6805 | 0.6237 | 0.6847 | 0.9856 |
| Midjourney | 0.9788 | 0.6822 | 0.6423 | 0.7164 | 0.9797 |
| VQDM | 0.9719 | 0.6023 | 0.6126 | 0.6267 | 0.9714 |
| Wukong | 0.9958 | 0.7180 | 0.7310 | 0.8105 | 0.9962 |
| **Mean (unseen)** | 0.9717 | 0.6536 | 0.6323 | 0.6809 | 0.9719 |

### Ablation — mean over unseen generators (clean)

| Model | Mean AUC (unseen) | Mean balanced acc. (unseen) |
|---|---:|---:|
| CLIP (baseline) | 0.9717 | 0.8663 |
| Physics: geometry | 0.6536 | 0.5964 |
| Physics: lighting | 0.6323 | 0.5916 |
| Physics: geometry+lighting | 0.6809 | 0.6106 |
| Fused: CLIP+geometry | 0.9719 | 0.8670 |
| Fused: CLIP+lighting | 0.9718 | 0.8642 |
| Fused: CLIP+geometry+lighting | 0.9719 | 0.8664 |

**Learned stacking weights** (how much the fusion trusts each branch):

| Fusion | Weight on CLIP | Weight on physics |
|---|---:|---:|
| CLIP + geometry | 1.681 | 0.135 |
| CLIP + lighting | 1.688 | 0.249 |
| CLIP + geometry+lighting | 1.668 | 0.216 |

### Robustness — mean AUC over unseen generators (models trained on clean images)

| Model | Clean | JPEG q75 | JPEG q50 | 0.5× downscale | Drop clean→q50 |
|---|---:|---:|---:|---:|---:|
| CLIP (baseline) | 0.972 | 0.879 | 0.786 | 0.948 | -0.186 |
| Physics: geometry+lighting | 0.681 | 0.679 | 0.672 | 0.660 | -0.008 |
| Fused: CLIP+geometry+lighting | 0.972 | 0.882 | 0.791 | 0.948 | -0.181 |

Mean balanced accuracy over unseen generators:

| Model | Clean | JPEG q75 | JPEG q50 | 0.5× downscale |
|---|---:|---:|---:|---:|
| CLIP (baseline) | 0.866 | 0.782 | 0.684 | 0.842 |
| Physics: geometry+lighting | 0.611 | 0.614 | 0.613 | 0.608 |
| Fused: CLIP+geometry+lighting | 0.866 | 0.787 | 0.689 | 0.842 |

### Per-generator AUC at JPEG q50

| Generator | CLIP | Phys: geometry | Phys: geo+light | Fused |
|---|---:|---:|---:|---:|
| SD 1.5 (seen) | 0.978 | 0.702 | 0.789 | 0.979 |
| ADM | 0.626 | 0.647 | 0.630 | 0.634 |
| BigGAN | 0.722 | 0.650 | 0.624 | 0.726 |
| Glide | 0.755 | 0.670 | 0.643 | 0.760 |
| Midjourney | 0.945 | 0.683 | 0.710 | 0.948 |
| VQDM | 0.710 | 0.587 | 0.610 | 0.715 |
| Wukong | 0.956 | 0.715 | 0.818 | 0.960 |
| **Mean (unseen)** | 0.786 | 0.659 | 0.672 | 0.791 |

**Findings**
- On clean images, physics adds nothing (fused − CLIP = +0.0002).
- Lighting adds a little to geometry (physics-only 0.654 → 0.681).
- **Physics is nearly compression-invariant** (−0.008 at q50), while CLIP drops by 0.186.
- The clean-trained fusion keeps trusting CLIP about 8× more than physics, so it does not exploit physics' robustness (+0.005 at q50).
- At q50, geometry alone beats CLIP on ADM (0.647 vs 0.626).

---

## Notebook 03 — Compression-aware training (fair baseline)

**Design**
- Every training image also gets a JPEG-compressed copy (quality drawn from 50–95, fixed per image). All branches are trained on clean + compressed copies, and that includes the CLIP baseline.
- The **quality-aware fusion** additionally receives the JPEG quality and its interactions with each branch.
- Test sets are identical to notebook 02. Resizing is **never** seen in training.
- Uncertainty comes from a paired bootstrap over test images (500 resamples).

### Mean AUC over unseen generators, with 95% bootstrap CI

| Model | Clean | JPEG q75 | JPEG q50 | 0.5× downscale |
|---|---:|---:|---:|---:|
| CLIP (clean-trained) | 0.972 [0.967, 0.976] | 0.879 [0.870, 0.890] | 0.786 [0.773, 0.799] | 0.948 [0.941, 0.954] |
| Fused (clean-trained) | 0.972 [0.968, 0.976] | 0.882 [0.873, 0.893] | 0.791 [0.778, 0.804] | 0.948 [0.942, 0.955] |
| CLIP (JPEG-aug) | 0.970 [0.965, 0.974] | 0.969 [0.965, 0.974] | 0.957 [0.951, 0.962] | 0.954 [0.948, 0.960] |
| Physics (JPEG-aug) | 0.683 [0.664, 0.700] | 0.677 [0.659, 0.693] | 0.675 [0.656, 0.693] | 0.653 [0.635, 0.670] |
| Fused (JPEG-aug) | 0.970 [0.966, 0.974] | 0.970 [0.966, 0.974] | 0.957 [0.951, 0.963] | 0.954 [0.948, 0.960] |
| Fused (JPEG-aug, quality-aware) | 0.970 [0.965, 0.974] | 0.970 [0.966, 0.974] | 0.957 [0.951, 0.963] | 0.954 [0.948, 0.960] |

### Paired differences in mean AUC (95% CI; ✱ = CI excludes 0)

| Comparison | Clean | JPEG q75 | JPEG q50 | 0.5× downscale |
|---|---:|---:|---:|---:|
| Fused (JPEG-aug, quality-aware) − CLIP (JPEG-aug) | +0.0003 [-0.0001, +0.0007] | +0.0003 [-0.0002, +0.0007] | +0.0002 [-0.0003, +0.0006] | +0.0003 [-0.0002, +0.0008] |
| Fused (JPEG-aug) − CLIP (JPEG-aug) | +0.0003 [-0.0001, +0.0007] | +0.0003 [-0.0001, +0.0007] | +0.0002 [-0.0003, +0.0006] | +0.0003 [-0.0002, +0.0008] |
| CLIP (JPEG-aug) − CLIP (clean-trained) | -0.0022 [-0.0041, -0.0003] ✱ | +0.0899 [+0.0825, +0.0964] ✱ | +0.1713 [+0.1616, +0.1807] ✱ | +0.0062 [+0.0038, +0.0084] ✱ |
| Fused (JPEG-aug, quality-aware) − CLIP (clean-trained) | -0.0019 [-0.0037, -0.0001] ✱ | +0.0902 [+0.0828, +0.0967] ✱ | +0.1715 [+0.1617, +0.1811] ✱ | +0.0065 [+0.0040, +0.0086] ✱ |

### Mean balanced accuracy over unseen generators

| Model | Clean | JPEG q75 | JPEG q50 | 0.5× downscale |
|---|---:|---:|---:|---:|
| CLIP (clean-trained) | 0.866 | 0.782 | 0.684 | 0.842 |
| Fused (clean-trained) | 0.866 | 0.787 | 0.689 | 0.842 |
| CLIP (JPEG-aug) | 0.865 | 0.883 | 0.840 | 0.857 |
| Physics (JPEG-aug) | 0.616 | 0.616 | 0.620 | 0.599 |
| Fused (JPEG-aug) | 0.866 | 0.883 | 0.842 | 0.860 |
| Fused (JPEG-aug, quality-aware) | 0.867 | 0.882 | 0.838 | 0.861 |

### What the quality-aware fusion learned

| JPEG quality | Weight on CLIP logit | Weight on physics logit |
|---|---:|---:|
| 96 (none) | 1.357 | 0.197 |
| 75 | 1.372 | 0.204 |
| 50 | 1.389 | 0.213 |

The weights barely move with compression. Once CLIP is trained with compression it no longer degrades, so there is nothing for physics to compensate for.

### Per-generator AUC at JPEG q50

| Generator | CLIP (clean-trained) | CLIP (JPEG-aug) | Physics (JPEG-aug) | Fused (JPEG-aug, q-aware) |
|---|---:|---:|---:|---:|
| SD 1.5 (seen) | 0.978 | 0.997 | 0.800 | 0.997 |
| ADM | 0.626 | 0.899 | 0.625 | 0.900 |
| BigGAN | 0.722 | 0.941 | 0.633 | 0.940 |
| Glide | 0.755 | 0.971 | 0.642 | 0.971 |
| Midjourney | 0.945 | 0.978 | 0.724 | 0.979 |
| VQDM | 0.710 | 0.961 | 0.603 | 0.959 |
| Wukong | 0.956 | 0.993 | 0.820 | 0.993 |
| **Mean (unseen)** | 0.786 | 0.957 | 0.675 | 0.957 |

**Findings**
- **JPEG augmentation closes CLIP's robustness gap:** +0.171 AUC at q50 [0.162, 0.181]. It also helps on unseen resizing (+0.006), at a tiny cost on clean images (−0.002).
- **Physics adds no measurable value** on top of the augmented CLIP, under any condition (+0.0002 to +0.0003; every CI includes 0).
- The robustness argument for physics features fails once the baseline gets standard augmentation: **stable but weak is still weak**.

---

## Notebook 04 — Controlled lighting violations on real photos

**Design**
- 400 real test photos; the right half of each crop is edited in four ways:
  - `original` (untouched);
  - `shift` (control: the right half is shifted down by 1/3, which creates seams but keeps the lighting direction);
  - `hflip` (violation: the right half is mirrored, reversing the horizontal light direction);
  - `rot180` (violation: the right half is rotated, reversing both the horizontal and vertical light direction).
- All versions get identical cropping and JPEG re-encoding.
- The expected direction of every feature was fixed before the experiment ran.
- A synthetic sanity check with ground-truth normals gave inconsistency-score AUC 0.54 (shift), 0.81 (hflip) and 0.99 (rot180).

### AUC for telling the edited version from the original — all 400 photos

| Feature | Shift (control) | Mirror half (violation) | Rotate half 180° (violation) |
|---|---:|---:|---:|
| `lit_r2_sh1` | 0.561 | 0.531 | 0.608 |
| `lit_cell_ang_med` | 0.524 | 0.530 | 0.580 |
| `lit_cell_ang_p90` | 0.481 | 0.523 | 0.567 |
| `lit_cell_resultant` | 0.467 | 0.500 | 0.550 |
| `lit_loo_r2` | 0.557 | 0.540 | 0.622 |
| `lit_loo_gap` | 0.507 | 0.527 | 0.563 |
| `spec_align` | 0.559 | 0.539 | 0.572 |
| **inconsistency score** | 0.557 | 0.545 | 0.617 |

### Same, on photos with clearly directional lighting (121 photos)

| Feature | Shift (control) | Mirror half (violation) | Rotate half 180° (violation) |
|---|---:|---:|---:|
| `lit_r2_sh1` | 0.570 | 0.569 | 0.615 |
| `lit_cell_ang_med` | 0.562 | 0.608 | 0.588 |
| `lit_cell_ang_p90` | 0.501 | 0.598 | 0.560 |
| `lit_cell_resultant` | 0.482 | 0.524 | 0.559 |
| `lit_loo_r2` | 0.565 | 0.570 | 0.618 |
| `lit_loo_gap` | 0.524 | 0.537 | 0.522 |
| `spec_align` | 0.524 | 0.551 | 0.511 |
| **inconsistency score** | 0.552 | 0.604 | 0.615 |

### Inconsistency score (training-free) — edit vs original

| Photos | Edit vs original | AUC | Moved expected way | Wilcoxon p |
|---|---:|---:|---:|---:|
| All (n=400) | Shift (control) | 0.557 | 63% | < 1e-5 |
| All (n=400) | Mirror half (violation) | 0.545 | 59% | 7e-05 |
| All (n=400) | Rotate half 180° (violation) | 0.617 | 68% | < 1e-5 |
| Directional (n=121) | Shift (control) | 0.552 | 60% | 0.009 |
| Directional (n=121) | Mirror half (violation) | 0.604 | 70% | < 1e-5 |
| Directional (n=121) | Rotate half 180° (violation) | 0.615 | 70% | 1e-05 |

### Sharpest test — violation vs seam-only control (same photos)

| Photos | Comparison | AUC | Moved expected way | Wilcoxon p |
|---|---:|---:|---:|---:|
| All (n=400) | Mirror half (violation) vs control | 0.491 | 50% | 0.508 |
| All (n=400) | Rotate half 180° (violation) vs control | 0.567 | 57% | 5e-05 |
| Directional (n=121) | Mirror half (violation) vs control | 0.559 | 59% | 0.004 |
| Directional (n=121) | Rotate half 180° (violation) vs control | 0.570 | 57% | 0.005 |

### Do the trained detectors (notebook 02) react to these edits?

All inputs are **real** photos. An AUC below 0.5 means the edited versions were rated *more real* than the originals.

| Detector (notebook 02) | Mean P(fake): original | Mean P(fake): shift | Mean P(fake): hflip | Mean P(fake): rot180 | AUC vs original: shift | AUC vs original: hflip | AUC vs original: rot180 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CLIP | 0.144 | 0.090 | 0.085 | 0.066 | 0.409 | 0.371 | 0.320 |
| Physics lighting | 0.324 | 0.313 | 0.326 | 0.307 | 0.478 | 0.507 | 0.476 |
| Fused | 0.116 | 0.066 | 0.065 | 0.049 | 0.408 | 0.371 | 0.320 |

**Findings**
- **The lighting features respond to lighting violations beyond editing seams,** but weakly: AUC 0.56–0.57 against the control, p < 0.005 (rotation on all photos; both edits on directionally lit photos).
- **Seams alone also move the features** (control AUC ≈ 0.56). Part of their response is to any break in shading continuity.
- **The mirror edit has no effect overall but a clear effect on directionally lit photos,** which is what physics predicts. Natural light is mostly overhead, and a left↔right mirror leaves the vertical light component unchanged; a 180° rotation flips it.
- **Synthetic vs real gap:** 0.81–0.99 with ground-truth normals against ≈ 0.57 with Marigold normals. The precision of monocular normal estimation is the bottleneck.
- **CLIP does not respond to lighting specifically.** Its P(fake) drops for *every* edit, including the control, so it reacts to editing in general, not to physics.
- **The trained physics detector does not respond either** (AUC ≈ 0.48–0.51). Its 0.63–0.68 AUC on GenImage therefore comes from other feature properties (e.g. light strength, normal statistics), not from detecting lighting violations.

---

## Overall conclusions

1. **Main result (negative):** monocular-physics features (perspective geometry + lighting consistency) do not improve cross-generator detection of fully generated images on GenImage, clean or compressed, once the baseline is trained fairly.
2. **Robustness is not usefulness:** physics features are compression-invariant, but simple JPEG augmentation gives CLIP the same robustness, with far higher discriminative power (0.957 vs 0.675 AUC at q50).
3. **Positive controlled finding:** the lighting features measure a real physical property (lighting consistency) that CLIP does not, but their sensitivity is limited by monocular normal estimation.
4. **Implication / future work:** physics cues may matter more for **partially manipulated real photos** (splicing, relighting, composites) than for fully generated images, and they need more precise geometry (better normal estimators, multi-view input, or explicit object segmentation).

## Limitations

- One training generator (SD 1.5) and one dataset (ImageNet-style content); no modern generators (SDXL, Flux) in the test set.
- Residual metadata shortcut (metadata-only AUC 0.667).
- Physics features depend on a single normal estimator (Marigold at 512 px, default steps).
- The controlled violations are synthetic edits of real photos, not real-world composites.
- Geometry features are uninformative on scenes without man-made structure, which is most of this dataset.
