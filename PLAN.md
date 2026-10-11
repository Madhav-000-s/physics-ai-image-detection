# Plan: a face deepfake detector where physics adds accuracy

**Goal:** a working demo of a deepfake image detector in which physics-based features *measurably* improve on a strong data-driven baseline.

## Why the setting changes

Notebooks 01–04 ([RESULTS.md](RESULTS.md)) show that on GenImage-style scenes, physics features add essentially nothing to a CLIP ViT-L/14 probe (+0.0002 AUC). The literature agrees: no 2023–2026 paper we found shows lighting or geometry features adding AUC over a CLIP probe on fully generated scenes. Notebook 04 named the bottleneck: monocular surface normals are too imprecise.

Faces remove that bottleneck:

- **Precise geometry for free.** MediaPipe Face Landmarker regresses a 478-point 3D mesh, including the irises, in milliseconds on a CPU. Normals come from a face model, not a per-pixel guess.
- **Published evidence that physics helps on faces:**
  - Corneal highlights that disagree between the two eyes: AUC 0.94 on StyleGAN2 faces ([Hu, Li, Lyu 2021](https://ar5iv.labs.arxiv.org/html/2009.11924)).
  - Irregular pupils: AUC 0.91 ([Guo et al. 2022](https://ar5iv.labs.arxiv.org/html/2109.00162)).
  - 3D-face-model lighting deviates for diffusion faces ([Boháček & Farid, CVPRW 2023](https://openaccess.thecvf.com/content/CVPR2023W/WMF/html/Bohacek_A_Geometric_and_Photometric_Exploration_of_GAN_and_Diffusion_Synthesized_CVPRW_2023_paper.html)).
  - A specular-reflection branch added +12–18 AUC to an image detector on unseen datasets ([SRI-Net, ICLR 2026](https://arxiv.org/html/2602.06452v1)). Caveat: the baseline was Xception, not CLIP.

This gives the project its storyline: **physics fails when the geometry is guessed (01–04) and helps when the geometry is precise (05+).**

## Detector

```
image → MediaPipe face mesh ─┬─ face crop → CLIP ViT-L/14 → linear probe        (baseline, as in 01–03)
                             └─ physfeat.face features → gradient boosting       (physics branch)
                                              ↓
                         stacked fusion (models.RobustDetector, JPEG-augmented, as in 03)
```

Physics branch: `physfeat/face.py`, already implemented and tested on synthetic renders.

| Group | Features | What breaks them |
|---|---|---|
| Skin lighting | `face_r2_sh1/sh2`, `face_light_strength/elev`, `face_lr_light_deg`, `face_lr_cross_r2`, `face_lr_gap`, `face_spec_align` | Shading painted inconsistently; an inner face lit differently from the head |
| Eyes | `face_eye_hl_iou`, `face_eye_hl_offset`, `face_eye_hl_area_ratio`, `face_eye_n_hl`, `face_pupil_irreg_mean/max`, `face_iris_px` | GAN eyes reflecting different scenes; irregular pupils |
| Pose | `face_pose_inner_outer_deg`, `face_yaw_deg`, `face_pitch_deg` | Face swaps (inner face pasted onto another head) |

Optional later addition: **face vs. background lighting.** Use Marigold normals on the background only. This is where notebook 04's positive finding applies directly.

## Phases

| Phase | Work | Go / no-go | Status |
|---|---|---|---|
| **0. Setup** | Move `physfeat` from the `%%writefile` cells into a package (`pip install -e .`). Add face-physics module + synthetic tests. Install CUDA PyTorch locally. | Tests pass; `torch.cuda.is_available()` on the RTX 5050 | **Done.** Checked on a real photo: half-to-half light angle 5° → 36° when one half is mirrored; 29 ms per face |
| **1. Data + baseline** | [Notebook 05](05_face_deepfakes.ipynb), sections 2–4: train real vs StyleGAN faces; test on unseen GAN, face-swap, diffusion and inpainting sets. Canonical crops for every image; shortcut audit. | **CLIP must leave headroom**: unseen AUC ≲ 0.95, or a clear JPEG drop. The notebook prints this check. | **Done** (Kaggle run `CV_nb5`): CLIP 0.69 on unseen sets, so there is ample headroom |
| **2. Face physics** | Notebook 05, sections 5 and 8: `physfeat.face` features, how often each is defined per set, and per-feature AUC per family (direction fixed on training data). | ≥ 1 feature with AUC > 0.6 on unseen sets | **Done:** corneal-highlight features 0.61–0.66 on diffusion face swaps; learned physics 0.55 overall |
| **3. Fusion + evaluation** | Notebook 05, sections 6–7: `RobustDetector` with JPEG augmentation; `protocol.bootstrap_sets` (each fake set against its own reals). | **Pre-registered:** fused − CLIP mean unseen AUC, 95% CI excludes 0. The notebook prints the verdict. | **Done:** met but negligible (+0.0016). Notebook 06 (leave one family out, physics prior, +0.01 effect-size bar): no useful gain. See RESULTS.md |
| **4. Demo** | Gradio app (below) | — | **Done:** `python demo/app.py`; physics used as explanation |

## Data

All of these are free to download; licenses must be checked before redistributing anything.

- **Train (seen):** [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces): FFHQ reals vs StyleGAN fakes, 256 px JPEG.
- **Test (unseen).** Each fake set is scored against reals from its own source:
  - **Diffusion, face swap, inpainting:** [DiffusionFace](https://github.com/Rapisurazurite/DiffFace) (Zenodo [part 1](https://zenodo.org/records/10865300) / [part 2](https://zenodo.org/records/10865065), CC BY 4.0, no access form). It has one ~3 GB tar per method, including `Real.tar`. The notebook streams only the first ~800 images of each archive. Methods: ADM, DDIM, LDM, PNDM, SD 1.5 / 2.1 img2img and text2img, MM_CelebA, **DiffSwap**, **Inpaint**.
  - **Face swap / reenactment:** FF++ frames on Kaggle (`adham7elmy/faceforencispp-extracted-frames`): Deepfakes, Face2Face, FaceSwap, FaceShifter, NeuralTextures, plus real frames. Unofficial mirror with unclear licensing; say so in the report. Faces are often small in these frames, so `FaceMesher` searches zoomed windows when the full frame yields no face.
  - **GAN, different content:** SFHQ part 1 (`selfishgene/synthetic-faces-high-quality-sfhq-part-1`). *Correction to earlier research:* this is **StyleGAN2** (paintings and 3D models encoded into StyleGAN2), not diffusion. It has no reals, so it uses the 140k test reals.
  - Not used: StyleGAN3 (`troykueh/real-vs-fake-faces-stylegan3`), and [DiFF](https://github.com/xaCheng1996/DiFF), whose reals need a form.
- **Fallback (partial edits):** [TGIF](https://github.com/IDLabMedia/tgif-dataset) (inpainting with masks: SD2, SDXL, Firefly, FLUX; CC BY-SA).

**Shortcut guard:** real and fake images come from different sources with different resolution and compression. Every image goes through the same pipeline: face crop → fixed size → JPEG q96. Re-run the nb01 metadata-only audit on the result and report its AUC.

## Demo (Gradio)

- **Input:** upload an image.
- **Output:**
  - P(fake) from **CLIP**, **physics** and **fused**, side by side.
  - Evidence panel from `face.draw_evidence`: skin mask + midline, observed vs. predicted shading with light arrows (global / left / right), and both irises with the highlight (red) and fitted pupil ellipse (green).
  - Plain-language reasons, e.g. "the two corneal highlights imply different light positions."
- **Speed:** MediaPipe + physics take milliseconds and CLIP about a second, so it runs locally on the RTX 5050 (or CPU) and can be hosted on Hugging Face Spaces.

## Fallback if phase 3 shows no gain

Evaluate on partial manipulations (TGIF inpainting). Comparing one region's lighting against another is what notebook 04 showed works, and a global CLIP embedding dilutes small local edits.

Because the success criterion is fixed in advance, a null result is still reportable, and the demo still shows physics as explanation alongside the score.

## Risks

- **Compression and low resolution:** eye cues need irises of at least about 4 px radius and visible highlights. The features return NaN otherwise, and the model sees `face_iris_px`.
- **Pupil segmentation in RGB** is unreliable for dark-brown eyes (it returns NaN when the contrast is low). Guo et al. used a dedicated segmentation network.
- **Newer generators** have largely fixed the corneal-highlight artifact ([survey](https://arxiv.org/pdf/2202.07145)), so lighting and pose may matter more there.
- **MediaPipe's mesh is a regression with a face prior.** It may smooth over exactly the geometric inconsistencies we want to detect. Measure, don't assume.

## Local setup

Keep the virtual environment **outside OneDrive** (it is several GB). Current one: `C:\Users\super\venvs\physfake`, with torch 2.11 + CUDA 12.8 on the RTX 5050, mediapipe 1.1, gradio 6.30, transformers 5.19 and diffusers 0.41.

```bash
python -m venv C:\Users\super\venvs\physfake
C:\Users\super\venvs\physfake\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # RTX 50-series needs CUDA 12.8+
pip install -e ".[deep,face,demo,test]"
python -m pytest
```

On first use, `physfeat.face` downloads MediaPipe's `face_landmarker.task` (about 4 MB) to `~/.cache/physfeat`.
