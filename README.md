# Physics-Based AI-Generated Image Detection

Computer-vision course project. Do physics cues help detect AI-generated images from generators a model has never seen? The cues tested are **perspective geometry** (vanishing-point consistency) and **lighting consistency** (shading under one light, using Marigold surface normals). They are compared against a CLIP ViT-L/14 linear-probe baseline.

**Full results: [RESULTS.md](RESULTS.md)**

## Notebooks

| Notebook | Contents |
|---|---|
| [01_data_and_features.ipynb](01_data_and_features.ipynb) | Dataset scan, de-duplication, metadata bias audit, geometry features, CLIP embeddings |
| [02_lighting_robustness_eval.ipynb](02_lighting_robustness_eval.ipynb) | Marigold normals → lighting features; cross-generator evaluation; robustness to JPEG and resizing |
| [03_compression_aware.ipynb](03_compression_aware.ipynb) | JPEG-augmented training for every model, quality-aware fusion, paired bootstrap CIs |
| [04_lighting_violations.ipynb](04_lighting_violations.ipynb) | Controlled lighting violations on real photos (shift / mirror / rotate half) |

Each notebook writes its own copy of the `physfeat` package with `%%writefile`, so it runs on Kaggle without extra setup.

## Next phase: face deepfakes

Physics didn't help on generic scenes because the geometry had to be guessed from a single image. The next phase moves to faces, where a 3D face mesh gives precise geometry. See **[PLAN.md](PLAN.md)** for details.

| Path | Contents |
|---|---|
| [06_leave_one_family_out.ipynb](06_leave_one_family_out.ipynb) | Reuses notebook 05's features: leave-one-family-out training, a training-free physics prior, and the all-families model used by the demo |
| [demo/](demo/) | Gradio demo: CLIP detector with physics explanations (see below) |
| [05_face_deepfakes.ipynb](05_face_deepfakes.ipynb) | Faces: train on StyleGAN, test on unseen GAN / face-swap / diffusion / inpainting sets; CLIP vs face physics vs fused, with the pre-registered bootstrap test. Built from [notebooks_src/05_face_deepfakes.py](notebooks_src/05_face_deepfakes.py) with `python tools/py2nb.py` |
| [physfeat/](physfeat/) | The canonical feature package (merged from the notebooks), plus [face.py](physfeat/face.py) (skin lighting, corneal highlights, pupil shape, pose consistency) and [facedata.py](physfeat/facedata.py) (face datasets, DiffusionFace streaming, paired test sets) |
| [tests/](tests/) | Synthetic known-answer tests for the face features, data handling and evaluation |

```bash
pip install -e ".[deep,face,test]"   # on Kaggle: pip install --no-deps git+https://github.com/Madhav-000-s/physics-ai-image-detection
python -m pytest
```

## Running

The notebooks are built for Kaggle (GPU T4, Internet on). Attach the dataset [Unbiased Tiny GenImage](https://www.kaggle.com/datasets/cartografia/unbiased-tiny-genimage) and run them in order. Notebooks 02–04 also take the outputs of the earlier notebooks as inputs. The setup cell of each notebook lists exactly what to attach.

## Summary

- Physics features do **not** improve cross-generator detection over CLIP (+0.0002 AUC).
- They are nearly compression-invariant, but JPEG-augmented CLIP gets the same robustness with much higher accuracy (0.957 vs 0.675 AUC at JPEG q50).
- The lighting features do respond to real lighting violations beyond editing seams (AUC ≈ 0.57, p < 0.005). They are limited by the precision of monocular normal estimation.
- On **faces**, where a 3D face mesh gives precise geometry (notebooks 05–06), physics still adds at most +0.0016 AUC to CLIP across GAN, diffusion, face-swap, reenactment and inpainting fakes. The physically expected directions even reverse between generators. What does improve detection is training on more kinds of fakes (unseen-family AUC 0.69 → 0.75).

## Demo

A Gradio app: upload a photo with a face and get
- P(fake) from **CLIP**, from **physics** and from the **fused** detector, side by side;
- an evidence panel: the skin used, observed vs. single-light shading with light arrows, and both irises (reflection and pupil fit);
- eight **physics checks**, each compared with about 14k real face photos and flagged when fewer than 5% of real faces are as extreme.

The detector is the all-families model from notebook 06: mean AUC 0.86 on held-out images, about 0.75 on a kind of fake it has never seen. Physics did not improve accuracy in our experiments ([RESULTS.md](RESULTS.md)); the demo uses it to explain each verdict.

```bash
pip install -e ".[demo]"
python demo/app.py
```

Then open http://127.0.0.1:7860. The first start downloads CLIP ViT-L/14 (about 1.7 GB, cached by Hugging Face) and MediaPipe's face model (about 4 MB). It runs on a GPU if available, at about 0.15 s per image on an RTX 5050.

The example gallery reads face crops from `demo/examples/`. Those crops come from datasets that may not be redistributed (FFHQ, FF++), so they are not in the repo; notebook 05 writes them to `demo_examples/` in its output.
