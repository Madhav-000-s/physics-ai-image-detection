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

Each notebook writes the shared `physfeat` feature-extraction package with `%%writefile`, so they run independently on Kaggle.

## Running

The notebooks are built for Kaggle (GPU T4, Internet on). Attach the dataset [Unbiased Tiny GenImage](https://www.kaggle.com/datasets/cartografia/unbiased-tiny-genimage) and run them in order. Notebooks 02–04 also take the outputs of the earlier notebooks as inputs. The setup cell of each notebook lists exactly what to attach.

## Summary

- Physics features do **not** improve cross-generator detection over CLIP (+0.0002 AUC).
- They are nearly compression-invariant, but JPEG-augmented CLIP gets the same robustness with much higher accuracy (0.957 vs 0.675 AUC at JPEG q50).
- The lighting features do respond to real lighting violations beyond editing seams (AUC ≈ 0.57, p < 0.005). They are limited by the precision of monocular normal estimation.
