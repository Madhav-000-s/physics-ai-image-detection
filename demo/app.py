"""Face deepfake detector demo: CLIP detector + physics explanations.

    python demo/app.py            (then open http://127.0.0.1:7860)

Pipeline (identical to training, notebooks 05-06):
  MediaPipe face mesh -> square face crop (margin 0.3) -> 256 px -> JPEG q95
  -> CLIP ViT-L/14 embedding -> linear probe           ("CLIP")
  -> face-physics features    -> gradient boosting      ("Physics")
  -> logistic stacking of both                          ("Fused", the headline score)

Physics also explains the image: each check is compared with ~14k real face photos.
In our evaluation (RESULTS.md, notebooks 05-06) physics did not measurably improve accuracy;
here it shows *what* is physically unusual about a face, not a better score.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import gradio as gr
import joblib
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from physfeat import face as F                       # noqa: E402
from physfeat.imaging import jpeg_roundtrip           # noqa: E402

MODEL_PATH = Path(os.environ.get("PHYSFAKE_DEMO_MODEL", ROOT / "models" / "face_detector_all.joblib"))
REF_PATH = ROOT / "assets" / "real_face_reference.json"
EXAMPLES_DIR = Path(os.environ.get("PHYSFAKE_EXAMPLES", ROOT / "examples"))
UNUSUAL = 0.05        # a check is "unusual" if fewer than 5% of real faces are at least as extreme

# physics checks shown to the user:
# feature -> (suspicious direction, label, what it measures, why it can be unmeasurable)
SKIN, EYES, PUPIL = ("too little usable skin (hair, glasses, shadow or a small face)",
                     "eyes too small, closed, or without a visible reflection in both eyes",
                     "pupil not separable from the iris (common for dark eyes) or too small")
CHECKS = {
    "face_r2_sh1": (-1, "One light explains the skin shading",
                    "how much of the skin's brightness a single light source accounts for", SKIN),
    "face_lr_light_deg": (+1, "Left and right halves agree on the light",
                          "angle between the light directions fitted on each half of the face", SKIN),
    "face_lr_cross_r2": (-1, "One half's light predicts the other half",
                         "how well the light fitted on one half reproduces the other half's shading", SKIN),
    "face_spec_align": (-1, "Skin highlights sit where the light says",
                        "whether shiny spots are where the estimated light would put them",
                        "no shiny highlights on the skin to check"),
    "face_eye_hl_iou": (-1, "The two eyes reflect the same scene",
                        "overlap of the corneal reflections of the two eyes", EYES),
    "face_eye_hl_offset": (+1, "Eye reflections in the same place",
                           "distance between the reflections' positions in the two irises", EYES),
    "face_pupil_irreg_mean": (+1, "Pupils are round / elliptical",
                              "how far the pupil outline departs from an ellipse", PUPIL),
    "face_pose_inner_outer_deg": (+1, "Inner face and face outline point the same way",
                                  "angle between the orientation of eyes-nose-mouth and of the face outline",
                                  "face mesh not available"),
}

# ----------------------------------------------------------------------------
# models (loaded once)
# ----------------------------------------------------------------------------
bundle = joblib.load(MODEL_PATH)
DET = bundle["detector"]
CROP = bundle["crop"]
REF = json.loads(REF_PATH.read_text())["features"]
MESHER = F.FaceMesher()

import torch                                          # noqa: E402
import transformers                                   # noqa: E402
from physfeat.clipfeat import PREPROCESS, load_clip   # noqa: E402
transformers.logging.set_verbosity_error()   # only CLIP's vision half is loaded: silence "unexpected key" notes
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP = load_clip(bundle["clip_model"], DEVICE)
DTYPE = next(CLIP.parameters()).dtype


@torch.no_grad()
def clip_embed(crop_rgb: np.ndarray) -> np.ndarray:
    x = PREPROCESS(Image.fromarray(crop_rgb)).unsqueeze(0).to(DEVICE, DTYPE)
    e = torch.nn.functional.normalize(CLIP(pixel_values=x).image_embeds.float(), dim=-1)
    return e.cpu().numpy().astype(np.float32)


# ----------------------------------------------------------------------------
# explanation helpers
# ----------------------------------------------------------------------------
def share_as_extreme(feature, value, direction):
    """Share of real faces whose value is at least as extreme (in the suspicious direction)."""
    p = np.asarray(REF[feature]["percentiles"])
    if direction < 0:   # low is suspicious: share of reals <= value
        return float(np.searchsorted(p, value, side="right")) / len(p)
    return float(len(p) - np.searchsorted(p, value, side="left")) / len(p)


def explain(feats):
    rows, notes = [], []
    for f, (direction, label, what, missing) in CHECKS.items():
        v = feats.get(f, np.nan)
        if f not in REF or not np.isfinite(v):
            rows.append([label, "not measurable", "—", missing])
            continue
        share = share_as_extreme(f, v, direction)
        status = "UNUSUAL" if share < UNUSUAL else "typical"
        rows.append([label, f"{v:.3g}", f"{100 * share:.0f}% of real faces are this extreme or more", status])
        if share < UNUSUAL:
            notes.append(f"**{label}: unusual.** {what.capitalize()} is more extreme than in "
                         f"{100 * (1 - share):.0f}% of real face photos.")
    return pd.DataFrame(rows, columns=["Physics check", "Measured", "Compared with real photos", "Status"]), notes


def annotate(rgb, L):
    out = rgb.copy()
    s = max(1, round(max(rgb.shape[:2]) / 400))
    for x, y in L[F.FACE_OVAL, :2].astype(int):
        cv2.circle(out, (int(x), int(y)), s + 1, (80, 200, 255), -1, cv2.LINE_AA)
    for c_idx, ring in F.IRISES:
        c = L[c_idx, :2]
        r = float(np.mean(np.linalg.norm(L[ring, :2] - c, axis=1)))
        cv2.circle(out, (int(c[0]), int(c[1])), max(2, int(r)), (255, 80, 80), s, cv2.LINE_AA)
    return out


# ----------------------------------------------------------------------------
# main analysis
# ----------------------------------------------------------------------------
def analyse(image):
    """Returns (verdict, annotated image, evidence strip, scores table, summary markdown, checks table)."""
    if image is None:
        return None, None, None, None, "Upload an image with a face.", None
    rgb = np.asarray(Image.fromarray(image).convert("RGB")) if isinstance(image, np.ndarray) \
        else np.asarray(image.convert("RGB"))
    L = MESHER(rgb)
    if L is None:
        return None, rgb, None, None, "**No face found.** The detector works on images with a clearly visible face.", None
    # canonical crop, exactly as in training
    crop = F.face_crop(rgb, L, size=CROP["size"], margin=CROP["margin"])
    crop = np.asarray(jpeg_roundtrip(Image.fromarray(crop), CROP["jpeg_q"]))
    emb = clip_embed(crop)
    feats, dbg = F.extract(crop, mesher=MESHER, return_debug=True)
    phys = pd.DataFrame([feats]).reindex(columns=F.FEATURE_NAMES).astype(float)
    names = type(DET).NAMES
    s = DET.predict(emb, phys, np.array([float(CROP["jpeg_q"])]))
    p_fused, p_clip, p_phys = (float(s[names[k]][0]) for k in ("fused_q", "clip", "phys"))

    verdict = {"AI-generated or manipulated": p_fused, "Real": 1 - p_fused}
    scores = pd.DataFrame([["Fused (CLIP + physics)", f"{p_fused:.3f}"], ["CLIP only", f"{p_clip:.3f}"],
                           ["Physics only", f"{p_phys:.3f}"]], columns=["Model", "P(fake)"])
    table, notes = explain(feats)
    evidence = F.draw_evidence(dbg) if dbg is not None else crop

    n_unusual = len(notes)
    head = ("### Likely AI-generated or manipulated" if p_fused >= 0.5 else "### Likely a real photo") + \
           f" (P(fake) = {p_fused:.2f})\n"
    if n_unusual:
        body = f"{n_unusual} physics check{'s' if n_unusual > 1 else ''} look{'s' if n_unusual == 1 else ''} " \
               "unusual compared with real photos:\n\n" + "\n".join(f"- {n}" for n in notes)
    else:
        body = "All measurable physics checks are within the range of real photos."
    body += ("\n\n*How to read this:* the score comes from the detector (CLIP and physics combined). The physics checks "
             "explain what is physically odd about this particular face. Each check is unusual for about 5% of real "
             "photos by construction, and many AI-generated faces pass them, so treat them as evidence, not proof.")
    return verdict, annotate(rgb, L), evidence, scores, head + "\n" + body, table


ABOUT = """
### About this detector
* **Trained on** ~40k face crops: real photos (FFHQ, DiffusionFace, FaceForensics++) and fakes from GANs (StyleGAN, StyleGAN2),
  diffusion models (ADM, DDIM, LDM, PNDM, Stable Diffusion img2img / text2img), face swaps (DiffSwap, FF++ Deepfakes /
  FaceSwap / FaceShifter), face reenactment (Face2Face, NeuralTextures) and inpainting.
* **Accuracy** on held-out images: mean AUC 0.86 across 16 test sets. On a kind of fake it has *never* seen, about 0.75.
  It is weakest on reenactment (NeuralTextures, Face2Face) and inpainting.
* **Physics** (face lighting from a 3D face mesh, corneal reflections, pupil shape, head-pose consistency) did *not*
  measurably improve accuracy in our experiments (fused − CLIP ≈ +0.001 AUC); here it is used to **explain** the image.
* Works on one face at a time; very small faces, profile views, sunglasses or heavy compression reduce what can be measured.
* A research demo: do not use it to make decisions about real people.
"""


def build():
    with gr.Blocks(title="Face deepfake detector") as demo:
        gr.Markdown("# Face deepfake detector\nCLIP detector with physics-based explanations — "
                    "upload a photo with a face.")
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(label="Image", type="numpy", sources=["upload", "clipboard"])
                btn = gr.Button("Analyse", variant="primary")
                examples = sorted(EXAMPLES_DIR.glob("*.jpg"))[:40] if EXAMPLES_DIR.exists() else []
                if examples:
                    gr.Examples([[str(p)] for p in examples], inputs=inp, label="Examples (face crops from the test sets)",
                                examples_per_page=12)
            with gr.Column(scale=1):
                verdict = gr.Label(label="Verdict", num_top_classes=2)
                summary = gr.Markdown()
                scores = gr.Dataframe(label="Scores", interactive=False)
        with gr.Row():
            annotated = gr.Image(label="Detected face (outline and irises)", interactive=False)
            evidence = gr.Image(label="Physics evidence: skin used | observed shading | shading from one light "
                                      "(arrows: global / left / right light) | irises (red = reflection, green = pupil fit)",
                                interactive=False)
        checks = gr.Dataframe(label="Physics checks", interactive=False, wrap=True)
        gr.Markdown(ABOUT)
        outs = [verdict, annotated, evidence, scores, summary, checks]
        btn.click(analyse, inputs=inp, outputs=outs)
        inp.upload(analyse, inputs=inp, outputs=outs)
    return demo


if __name__ == "__main__":
    build().launch(server_name=os.environ.get("PHYSFAKE_HOST", "127.0.0.1"),
                   server_port=int(os.environ.get("PHYSFAKE_PORT", "7860")))
