"""Monocular surface-normal estimation (GPU) + the lighting features that consume it.

Backends
--------
* "marigold" (default): prs-eth/marigold-normals-v1-1 via diffusers.MarigoldNormalsPipeline.
  Output: unit normals in the camera frame, x right, y up, z toward the viewer.
* "depth": depth-anything/Depth-Anything-V2-Small-hf via transformers; normals are
  derived from the gradients of the relative (inverse) depth. Much faster, coarser.
  Used automatically if Marigold cannot be loaded.

Every image is centre-cropped to a square and resized to RES x RES so batches have
one shape; the lighting features are computed on exactly the same crop.
"""
from __future__ import annotations

import gc

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from . import lighting
from .imaging import load_pil

MARIGOLD_ID = "prs-eth/marigold-normals-v1-1"
DEPTH_ID = "depth-anything/Depth-Anything-V2-Small-hf"


def square_crop(im, res):
    w, h = im.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return im.crop((left, top, left + s, top + s)).resize((res, res), resample=3)  # bicubic


class CropList(Dataset):
    def __init__(self, paths, res, normalise_format=True, perturb=None):
        self.paths, self.res, self.norm, self.perturb = list(paths), res, normalise_format, perturb

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            im = square_crop(load_pil(self.paths[i], self.norm, self.perturb), self.res)
            return torch.from_numpy(np.asarray(im).copy()), True
        except Exception:
            return torch.zeros(self.res, self.res, 3, dtype=torch.uint8), False


class NormalEstimator:
    def __init__(self, backend="marigold", device="cuda", steps=None, res=512, model_id=None):
        self.device, self.steps, self.res = device, steps, res
        self.dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.backend = backend
        if backend == "marigold":
            from diffusers import MarigoldNormalsPipeline
            mid = model_id or MARIGOLD_ID
            try:
                self.pipe = MarigoldNormalsPipeline.from_pretrained(mid, variant="fp16", torch_dtype=self.dtype)
            except Exception:
                self.pipe = MarigoldNormalsPipeline.from_pretrained(mid, torch_dtype=self.dtype)
            self.pipe = self.pipe.to(device)
            self.pipe.set_progress_bar_config(disable=True)
        elif backend == "depth":
            from transformers import AutoModelForDepthEstimation
            self.model = AutoModelForDepthEstimation.from_pretrained(
                model_id or DEPTH_ID, torch_dtype=self.dtype).to(device).eval()
        else:
            raise ValueError(backend)

    @torch.no_grad()
    def __call__(self, batch_uint8: torch.Tensor) -> np.ndarray:
        """batch_uint8: (B,R,R,3) uint8 -> (B,R',R',3) float32 unit normals (x right, y up, z to viewer)."""
        x = batch_uint8.permute(0, 3, 1, 2).float().div(255.0)
        if self.backend == "marigold":
            out = self.pipe(x.to(self.device, self.dtype), num_inference_steps=self.steps,
                            processing_resolution=self.res, match_input_resolution=False,
                            batch_size=len(x), output_type="np",
                            generator=torch.Generator(device=self.device).manual_seed(0))
            return np.asarray(out.prediction, np.float32)
        # ---- depth backend: normals from relative inverse-depth gradients ----------------
        side = (self.res // 14) * 14
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        inp = F.interpolate((x - mean) / std, size=(side, side), mode="bilinear", align_corners=False)
        d = self.model(pixel_values=inp.to(self.device, self.dtype)).predicted_depth.float()
        d = F.interpolate(d[:, None], size=(side, side), mode="bilinear", align_corners=False)[:, 0]
        d = d / (d.flatten(1).quantile(0.95, dim=1).view(-1, 1, 1) + 1e-6)   # scale-normalise
        gx = torch.zeros_like(d); gy = torch.zeros_like(d)
        gx[:, :, 1:-1] = (d[:, :, 2:] - d[:, :, :-2]) / 2
        gy[:, 1:-1, :] = (d[:, 2:, :] - d[:, :-2, :]) / 2
        k = 0.577 / (torch.sqrt(gx ** 2 + gy ** 2).flatten(1).median(1).values.view(-1, 1, 1) + 1e-6)
        n = torch.stack([-gx * k, gy * k, torch.ones_like(d)], -1)   # height = disparity (toward viewer)
        n = n / n.norm(dim=-1, keepdim=True)
        return n.cpu().numpy().astype(np.float32)


def lighting_features(estimator: NormalEstimator, paths, batch_size=8, num_workers=4,
                      normalise_format=True, perturb=None, progress=True, keep_examples=0):
    """Returns (list of feature dicts, list of (rgb, normals) examples for plotting)."""
    dl = DataLoader(CropList(paths, estimator.res, normalise_format, perturb), batch_size=batch_size,
                    num_workers=num_workers, pin_memory=estimator.device.startswith("cuda"))
    feats, examples = [], []
    it = dl
    bar = None
    if progress:
        from tqdm.auto import tqdm
        bar = it = tqdm(dl, desc=f"normals{'/' + perturb if perturb else ''}", leave=False)
    try:
        for imgs, ok in it:
            normals = estimator(imgs)
            rgbs = imgs.numpy()
            for rgb, n, good in zip(rgbs, normals, ok.numpy()):
                if not good:
                    f = {k: np.nan for k in lighting.FEATURE_NAMES}
                else:
                    try:
                        f = lighting.extract(rgb, n)
                    except Exception:
                        f = {k: np.nan for k in lighting.FEATURE_NAMES}
                feats.append(f)
                if len(examples) < keep_examples and good:
                    examples.append((rgb, n.astype(np.float16)))
    finally:  # tidy DataLoader workers explicitly (avoids noisy __del__ tracebacks in forked children)
        if bar is not None:
            bar.close()
        del it, dl
        gc.collect()
    return feats, examples