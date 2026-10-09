"""CLIP ViT-L/14 image embeddings (the UnivFD-style data-driven baseline)."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from .imaging import load_pil

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

PREPROCESS = T.Compose([
    T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(CLIP_MEAN, CLIP_STD),
])


class ImageList(Dataset):
    def __init__(self, paths, normalise_format=True, perturb=None):
        self.paths, self.norm, self.perturb = list(paths), normalise_format, perturb

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            return PREPROCESS(load_pil(self.paths[i], self.norm, self.perturb)), True
        except Exception:
            return torch.zeros(3, 224, 224), False


def load_clip(model_name: str, device: str):
    from transformers import CLIPVisionModelWithProjection
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = CLIPVisionModelWithProjection.from_pretrained(model_name, torch_dtype=dtype)
    return model.to(device).eval()


@torch.no_grad()
def embed(model, paths, device, batch_size=128, num_workers=4,
          normalise_format=True, perturb=None, progress=True):
    """Returns (N, D) float16 L2-normalised embeddings and an (N,) ok-mask."""
    dl = DataLoader(ImageList(paths, normalise_format, perturb), batch_size=batch_size,
                    num_workers=num_workers, pin_memory=device.startswith("cuda"))
    dtype = next(model.parameters()).dtype
    feats, oks = [], []
    it, bar = dl, None
    if progress:
        from tqdm.auto import tqdm
        it = bar = tqdm(dl, desc=f"CLIP{'/' + perturb if perturb else ''}", leave=False)
    try:
        for x, ok in it:
            e = model(pixel_values=x.to(device, dtype)).image_embeds.float()
            e = torch.nn.functional.normalize(e, dim=-1)
            feats.append(e.cpu().numpy().astype(np.float16))
            oks.append(ok.numpy())
    finally:  # shut DataLoader workers down here, not in a later forked child (noisy tracebacks)
        if bar is not None:
            bar.close()
        del it, dl
        import gc
        gc.collect()
    return np.concatenate(feats), np.concatenate(oks)