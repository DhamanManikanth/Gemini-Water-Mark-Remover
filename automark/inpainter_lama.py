"""LaMa image inpainting wrapper with tight ROI cropping and bit-identical outside-mask guarantee."""

from __future__ import annotations

import os
import cv2
import numpy as np
from PIL import Image
import torch

from .config import get_device

_LAMA_INSTANCE = None


class _SafeSimpleLama:
    """SimpleLama wrapper that passes map_location=device to torch.jit.load for CPU fallback."""

    def __init__(self, device: torch.device) -> None:
        from simple_lama_inpainting.models.model import LAMA_MODEL_URL
        from simple_lama_inpainting.utils import download_model, prepare_img_and_mask

        self._prepare = prepare_img_and_mask
        model_path = os.environ.get("LAMA_MODEL") or download_model(LAMA_MODEL_URL)
        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()
        self.model.to(device)
        self.device = device

    def __call__(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        img_t, mask_t = self._prepare(image, mask, self.device)
        with torch.inference_mode():
            inpainted = self.model(img_t, mask_t)
            cur_res = inpainted[0].permute(1, 2, 0).detach().cpu().numpy()
            cur_res = np.clip(cur_res * 255, 0, 255).astype(np.uint8)
            return Image.fromarray(cur_res)


def _get_lama():
    """Lazy-load LaMa on CUDA (if available) or CPU."""
    global _LAMA_INSTANCE
    if _LAMA_INSTANCE is None:
        _LAMA_INSTANCE = _SafeSimpleLama(device=get_device())
    return _LAMA_INSTANCE


def inpaint_image_lama(
    image_bgr: np.ndarray,
    hard_mask: np.ndarray,
    soft_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Inpaint `image_bgr` strictly inside `hard_mask` using LaMa.

    Every pixel outside `hard_mask` is copied bit-identically from `image_bgr`.
    """
    mask_bool = hard_mask > 0
    if not mask_bool.any():
        return image_bgr.copy()

    h, w = image_bgr.shape[:2]
    ys, xs = np.where(mask_bool)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1

    # Generous context window for LaMa, aligned to multiples of 8
    bw, bh = x1 - x0, y1 - y0
    pad = max(96, 3 * max(bw, bh))
    cx0 = max(0, x0 - pad)
    cy0 = max(0, y0 - pad)
    cx1 = min(w, x1 + pad)
    cy1 = min(h, y1 + pad)

    crop_bgr = image_bgr[cy0:cy1, cx0:cx1]
    crop_mask = (hard_mask[cy0:cy1, cx0:cx1] > 0).astype(np.uint8) * 255

    # Pad crop to multiple of 8 for LaMa UNet
    ch, cw = crop_bgr.shape[:2]
    pad_h = (8 - (ch % 8)) % 8
    pad_w = (8 - (cw % 8)) % 8
    if pad_h or pad_w:
        crop_bgr_p = cv2.copyMakeBorder(crop_bgr, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        crop_mask_p = cv2.copyMakeBorder(crop_mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
    else:
        crop_bgr_p = crop_bgr
        crop_mask_p = crop_mask

    rgb_pil = Image.fromarray(cv2.cvtColor(crop_bgr_p, cv2.COLOR_BGR2RGB))
    mask_pil = Image.fromarray(crop_mask_p)

    lama = _get_lama()
    inpainted_pil = lama(rgb_pil, mask_pil)
    inpainted_rgb = np.array(inpainted_pil)[:ch, :cw]
    inpainted_bgr = cv2.cvtColor(inpainted_rgb, cv2.COLOR_RGB2BGR)

    # Start with exact bit-identical copy of source
    out = image_bgr.copy()

    if soft_mask is not None:
        alpha = soft_mask[cy0:cy1, cx0:cx1][..., None]
        blended = alpha * inpainted_bgr.astype(np.float32) + (1.0 - alpha) * crop_bgr.astype(np.float32)
        blended_u8 = np.clip(np.round(blended), 0, 255).astype(np.uint8)
        crop_out = out[cy0:cy1, cx0:cx1]
        local_bool = mask_bool[cy0:cy1, cx0:cx1]
        crop_out[local_bool] = blended_u8[local_bool]
    else:
        crop_out = out[cy0:cy1, cx0:cx1]
        local_bool = mask_bool[cy0:cy1, cx0:cx1]
        crop_out[local_bool] = inpainted_bgr[local_bool]

    # Final hard enforcement: outside mask MUST be bit-identical
    out[~mask_bool] = image_bgr[~mask_bool]
    return out
