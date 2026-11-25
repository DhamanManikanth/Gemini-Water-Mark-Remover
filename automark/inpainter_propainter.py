"""Video watermark removal: temporal alpha/logo inversion (I = a*L + (1-a)*B) + ProPainter fill."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from .config import (
    ALPHA_INPAINT_THRESHOLD,
    MODELS_DIR,
    PROPAINTER_CHUNK_FRAMES,
    PROPAINTER_DIR,
    PROPAINTER_LOW_VRAM_CHUNK,
    PROPAINTER_LOW_VRAM_THRESHOLD_GB,
    PROPAINTER_OVERLAP_FRAMES,
    get_device,
    get_vram_gb,
)


def estimate_temporal_alpha_and_logo(
    frames_bgr: list[np.ndarray],
    hard_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Estimate per-pixel alpha `a` and pre-multiplied logo `a*L` from temporal statistics.

    Model: I_t = a*L + (1-a)*B_t
      Var_t(I_t) = (1-a)^2 * Var_t(B_t)  =>  1-a = sqrt(Var_t(I_t) / Var_ring)
      a*L = median_t(I_t) - (1-a) * B_ring_est

    Returns:
        alpha_map: float32 (H, W) in [0, 1]
        al_map: float32 (H, W, 3) estimated a*L
        valid: True if background had enough temporal variance for reliable alpha estimation
    """
    h, w = hard_mask.shape[:2]
    mask_bool = hard_mask > 0
    if not mask_bool.any() or len(frames_bgr) < 12:
        return np.zeros((h, w), dtype=np.float32), np.zeros((h, w, 3), dtype=np.float32), False

    ys, xs = np.where(mask_bool)
    pad = 18
    y0, y1 = max(0, int(ys.min()) - pad), min(h, int(ys.max()) + 1 + pad)
    x0, x1 = max(0, int(xs.min()) - pad), min(w, int(xs.max()) + 1 + pad)

    step = max(1, len(frames_bgr) // 80)
    sampled = frames_bgr[::step]
    stack = np.stack([fr[y0:y1, x0:x1].astype(np.float32) for fr in sampled], axis=0)  # (T, rh, rw, 3)

    local_mask = mask_bool[y0:y1, x0:x1]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    ring = (cv2.dilate(local_mask.astype(np.uint8), k) > 0) & (~local_mask)

    var_px = np.var(stack, axis=0).mean(axis=-1)  # (rh, rw)
    ring_var = float(np.median(var_px[ring])) if ring.any() else 0.0
    if ring_var < 8.0:
        # Static background: cannot separate alpha temporally
        return np.zeros((h, w), dtype=np.float32), np.zeros((h, w, 3), dtype=np.float32), False

    # 1 - a = sqrt(var_px / ring_var)
    one_minus_a = np.sqrt(np.clip(var_px / (ring_var + 1e-6), 0.0, 1.0))
    local_alpha = np.clip(1.0 - one_minus_a, 0.0, 1.0)
    local_alpha[~local_mask] = 0.0
    local_alpha = cv2.GaussianBlur(local_alpha, (3, 3), 0)
    local_alpha[~local_mask] = 0.0

    # Estimate temporal median of I_t and smooth background B_est from surrounding ring
    med_i = np.median(stack, axis=0)  # (rh, rw, 3)
    b_est = med_i.copy()
    if ring.any():
        ring_color = np.median(med_i[ring], axis=0)
        b_est[local_mask] = ring_color

    local_al = med_i - (1.0 - local_alpha[..., None]) * b_est
    local_al[~local_mask] = 0.0

    alpha_map = np.zeros((h, w), dtype=np.float32)
    al_map = np.zeros((h, w, 3), dtype=np.float32)
    alpha_map[y0:y1, x0:x1] = local_alpha
    al_map[y0:y1, x0:x1] = local_al
    return alpha_map, al_map, True


def recover_frames_alpha_inversion(
    frames_bgr: list[np.ndarray],
    hard_mask: np.ndarray,
    alpha_map: np.ndarray,
    al_map: np.ndarray,
) -> list[np.ndarray]:
    """Recover B = (I - a*L) / (1 - a) in float32, clamped, for pixels where a <= 0.85."""
    mask_bool = (hard_mask > 0) & (alpha_map <= ALPHA_INPAINT_THRESHOLD)
    if not mask_bool.any():
        return [fr.copy() for fr in frames_bgr]

    ys, xs = np.where(mask_bool)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1

    local_bool = mask_bool[y0:y1, x0:x1]
    denom = np.maximum(1.0 - alpha_map[y0:y1, x0:x1], 0.15)[..., None]
    al_crop = al_map[y0:y1, x0:x1]

    out_frames: list[np.ndarray] = []
    for fr in frames_bgr:
        out = fr.copy()
        patch = fr[y0:y1, x0:x1].astype(np.float32)
        rec = np.clip((patch - al_crop) / denom, 0.0, 255.0)
        rec_u8 = np.round(rec).astype(np.uint8)
        out_crop = out[y0:y1, x0:x1]
        out_crop[local_bool] = rec_u8[local_bool]
        out[~(hard_mask > 0)] = fr[~(hard_mask > 0)]
        out_frames.append(out)
    return out_frames


def propainter_available() -> bool:
    """Return True if third_party/ProPainter is present."""
    return (PROPAINTER_DIR / "model" / "propainter.py").is_file()


def _ensure_propainter() -> None:
    """Clone ProPainter into third_party/ProPainter on demand if missing."""
    if propainter_available():
        return
    import shutil
    import subprocess

    git_bin = shutil.which("git")
    if not git_bin:
        raise RuntimeError(
            "ProPainter is not installed in third_party/ProPainter and git is not available to clone it."
        )
    PROPAINTER_DIR.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [git_bin, "clone", "--depth", "1", "https://github.com/sczhou/ProPainter.git", str(PROPAINTER_DIR)],
        check=True,
    )


def _run_propainter_chunk(
    chunk_bgr: list[np.ndarray],
    crop_mask_u8: np.ndarray,
    use_fp16: bool,
    device: torch.device,
) -> list[np.ndarray]:
    """Run ProPainter on a single ROI chunk (T, H, W, 3) with H,W divisible by 8."""
    if str(PROPAINTER_DIR) not in sys.path:
        sys.path.insert(0, str(PROPAINTER_DIR))

    from model.modules.flow_comp_raft import RAFT_bi
    from model.recurrent_flow_completion import RecurrentFlowCompleteNet
    from model.propainter import InpaintGenerator
    from utils.download_util import load_file_from_url

    weights_dir = MODELS_DIR / "propainter"
    weights_dir.mkdir(parents=True, exist_ok=True)
    pretrain_url = "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/"

    raft_path = load_file_from_url(pretrain_url + "raft-things.pth", model_dir=str(weights_dir))
    flow_path = load_file_from_url(pretrain_url + "recurrent_flow_completion.pth", model_dir=str(weights_dir))
    prop_path = load_file_from_url(pretrain_url + "ProPainter.pth", model_dir=str(weights_dir))

    raft = RAFT_bi(raft_path, device)
    fix_flow = RecurrentFlowCompleteNet(flow_path).to(device).eval()
    model = InpaintGenerator(model_path=prop_path).to(device).eval()

    if use_fp16 and device.type == "cuda":
        fix_flow = fix_flow.half()
        model = model.half()

    t_len = len(chunk_bgr)
    # Prepare tensors (1, T, C, H, W) in [-1, 1] and mask (1, T, 1, H, W) in {0, 1}
    frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0 for f in chunk_bgr]
    frames_t = torch.from_numpy(np.stack(frames_rgb, axis=0)).permute(0, 3, 1, 2).unsqueeze(0).to(device)
    m_f = (crop_mask_u8 > 0).astype(np.float32)[None, None, None, :, :]
    masks_t = torch.from_numpy(np.repeat(m_f, t_len, axis=1)).to(device)

    if use_fp16 and device.type == "cuda":
        frames_t = frames_t.half()
        masks_t = masks_t.half()

    with torch.inference_mode():
        gt_flows_bi = raft(frames_t, iters=20)
        if use_fp16 and device.type == "cuda":
            gt_flows_bi = (gt_flows_bi[0].half(), gt_flows_bi[1].half())
        pred_flows_bi, _ = fix_flow.forward_bidirect_flow(gt_flows_bi, masks_t)
        pred_flows_bi = fix_flow.combine_flow(gt_flows_bi, pred_flows_bi, masks_t)

        masked_frames = frames_t * (1.0 - masks_t)
        comp_frames = model(masked_frames, pred_flows_bi, masks_t, masks_t, num_local_frames=t_len)
        comp_frames = comp_frames.float().clamp(-1.0, 1.0)
        out_np = ((comp_frames[0].permute(0, 2, 3, 1).cpu().numpy() + 1.0) * 127.5).round().astype(np.uint8)

    return [cv2.cvtColor(out_np[i], cv2.COLOR_RGB2BGR) for i in range(t_len)]


def inpaint_video_propainter(
    frames_bgr: list[np.ndarray],
    hard_mask: np.ndarray,
) -> list[np.ndarray]:
    """Inpaint `frames_bgr` strictly inside `hard_mask` using ProPainter in overlapping chunks.

    Chunks: 80 frames (40 on low VRAM <= 6.5 GB), 10-frame overlap, linear cross-fade.
    Uses fp16 on CUDA. Every pixel outside `hard_mask` is copied bit-identically.
    """
    mask_bool = hard_mask > 0
    if not mask_bool.any():
        return [fr.copy() for fr in frames_bgr]

    _ensure_propainter()

    device = get_device()
    vram_gb = get_vram_gb()
    low_vram = device.type == "cuda" and vram_gb <= PROPAINTER_LOW_VRAM_THRESHOLD_GB
    chunk_size = PROPAINTER_LOW_VRAM_CHUNK if low_vram else PROPAINTER_CHUNK_FRAMES
    overlap = min(PROPAINTER_OVERLAP_FRAMES, chunk_size // 4)
    use_fp16 = device.type == "cuda"

    h, w = frames_bgr[0].shape[:2]
    ys, xs = np.where(mask_bool)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    bw, bh = x1 - x0, y1 - y0
    pad = max(96, 3 * max(bw, bh))
    cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
    cx1, cy1 = min(w, x1 + pad), min(h, y1 + pad)

    # Align crop dimensions to multiples of 8
    cw, ch = cx1 - cx0, cy1 - cy0
    pad_w = (8 - (cw % 8)) % 8
    pad_h = (8 - (ch % 8)) % 8

    crop_mask = (hard_mask[cy0:cy1, cx0:cx1] > 0).astype(np.uint8) * 255
    if pad_h or pad_w:
        crop_mask_p = cv2.copyMakeBorder(crop_mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
    else:
        crop_mask_p = crop_mask

    n = len(frames_bgr)
    step = max(1, chunk_size - overlap)
    accum = [np.zeros((ch, cw, 3), dtype=np.float32) for _ in range(n)]
    weights = np.zeros(n, dtype=np.float32)

    start = 0
    while start < n:
        end = min(n, start + chunk_size)
        chunk_crops = []
        for idx in range(start, end):
            c = frames_bgr[idx][cy0:cy1, cx0:cx1]
            if pad_h or pad_w:
                c = cv2.copyMakeBorder(c, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            chunk_crops.append(c)

        pred_chunk = _run_propainter_chunk(chunk_crops, crop_mask_p, use_fp16=use_fp16, device=device)

        c_len = end - start
        for local_i in range(c_len):
            global_i = start + local_i
            w_val = 1.0
            if start > 0 and local_i < overlap:
                w_val = float(local_i + 1) / float(overlap + 1)
            elif end < n and (c_len - 1 - local_i) < overlap:
                w_val = float(c_len - local_i) / float(overlap + 1)
            accum[global_i] += pred_chunk[local_i][:ch, :cw].astype(np.float32) * w_val
            weights[global_i] += w_val

        if end == n:
            break
        start += step

    local_bool = mask_bool[cy0:cy1, cx0:cx1]
    out_frames: list[np.ndarray] = []
    for i, src in enumerate(frames_bgr):
        out = src.copy()
        blended = np.clip(np.round(accum[i] / max(weights[i], 1e-6)), 0, 255).astype(np.uint8)
        out_crop = out[cy0:cy1, cx0:cx1]
        out_crop[local_bool] = blended[local_bool]
        out[~mask_bool] = src[~mask_bool]
        out_frames.append(out)
    return out_frames


def remove_watermark_video_frames(
    frames_bgr: list[np.ndarray],
    hard_mask: np.ndarray,
) -> list[np.ndarray]:
    """Remove watermark from video frames: temporal I=a*L+(1-a)*B recovery, ProPainter where a>0.85."""
    alpha_map, al_map, valid = estimate_temporal_alpha_and_logo(frames_bgr, hard_mask)
    if valid:
        recovered = recover_frames_alpha_inversion(frames_bgr, hard_mask, alpha_map, al_map)
        high_alpha_mask = ((hard_mask > 0) & (alpha_map > ALPHA_INPAINT_THRESHOLD)).astype(np.uint8) * 255
        if high_alpha_mask.any():
            return inpaint_video_propainter(recovered, high_alpha_mask)
        return recovered
    return inpaint_video_propainter(frames_bgr, hard_mask)
