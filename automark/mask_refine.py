"""Mask refinement: region growing from surrounding ring, largest component, dilation, feathering."""

from __future__ import annotations

from collections import deque
import cv2
import numpy as np

from .config import DEFAULT_DILATION_PX, DEFAULT_FEATHER_PX
from .detect import Candidate


def refine_mask_from_candidate(
    image_bgr: np.ndarray,
    candidate: Candidate,
    initial_mask: np.ndarray | None = None,
    dilation_px: int = DEFAULT_DILATION_PX,
    feather_px: int = DEFAULT_FEATHER_PX,
) -> tuple[np.ndarray, np.ndarray]:
    """Refine a candidate bounding box / seed mask into exact logo shape.

    1. Region growing on gradient and brightness difference from the surrounding ring.
    2. Keep the largest connected component.
    3. Apply dilation (1-2 px) and feathering (1-2 px).

    Returns:
        hard_mask: uint8 (H, W) binary {0, 255} mask (strictly defines editable pixels).
        soft_mask: float32 (H, W) in [0, 1] (0 outside hard_mask, feathered near boundary).
    """
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # Define ROI around candidate with a surrounding ring
    pad = max(12, int(round(0.45 * max(candidate.w, candidate.h))))
    x0 = max(0, candidate.x - pad)
    y0 = max(0, candidate.y - pad)
    x1 = min(w, candidate.x + candidate.w + pad)
    y1 = min(h, candidate.y + candidate.h + pad)

    roi_gray = gray[y0:y1, x0:x1]
    rh, rw = roi_gray.shape

    gx = cv2.Sobel(roi_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi_gray, cv2.CV_32F, 0, 1, ksize=3)
    roi_grad = np.hypot(gx, gy)

    # Estimate surrounding ring statistics (outer border of ROI)
    ring = np.ones((rh, rw), dtype=bool)
    inner_x0 = max(1, candidate.x - x0 - 2)
    inner_y0 = max(1, candidate.y - y0 - 2)
    inner_x1 = min(rw - 1, candidate.x + candidate.w - x0 + 2)
    inner_y1 = min(rh - 1, candidate.y + candidate.h - y0 + 2)
    ring[inner_y0:inner_y1, inner_x0:inner_x1] = False

    ring_vals = roi_gray[ring] if ring.any() else roi_gray.ravel()
    ring_med = float(np.median(ring_vals))
    ring_mad = float(np.median(np.abs(ring_vals - ring_med))) + 0.015
    ring_grad_med = float(np.median(roi_grad[ring])) if ring.any() else float(np.median(roi_grad))

    # Brightness difference & gradient elevation relative to surrounding ring
    b_diff = np.abs(roi_gray - ring_med) / (1.4826 * ring_mad)
    g_diff = np.maximum(0.0, roi_grad - ring_grad_med) / (ring_grad_med + 0.025)
    score_map = 0.65 * b_diff + 0.35 * g_diff

    # Seed from initial_mask if provided, else center of candidate
    seed = np.zeros((rh, rw), dtype=bool)
    if initial_mask is not None:
        sub_init = initial_mask[y0:y1, x0:x1] > 0
        if sub_init.any():
            seed = sub_init
    if not seed.any():
        scx = int(np.clip(round(candidate.cx - x0), 0, rw - 1))
        scy = int(np.clip(round(candidate.cy - y0), 0, rh - 1))
        seed[max(0, scy - 2):min(rh, scy + 3), max(0, scx - 2):min(rw, scx + 3)] = True

    # Region growing within candidate box (+ small margin)
    allowed = np.zeros((rh, rw), dtype=bool)
    ax0 = max(0, candidate.x - x0 - 4)
    ay0 = max(0, candidate.y - y0 - 4)
    ax1 = min(rw, candidate.x + candidate.w - x0 + 4)
    ay1 = min(rh, candidate.y + candidate.h - y0 + 4)
    allowed[ay0:ay1, ax0:ax1] = True

    grown = seed.copy()
    q = deque(zip(*np.where(seed)))
    grow_thresh = 1.65
    while q:
        cy, cx = q.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < rh and 0 <= nx < rw and allowed[ny, nx] and not grown[ny, nx]:
                if score_map[ny, nx] >= grow_thresh:
                    grown[ny, nx] = True
                    q.append((ny, nx))

    # Keep largest connected component
    grown_u8 = grown.astype(np.uint8) * 255
    num_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(grown_u8, connectivity=8)
    if num_lbl > 1:
        best_lbl = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        grown_u8 = ((labels == best_lbl).astype(np.uint8)) * 255

    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[y0:y1, x0:x1] = grown_u8

    return finalize_mask(full_mask, dilation_px=dilation_px, feather_px=feather_px)


def finalize_mask(
    binary_mask: np.ndarray,
    dilation_px: int = DEFAULT_DILATION_PX,
    feather_px: int = DEFAULT_FEATHER_PX,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply morphological dilation and feathering to a binary uint8 mask.

    Outside `hard_mask`, `soft_mask` is guaranteed to be 0.0 so source pixels remain bit-identical.
    """
    m = (binary_mask > 0).astype(np.uint8) * 255
    if dilation_px > 0:
        k = 2 * int(dilation_px) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        m = cv2.dilate(m, kernel)

    hard_mask = (m > 0).astype(np.uint8) * 255
    soft_mask = (hard_mask > 0).astype(np.float32)

    if feather_px > 0:
        # Distance-based inward feathering so soft_mask == 0 strictly outside hard_mask
        dist = cv2.distanceTransform((hard_mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
        soft_mask = np.clip(dist / float(feather_px + 0.5), 0.0, 1.0).astype(np.float32)
        soft_mask[hard_mask == 0] = 0.0

    return hard_mask, soft_mask
