"""Synthetic image and moving-background video generators for testing (no external assets)."""

from __future__ import annotations

import cv2
import numpy as np

from automark.detect import generate_sparkle_template


def make_textured_background(h: int, w: int, seed: int = 42) -> np.ndarray:
    """Generate a realistic multi-frequency smooth+textured BGR image."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # Smooth sky/landscape gradient + low-frequency waves
    b = 90.0 + 70.0 * (1.0 - yy / h) + 25.0 * np.sin(xx / 45.0) * np.cos(yy / 60.0)
    g = 110.0 + 50.0 * (yy / h) + 20.0 * np.cos(xx / 55.0 + 0.5)
    r = 120.0 + 60.0 * (xx / w) + 25.0 * np.sin(yy / 50.0 + 1.1)

    # Subtle natural texture (low-amplitude Perlin-like smoothed noise)
    noise = rng.normal(0, 1.0, (h // 4 + 1, w // 4 + 1, 3)).astype(np.float32)
    noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
    noise = cv2.GaussianBlur(noise, (9, 9), 0) * 14.0

    img = np.stack([b, g, r], axis=-1) + noise
    return np.clip(img, 0, 255).astype(np.uint8)


def overlay_synthetic_sparkle(
    image_bgr: np.ndarray,
    corner: str,
    size: int,
    alpha: float = 0.45,
    margin: int | None = None,
    soft_edge: bool = False,
) -> tuple[np.ndarray, float, float, np.ndarray]:
    """Composite a semi-transparent white 4-point star onto `image_bgr`.

    Returns:
        watermarked_bgr: uint8 BGR image
        gt_cx, gt_cy: ground-truth center coordinates
        gt_mask: uint8 (H, W) ground-truth binary mask
    """
    h, w = image_bgr.shape[:2]
    tpl, _, _ = generate_sparkle_template(size, soft_edge=soft_edge)
    m = margin if margin is not None else max(14, int(round(0.035 * min(h, w))))

    if corner == "bottom-right":
        x0, y0 = w - m - size, h - m - size
    elif corner == "bottom-left":
        x0, y0 = m, h - m - size
    elif corner == "top-right":
        x0, y0 = w - m - size, m
    elif corner == "top-left":
        x0, y0 = m, m
    else:
        raise ValueError(f"Unknown corner: {corner}")

    a_patch = (tpl * alpha)[..., None]  # (size, size, 1)
    logo_color = np.full((size, size, 3), 255.0, dtype=np.float32)

    out = image_bgr.astype(np.float32)
    roi = out[y0:y0 + size, x0:x0 + size]
    out[y0:y0 + size, x0:x0 + size] = a_patch * logo_color + (1.0 - a_patch) * roi

    gt_mask = np.zeros((h, w), dtype=np.uint8)
    gt_mask[y0:y0 + size, x0:x0 + size] = (tpl > 0.1).astype(np.uint8) * 255

    gt_cx = x0 + size / 2.0
    gt_cy = y0 + size / 2.0
    return np.clip(np.round(out), 0, 255).astype(np.uint8), gt_cx, gt_cy, gt_mask


def make_moving_background_clip(
    n_frames: int = 70,
    h: int = 360,
    w: int = 640,
    corner: str = "bottom-right",
    size: int = 32,
    alpha: float = 0.50,
    seed: int = 123,
) -> tuple[list[np.ndarray], list[np.ndarray], float, float, np.ndarray]:
    """Generate a moving-background clip + clean ground-truth frames + static sparkle watermark."""
    # Create a large panoramic background and pan across it
    pano = make_textured_background(h + 160, w + 320, seed=seed)
    clean_frames: list[np.ndarray] = []
    wm_frames: list[np.ndarray] = []
    gt_cx, gt_cy, gt_mask = 0.0, 0.0, np.zeros((h, w), dtype=np.uint8)

    for i in range(n_frames):
        ox = int(round(3.5 * i)) % 280
        oy = int(round(1.5 * i)) % 140
        clean = pano[oy:oy + h, ox:ox + w].copy()
        wm, gt_cx, gt_cy, gt_mask = overlay_synthetic_sparkle(
            clean, corner=corner, size=size, alpha=alpha, soft_edge=False
        )
        clean_frames.append(clean)
        wm_frames.append(wm)

    return wm_frames, clean_frames, gt_cx, gt_cy, gt_mask
