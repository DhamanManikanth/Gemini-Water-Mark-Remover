"""Verification checks: outside-mask bit-identity, re-detection, edge energy, video flicker, review crops."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import OUTPUT_MAX_SCORE_THRESHOLD, REVIEW_DIR
from .detect import detect_image, detect_video_frames


@dataclass
class VerificationReport:
    """Result of post-removal verification."""

    passed: bool
    outside_diff_zero: bool
    max_redetect_score: float
    inner_edge_energy: float
    ring_edge_energy: float
    edge_ratio: float
    flicker_ratio: float | None
    review_crop_path: Path | None
    messages: list[str]


def check_outside_mask_identical(
    source: np.ndarray,
    output: np.ndarray,
    hard_mask: np.ndarray,
) -> bool:
    """Return True iff every pixel outside `hard_mask` is bit-identical to `source`."""
    outside = hard_mask == 0
    if not outside.any():
        return True
    return bool(np.array_equal(source[outside], output[outside]))


def compute_edge_energy_comparison(
    image_bgr: np.ndarray,
    hard_mask: np.ndarray,
    ring_width_px: int = 8,
) -> tuple[float, float, float]:
    """Compare mean Sobel edge energy inside `hard_mask` vs surrounding ring."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)

    mask_u8 = (hard_mask > 0).astype(np.uint8)
    k = 2 * ring_width_px + 1
    dil = cv2.dilate(mask_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    ring = (dil > 0) & (mask_u8 == 0)

    inner_e = float(mag[mask_u8 > 0].mean()) if (mask_u8 > 0).any() else 0.0
    ring_e = float(mag[ring].mean()) if ring.any() else 0.0
    ratio = inner_e / (ring_e + 1e-4)
    return inner_e, ring_e, ratio


def save_review_crop(
    before_bgr: np.ndarray,
    after_bgr: np.ndarray,
    hard_mask: np.ndarray,
    stem: str = "review",
    scale: int = 4,
) -> Path:
    """Save an enlarged side-by-side before | after | mask crop to output/review/."""
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    h, w = before_bgr.shape[:2]
    ys, xs = np.where(hard_mask > 0)
    if len(ys) == 0:
        x0, y0, x1, y1 = max(0, w - 120), max(0, h - 120), w, h
    else:
        pad = 24
        x0 = max(0, int(xs.min()) - pad)
        y0 = max(0, int(ys.min()) - pad)
        x1 = min(w, int(xs.max()) + 1 + pad)
        y1 = min(h, int(ys.max()) + 1 + pad)

    cb = before_bgr[y0:y1, x0:x1]
    ca = after_bgr[y0:y1, x0:x1]
    cm = cv2.cvtColor(hard_mask[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)

    panel = np.hstack([cb, ca, cm])
    enlarged = cv2.resize(
        panel,
        (panel.shape[1] * scale, panel.shape[0] * scale),
        interpolation=cv2.INTER_NEAREST,
    )
    out_path = REVIEW_DIR / f"{stem}_before_after.png"
    cv2.imwrite(str(out_path), enlarged)
    return out_path


def verify_image(
    source_bgr: np.ndarray,
    output_bgr: np.ndarray,
    hard_mask: np.ndarray,
    stem: str = "image",
    max_score_thresh: float = OUTPUT_MAX_SCORE_THRESHOLD,
) -> VerificationReport:
    """Run all Step 3 verification checks on a cleaned image."""
    msgs: list[str] = []

    outside_ok = check_outside_mask_identical(source_bgr, output_bgr, hard_mask)
    if not outside_ok:
        msgs.append("FAIL: Pixels outside mask are NOT bit-identical to source.")

    # Re-run detection on the cleaned output
    redet = detect_image(output_bgr, threshold=max_score_thresh)
    all_cands = redet.candidates_a + redet.candidates_b
    mask_u8 = (hard_mask > 0).astype(np.uint8)
    mask_scores = []
    for c in all_cands:
        cx_i = int(np.clip(round(c.cx), 0, output_bgr.shape[1] - 1))
        cy_i = int(np.clip(round(c.cy), 0, output_bgr.shape[0] - 1))
        if mask_u8[max(0, cy_i - 8):cy_i + 9, max(0, cx_i - 8):cx_i + 9].any():
            mask_scores.append(c.score)
    max_score = max(mask_scores, default=0.0)
    redet_ok = max_score <= max_score_thresh
    if not redet_ok:
        msgs.append(f"FAIL: Re-detection near mask scored {max_score:.3f} > {max_score_thresh:.3f}.")
    else:
        msgs.append(f"PASS: Re-detection max score near mask={max_score:.3f} <= {max_score_thresh:.3f}.")

    inner_e, ring_e, ratio = compute_edge_energy_comparison(output_bgr, hard_mask)
    msgs.append(f"Edge energy: inner={inner_e:.4f}, ring={ring_e:.4f}, ratio={ratio:.2f}")

    review_path = save_review_crop(source_bgr, output_bgr, hard_mask, stem=stem)
    passed = outside_ok and redet_ok
    return VerificationReport(
        passed=passed,
        outside_diff_zero=outside_ok,
        max_redetect_score=max_score,
        inner_edge_energy=inner_e,
        ring_edge_energy=ring_e,
        edge_ratio=ratio,
        flicker_ratio=None,
        review_crop_path=review_path,
        messages=msgs,
    )


def verify_video(
    source_frames: list[np.ndarray],
    output_frames: list[np.ndarray],
    hard_mask: np.ndarray,
    stem: str = "video",
    max_score_thresh: float = OUTPUT_MAX_SCORE_THRESHOLD,
) -> VerificationReport:
    """Run Step 3 verification on cleaned video frames including temporal flicker check."""
    msgs: list[str] = []
    outside_ok = all(
        check_outside_mask_identical(s, o, hard_mask)
        for s, o in zip(source_frames, output_frames)
    )
    if not outside_ok:
        msgs.append("FAIL: Some frames have non-zero diff outside the mask.")

    # Re-run corner detection on median of sampled output frames (fast verification)
    step = max(1, len(output_frames) // 25)
    med_out = np.median(np.stack(output_frames[::step], axis=0), axis=0).astype(np.uint8)
    redet = detect_image(med_out, threshold=max_score_thresh, search_full_frame=False)
    all_cands = redet.candidates_a + redet.candidates_b
    max_score = max((c.score for c in all_cands), default=0.0)
    mask_u8 = (hard_mask > 0).astype(np.uint8)
    redet_ok = True
    for c in all_cands:
        cx_i = int(np.clip(round(c.cx), 0, output_frames[0].shape[1] - 1))
        cy_i = int(np.clip(round(c.cy), 0, output_frames[0].shape[0] - 1))
        if mask_u8[max(0, cy_i - 8):cy_i + 9, max(0, cx_i - 8):cx_i + 9].any() and c.score > max_score_thresh:
            redet_ok = False
            msgs.append(f"FAIL: Video re-detection near mask scored {c.score:.3f} > {max_score_thresh:.3f}.")
            break

    # Edge energy on middle frame
    mid = len(output_frames) // 2
    inner_e, ring_e, ratio = compute_edge_energy_comparison(output_frames[mid], hard_mask)

    # Flicker check: frame-to-frame change inside mask vs surrounding ring
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    ring = (cv2.dilate(mask_u8, k) > 0) & (mask_u8 == 0)
    inner_diffs, ring_diffs = [], []
    for i in range(1, len(output_frames)):
        diff = np.abs(output_frames[i].astype(np.float32) - output_frames[i - 1].astype(np.float32))
        if (mask_u8 > 0).any():
            inner_diffs.append(float(diff[mask_u8 > 0].mean()))
        if ring.any():
            ring_diffs.append(float(diff[ring].mean()))
    inner_f = float(np.mean(inner_diffs)) if inner_diffs else 0.0
    ring_f = float(np.mean(ring_diffs)) if ring_diffs else 0.0
    flicker_ratio = inner_f / (ring_f + 1e-4)
    msgs.append(f"Video flicker ratio (inner/ring): {flicker_ratio:.2f}")

    review_path = save_review_crop(source_frames[mid], output_frames[mid], hard_mask, stem=stem)
    passed = outside_ok and redet_ok
    return VerificationReport(
        passed=passed,
        outside_diff_zero=outside_ok,
        max_redetect_score=max_score,
        inner_edge_energy=inner_e,
        ring_edge_energy=ring_e,
        edge_ratio=ratio,
        flicker_ratio=flicker_ratio,
        review_crop_path=review_path,
        messages=msgs,
    )
