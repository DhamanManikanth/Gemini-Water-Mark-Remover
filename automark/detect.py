"""Watermark detection: Method A (sparkle template), Method B (corner anomaly), Method C (temporal video)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import (
    AGREEMENT_RADIUS_PX,
    CORNER_FRACTION,
    MAX_TEMPLATE_FRACTION,
    MIN_TEMPLATE_PX,
    NUM_TEMPLATE_SCALES,
    PROFILES_DIR,
    REJECTION_THRESHOLD,
)


@dataclass
class Candidate:
    """A single watermark detection candidate."""

    method: str                 # 'A', 'B', 'C', or 'AB'/'AC'/'BC'/'ABC'
    x: int                      # Top-left x in full frame
    y: int                      # Top-left y in full frame
    w: int                      # Bounding box width
    h: int                      # Bounding box height
    cx: float                   # Center x in full frame
    cy: float                   # Center y in full frame
    score: float                # Confidence score in [0, 1]
    corner: str                 # 'bottom-right', 'bottom-left', 'top-right', 'top-left', or 'full'
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectionResult:
    """Combined detection output across all applicable methods."""

    accepted: bool
    reason: str
    chosen: Candidate | None
    candidates_a: list[Candidate]
    candidates_b: list[Candidate]
    candidates_c: list[Candidate]
    heatmaps: dict[str, np.ndarray]   # e.g. {'bottom-right': float32 2D map, 'full': ...}
    mask_guess: np.ndarray | None     # Binary uint8 mask (H, W) if accepted or from best candidate


def generate_sparkle_template(size: int, soft_edge: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a four-pointed concave diamond star (Gemini sparkle) template.

    Returns:
        tpl_gray: float32 (size, size) in [0, 1] representing the filled sparkle.
        tpl_grad: float32 (size, size) in [0, 1] representing the sparkle edge gradient magnitude.
        mask: float32 (size, size) in [0, 1] support mask for cv2.matchTemplate.
    """
    s = max(8, int(size))
    # Supersample 4x for anti-aliased subpixel accuracy
    ss = s * 4
    yy, xx = np.mgrid[0:ss, 0:ss].astype(np.float32)
    c = (ss - 1) / 2.0
    r = ss * 0.44  # Leave small padding around tips inside template box

    nx = np.abs(xx - c) / max(r, 1e-5)
    ny = np.abs(yy - c) / max(r, 1e-5)

    # Concave 4-pointed star (astroid-like curve: |x|^p + |y|^p <= 1 with p ~ 0.55)
    p = 0.55
    dist = np.power(nx, p) + np.power(ny, p)

    # Smooth anti-aliased boundary
    aa_width = 0.14 if soft_edge else 0.05
    star_hi = np.clip((1.0 - dist) / aa_width + 0.5, 0.0, 1.0)

    if soft_edge:
        k = max(3, (ss // 12) | 1)
        star_hi = cv2.GaussianBlur(star_hi, (k, k), 0)
        mx = float(star_hi.max())
        if mx > 1e-6:
            star_hi /= mx

    tpl_gray = cv2.resize(star_hi, (s, s), interpolation=cv2.INTER_AREA).astype(np.float32)

    # Gradient magnitude of the sparkle template
    gx = cv2.Sobel(tpl_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(tpl_gray, cv2.CV_32F, 0, 1, ksize=3)
    tpl_grad = np.hypot(gx, gy)
    gmax = float(tpl_grad.max())
    if gmax > 1e-6:
        tpl_grad /= gmax

    # Mask covers the star + a tight ring around it so background outside doesn't dilute
    dil_k = max(3, (s // 6) | 1)
    mask = cv2.dilate((tpl_gray > 0.05).astype(np.float32), np.ones((dil_k, dil_k), np.uint8))
    mask = np.clip(mask, 0.0, 1.0)

    return tpl_gray, tpl_grad, mask


def _compute_sobel_mag(gray_f32: np.ndarray) -> np.ndarray:
    """Compute normalized Sobel gradient magnitude on a float32 grayscale image."""
    gx = cv2.Sobel(gray_f32, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f32, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    return mag


def _corner_regions(h: int, w: int, frac: float = CORNER_FRACTION) -> list[tuple[str, int, int, int, int]]:
    """Return corner ROIs ordered with bottom-right highest priority."""
    cw = max(32, int(round(w * frac)))
    ch = max(32, int(round(h * frac)))
    return [
        ("bottom-right", w - cw, h - ch, w, h),
        ("bottom-left", 0, h - ch, cw, h),
        ("top-right", w - cw, 0, w, ch),
        ("top-left", 0, 0, cw, ch),
    ]


def _get_template_sizes(h: int, w: int) -> list[int]:
    """Generate 40+ template sizes from 16 px up to 15% of the shorter image side."""
    max_px = max(MIN_TEMPLATE_PX + 4, int(round(min(h, w) * MAX_TEMPLATE_FRACTION)))
    raw = np.linspace(MIN_TEMPLATE_PX, max_px, NUM_TEMPLATE_SCALES)
    sizes = sorted({int(round(v)) for v in raw if int(round(v)) >= MIN_TEMPLATE_PX})
    # Ensure at least 40 distinct sizes if range allows, or include every integer step
    if len(sizes) < 40 and (max_px - MIN_TEMPLATE_PX + 1) >= 40:
        sizes = np.linspace(MIN_TEMPLATE_PX, max_px, 44).round().astype(int).tolist()
        sizes = sorted(set(sizes))
    return sizes


def _nms_candidates(candidates: list[Candidate], iou_thresh: float = 0.3, top_k: int = 5) -> list[Candidate]:
    """Non-maximum suppression by score and spatial overlap."""
    ordered = sorted(candidates, key=lambda c: c.score, reverse=True)
    kept: list[Candidate] = []
    for cand in ordered:
        duplicate = False
        for k in kept:
            # Check center distance or box IoU
            dist = np.hypot(cand.cx - k.cx, cand.cy - k.cy)
            if dist < max(8.0, 0.35 * min(cand.w, k.w)):
                duplicate = True
                break
            ix1 = max(cand.x, k.x)
            iy1 = max(cand.y, k.y)
            ix2 = min(cand.x + cand.w, k.x + k.w)
            iy2 = min(cand.y + cand.h, k.y + k.h)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = cand.w * cand.h + k.w * k.h - inter
            if union > 0 and (inter / union) > iou_thresh:
                duplicate = True
                break
        if not duplicate:
            kept.append(cand)
            if len(kept) >= top_k:
                break
    return kept


def detect_method_a(
    image_bgr: np.ndarray,
    search_full_frame: bool = True,
) -> tuple[list[Candidate], dict[str, np.ndarray]]:
    """Method A: Procedural sparkle template matching on grayscale + Sobel gradient-magnitude."""
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    grad = _compute_sobel_mag(gray)

    # Local high-pass grayscale helps detect semi-transparent overlays on bright/dark backgrounds
    blur_k = max(15, (min(h, w) // 16) | 1)
    gray_hp = gray - cv2.GaussianBlur(gray, (blur_k, blur_k), 0)

    sizes = _get_template_sizes(h, w)
    # Precompute templates for both crisp and soft-edge variants
    templates: list[tuple[int, bool, np.ndarray, np.ndarray, np.ndarray]] = []
    for s in sizes:
        for soft in (False, True):
            tg, tgrad, tmask = generate_sparkle_template(s, soft_edge=soft)
            templates.append((s, soft, tg, tgrad, tmask))

    corners = _corner_regions(h, w)
    # Priority boost for bottom-right, slight priority for corners over full frame
    corner_priority = {
        "bottom-right": 1.04,
        "bottom-left": 1.00,
        "top-right": 1.00,
        "top-left": 1.00,
        "full": 0.96,
    }

    all_candidates: list[Candidate] = []
    heatmaps: dict[str, np.ndarray] = {}

    def _scan_region(
        name: str,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        scale_step: int = 1,
    ) -> None:
        roi_gray = gray_hp[y0:y1, x0:x1]
        roi_grad = grad[y0:y1, x0:x1]
        rh, rw = roi_gray.shape
        best_map = np.zeros((rh, rw), dtype=np.float32)

        for idx, (s, soft, tg, tgrad, tmask) in enumerate(templates):
            if scale_step > 1 and (idx % (2 * scale_step) >= 2):
                continue
            if s >= rh - 2 or s >= rw - 2:
                continue

            # Match on high-pass grayscale (take abs because watermark may be lighter or darker)
            res_gray = cv2.matchTemplate(roi_gray, tg - float(tg.mean()), cv2.TM_CCORR_NORMED, mask=tmask)
            res_gray = np.abs(res_gray)
            res_gray = np.nan_to_num(res_gray, nan=0.0, posinf=0.0, neginf=0.0)

            # Match on Sobel gradient magnitude
            res_grad = cv2.matchTemplate(roi_grad, tgrad, cv2.TM_CCORR_NORMED, mask=tmask)
            res_grad = np.nan_to_num(res_grad, nan=0.0, posinf=0.0, neginf=0.0)

            # Combined score: gradient match is most invariant to alpha; grayscale confirms polarity
            combined = 0.60 * res_grad + 0.40 * res_gray
            combined *= corner_priority.get(name, 1.0)

            min_val, max_val, _, max_loc = cv2.minMaxLoc(combined)
            # Place peak into full-ROI heatmap
            mh, mw = combined.shape
            ox, oy = s // 2, s // 2
            sub = best_map[oy:oy + mh, ox:ox + mw]
            np.maximum(sub, combined, out=sub)

            if max_val > 0.20:
                px = x0 + max_loc[0]
                py = y0 + max_loc[1]
                all_candidates.append(
                    Candidate(
                        method="A",
                        x=int(px),
                        y=int(py),
                        w=int(s),
                        h=int(s),
                        cx=float(px + s / 2.0),
                        cy=float(py + s / 2.0),
                        score=float(min(1.0, max_val)),
                        corner=name,
                        details={
                            "size": s,
                            "soft_edge": soft,
                            "grad_score": float(res_grad[max_loc[1], max_loc[0]]),
                            "gray_score": float(res_gray[max_loc[1], max_loc[0]]),
                        },
                    )
                )

        heatmaps[name] = best_map

    # 1. Search the four corners first (bottom-right highest priority)
    for cname, x0, y0, x1, y1 in corners:
        _scan_region(cname, x0, y0, x1, y1, scale_step=1)

    # 2. Then search the full frame (strided scales for speed on large images)
    if search_full_frame:
        step = 2 if min(h, w) > 720 else 1
        _scan_region("full", 0, 0, w, h, scale_step=step)

    top5 = _nms_candidates(all_candidates, top_k=5)
    return top5, heatmaps


def detect_method_b(image_bgr: np.ndarray) -> list[Candidate]:
    """Method B: Corner anomaly scan (30% x 30% corners) scored by edge density, symmetry, margin consistency."""
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    grad = _compute_sobel_mag(gray)

    min_side = min(h, w)
    min_box = max(12, MIN_TEMPLATE_PX - 2)
    max_box = max(min_box + 8, int(round(min_side * MAX_TEMPLATE_FRACTION * 1.3)))

    candidates: list[Candidate] = []

    for cname, x0, y0, x1, y1 in _corner_regions(h, w):
        roi_gray = gray[y0:y1, x0:x1]
        roi_grad = grad[y0:y1, x0:x1]
        rh, rw = roi_gray.shape

        # Local contrast deviation from smooth background
        bg_k = max(15, (min(rh, rw) // 6) | 1)
        bg = cv2.GaussianBlur(roi_gray, (bg_k, bg_k), 0)
        diff = np.abs(roi_gray - bg)

        # Normalize and threshold adaptive anomaly map
        d_med = float(np.median(diff))
        d_mad = float(np.median(np.abs(diff - d_med))) + 1e-5
        z_diff = (diff - d_med) / (1.4826 * d_mad)

        g_med = float(np.median(roi_grad))
        g_mad = float(np.median(np.abs(roi_grad - g_med))) + 1e-5
        z_grad = (roi_grad - g_med) / (1.4826 * g_mad)

        anomaly = ((z_diff > 2.8) | (z_grad > 3.0)).astype(np.uint8) * 255
        # Clean tiny noise and bridge thin star arms
        anomaly = cv2.morphologyEx(anomaly, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        anomaly = cv2.morphologyEx(anomaly, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(anomaly, connectivity=8)
        for lbl in range(1, num_labels):
            bx, by, bw, bh, area = stats[lbl]
            if bw < min_box or bh < min_box or bw > max_box or bh > max_box:
                continue
            # Aspect ratio should be compact
            aspect = max(bw, bh) / max(1, min(bw, bh))
            if aspect > 2.5:
                continue

            # Ignore blobs touching the exact image border (border artifacts)
            fx, fy = x0 + bx, y0 + by
            if fx <= 1 or fy <= 1 or (fx + bw) >= w - 1 or (fy + bh) >= h - 1:
                continue

            # 1. Edge density vs surrounding ring
            pad = max(4, int(round(0.25 * max(bw, bh))))
            rx0, ry0 = max(0, bx - pad), max(0, by - pad)
            rx1, ry1 = min(rw, bx + bw + pad), min(rh, by + bh + pad)
            comp_mask = (labels[ry0:ry1, rx0:rx1] == lbl)
            ring_mask = ~cv2.dilate(comp_mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            grad_crop = roi_grad[ry0:ry1, rx0:rx1]
            inner_edge = float(grad_crop[comp_mask].mean()) if comp_mask.any() else 0.0
            ring_edge = float(grad_crop[ring_mask].mean()) if ring_mask.any() else 0.0
            edge_ratio = inner_edge / (ring_edge + 0.02)
            edge_score = float(np.clip(edge_ratio / 3.5, 0.0, 1.0))

            # 2. Horizontal + vertical symmetry of the local diff patch
            patch = diff[by:by + bh, bx:bx + bw]
            p_norm = patch - float(patch.mean())
            p_std = float(np.sqrt(np.mean(p_norm ** 2))) + 1e-6
            sym_h = float(np.mean(p_norm * np.fliplr(p_norm)) / (p_std ** 2))
            sym_v = float(np.mean(p_norm * np.flipud(p_norm)) / (p_std ** 2))
            sym_score = float(np.clip(0.5 * (sym_h + sym_v), 0.0, 1.0))

            # 3. Margin consistency from the two nearest frame edges
            margin_x = fx if "left" in cname else (w - (fx + bw))
            margin_y = fy if "top" in cname else (h - (fy + bh))
            margin_diff = abs(margin_x - margin_y) / max(1.0, float(max(margin_x, margin_y, 12)))
            in_reasonable_margin = (0.005 * min_side <= min(margin_x, margin_y) <= 0.15 * min_side)
            margin_score = float(np.clip(1.0 - margin_diff, 0.0, 1.0)) if in_reasonable_margin else 0.25

            score = 0.40 * edge_score + 0.35 * sym_score + 0.25 * margin_score
            if cname == "bottom-right":
                score = min(1.0, score * 1.03)

            cx, cy = x0 + centroids[lbl][0], y0 + centroids[lbl][1]
            candidates.append(
                Candidate(
                    method="B",
                    x=int(fx),
                    y=int(fy),
                    w=int(bw),
                    h=int(bh),
                    cx=float(cx),
                    cy=float(cy),
                    score=float(score),
                    corner=cname,
                    details={
                        "edge_score": edge_score,
                        "sym_score": sym_score,
                        "margin_score": margin_score,
                        "area": int(area),
                    },
                )
            )

    return _nms_candidates(candidates, top_k=5)


def detect_method_c(
    frames_bgr: list[np.ndarray],
    candidates_a: list[Candidate] | None = None,
) -> list[Candidate]:
    """Method C: Temporal static-overlay detection on sampled video frames (60-100 frames).

    Watermark = low-variance pixels with stable edges inside a high-variance area.
    """
    if len(frames_bgr) < 10:
        return []

    h, w = frames_bgr[0].shape[:2]
    min_side = min(h, w)
    min_box = max(12, MIN_TEMPLATE_PX - 2)
    max_box = max(min_box + 8, int(round(min_side * MAX_TEMPLATE_FRACTION * 1.4)))

    candidates: list[Candidate] = []

    for cname, x0, y0, x1, y1 in _corner_regions(h, w):
        # Stack grayscale and gradient magnitude across sampled frames for this corner
        roi_grays = []
        roi_grads = []
        for fr in frames_bgr:
            g = cv2.cvtColor(fr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            roi_grays.append(g)
            roi_grads.append(_compute_sobel_mag(g))

        stack_gray = np.stack(roi_grays, axis=0)   # (T, rh, rw)
        stack_grad = np.stack(roi_grads, axis=0)   # (T, rh, rw)

        temp_var = np.var(stack_gray, axis=0)
        med_grad = np.median(stack_grad, axis=0)
        grad_var = np.var(stack_grad, axis=0)

        # Watermark: high temporal median gradient, low temporal variance relative to surroundings
        bg_var = float(np.median(temp_var))
        if bg_var < 1e-5:
            # Static background video: temporal variance cannot distinguish static overlay
            continue

        med_g_thresh = max(0.03, float(np.percentile(med_grad, 88)))
        stable_edges = (med_grad > med_g_thresh) & (temp_var < bg_var * 0.65)
        mask = stable_edges.astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for lbl in range(1, num_labels):
            bx, by, bw, bh, area = stats[lbl]
            if bw < min_box or bh < min_box or bw > max_box or bh > max_box:
                continue

            comp = (labels == lbl)
            comp_var = float(temp_var[comp].mean())
            comp_grad = float(med_grad[comp].mean())
            stability = float(np.clip(1.0 - comp_var / (bg_var + 1e-6), 0.0, 1.0))
            edge_strength = float(np.clip(comp_grad / 0.25, 0.0, 1.0))
            score = 0.55 * stability + 0.45 * edge_strength

            fx, fy = x0 + bx, y0 + by
            cx, cy = x0 + centroids[lbl][0], y0 + centroids[lbl][1]

            # Boost/prefer component matching Method A's candidates if provided
            a_match_dist = None
            if candidates_a:
                dists = [float(np.hypot(cx - ca.cx, cy - ca.cy)) for ca in candidates_a]
                a_match_dist = min(dists) if dists else None
                if a_match_dist is not None and a_match_dist <= AGREEMENT_RADIUS_PX * 2.0:
                    score = min(1.0, score * 1.15)

            candidates.append(
                Candidate(
                    method="C",
                    x=int(fx),
                    y=int(fy),
                    w=int(bw),
                    h=int(bh),
                    cx=float(cx),
                    cy=float(cy),
                    score=float(score),
                    corner=cname,
                    details={
                        "stability": stability,
                        "edge_strength": edge_strength,
                        "a_match_dist": a_match_dist,
                        "grad_var": float(grad_var[comp].mean()),
                    },
                )
            )

    return _nms_candidates(candidates, top_k=5)


def combine_detections(
    h: int,
    w: int,
    cand_a: list[Candidate],
    cand_b: list[Candidate],
    cand_c: list[Candidate],
    heatmaps: dict[str, np.ndarray],
    threshold: float = REJECTION_THRESHOLD,
    agreement_px: float = AGREEMENT_RADIUS_PX,
) -> DetectionResult:
    """Combine A, B, and C by spatial agreement within `agreement_px` (default 3 px)."""
    groups: list[tuple[str, list[Candidate]]] = []
    if cand_a:
        groups.append(("A", cand_a))
    if cand_b:
        groups.append(("B", cand_b))
    if cand_c:
        groups.append(("C", cand_c))

    best_pair: tuple[float, Candidate, list[Candidate]] | None = None

    # Check every candidate from one method against candidates from other methods
    for i in range(len(groups)):
        m1, list1 = groups[i]
        for c1 in list1:
            matched: list[Candidate] = [c1]
            methods_hit = {m1}
            for j in range(len(groups)):
                if i == j:
                    continue
                m2, list2 = groups[j]
                close = [
                    c2 for c2 in list2
                    if np.hypot(c1.cx - c2.cx, c1.cy - c2.cy) <= agreement_px
                ]
                if close:
                    best_c2 = max(close, key=lambda c: c.score)
                    matched.append(best_c2)
                    methods_hit.add(m2)

            if len(methods_hit) >= 2:
                avg_score = float(np.mean([c.score for c in matched]))
                # Prefer Method A's tight bounding box if A is in the agreement
                anchor = next((c for c in matched if c.method == "A"), matched[0])
                combined_score = min(1.0, avg_score + 0.05 * (len(methods_hit) - 2))
                merged = Candidate(
                    method="".join(sorted(methods_hit)),
                    x=anchor.x,
                    y=anchor.y,
                    w=anchor.w,
                    h=anchor.h,
                    cx=float(np.mean([c.cx for c in matched])),
                    cy=float(np.mean([c.cy for c in matched])),
                    score=combined_score,
                    corner=anchor.corner,
                    details={"agreed_methods": sorted(methods_hit), "raw_scores": {c.method: c.score for c in matched}},
                )
                if best_pair is None or merged.score > best_pair[0]:
                    best_pair = (merged.score, merged, matched)

    if best_pair is not None and best_pair[1].score >= threshold:
        chosen = best_pair[1]
        mask = np.zeros((h, w), dtype=np.uint8)
        tpl, _, _ = generate_sparkle_template(max(chosen.w, chosen.h), soft_edge=False)
        th, tw = tpl.shape
        x0 = max(0, min(w - tw, int(round(chosen.cx - tw / 2.0))))
        y0 = max(0, min(h - th, int(round(chosen.cy - th / 2.0))))
        mask[y0:y0 + th, x0:x0 + tw] = (tpl > 0.25).astype(np.uint8) * 255
        return DetectionResult(
            accepted=True,
            reason=f"Accepted: methods {chosen.method} agreed within {agreement_px:.1f} px (score={chosen.score:.3f}).",
            chosen=chosen,
            candidates_a=cand_a,
            candidates_b=cand_b,
            candidates_c=cand_c,
            heatmaps=heatmaps,
            mask_guess=mask,
        )

    if best_pair is not None:
        reason = (
            f"Low confidence: methods {best_pair[1].method} agreed but score "
            f"{best_pair[1].score:.3f} < threshold {threshold:.3f}. Manual selection/brush required."
        )
    else:
        reason = "Disagreement: no two detection methods agreed within 3 px. Manual selection/brush required."

    return DetectionResult(
        accepted=False,
        reason=reason,
        chosen=None,
        candidates_a=cand_a,
        candidates_b=cand_b,
        candidates_c=cand_c,
        heatmaps=heatmaps,
        mask_guess=None,
    )


def detect_image(
    image_bgr: np.ndarray,
    threshold: float = REJECTION_THRESHOLD,
    search_full_frame: bool = True,
) -> DetectionResult:
    """Run Method A + Method B on a single image and combine by agreement."""
    h, w = image_bgr.shape[:2]
    cand_a, heatmaps = detect_method_a(image_bgr, search_full_frame=search_full_frame)
    cand_b = detect_method_b(image_bgr)
    return combine_detections(h, w, cand_a, cand_b, [], heatmaps, threshold=threshold)


def detect_video_frames(
    frames_bgr: list[np.ndarray],
    threshold: float = REJECTION_THRESHOLD,
) -> DetectionResult:
    """Run Method A (on temporal median frame), Method B (on median frame), and Method C (temporal)."""
    if not frames_bgr:
        raise ValueError("No frames provided for video detection.")
    h, w = frames_bgr[0].shape[:2]
    # Temporal median frame suppresses moving background while keeping static watermark sharp
    sample_for_med = frames_bgr[:: max(1, len(frames_bgr) // 25)]
    median_frame = np.median(np.stack(sample_for_med, axis=0), axis=0).astype(np.uint8)

    cand_a, heatmaps = detect_method_a(median_frame, search_full_frame=True)
    cand_b = detect_method_b(median_frame)
    cand_c = detect_method_c(frames_bgr, candidates_a=cand_a)
    return combine_detections(h, w, cand_a, cand_b, cand_c, heatmaps, threshold=threshold)


def profile_path_for(width: int, height: int) -> Path:
    """Return profile path profiles/<width>x<height>.npz."""
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    return PROFILES_DIR / f"{width}x{height}.npz"


def save_profile(
    width: int,
    height: int,
    mask: np.ndarray,
    candidate: Candidate | None = None,
) -> Path:
    """Save confirmed mask & bounding box to profiles/<width>x<height>.npz."""
    path = profile_path_for(width, height)
    bbox = np.array(
        [candidate.x, candidate.y, candidate.w, candidate.h] if candidate else [0, 0, 0, 0],
        dtype=np.int32,
    )
    score = np.array([candidate.score if candidate else 1.0], dtype=np.float32)
    np.savez_compressed(path, mask=(mask > 0).astype(np.uint8) * 255, bbox=bbox, score=score)
    return path


def load_profile(width: int, height: int) -> tuple[np.ndarray, tuple[int, int, int, int], float] | None:
    """Load saved profile for <width>x<height> if it exists."""
    path = profile_path_for(width, height)
    if not path.exists():
        return None
    data = np.load(path)
    mask = data["mask"].astype(np.uint8)
    bbox = tuple(int(v) for v in data["bbox"])
    score = float(data["score"][0])
    return mask, bbox, score
