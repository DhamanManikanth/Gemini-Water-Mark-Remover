"""End-to-end pipelines and debug visualization helpers for AutoMarkRemover."""

from __future__ import annotations

from pathlib import Path
import zipfile

import cv2
import numpy as np

from .config import (
    DEFAULT_DILATION_PX,
    DEFAULT_FEATHER_PX,
    OUTPUT_DIR,
    REJECTION_THRESHOLD,
    ensure_dirs,
)
from .detect import (
    Candidate,
    DetectionResult,
    detect_image,
    detect_video_frames,
    load_profile,
    save_profile,
)
from .inpainter_lama import inpaint_image_lama
from .inpainter_propainter import remove_watermark_video_frames
from .mask_refine import finalize_mask, refine_mask_from_candidate
from .verify import VerificationReport, verify_image, verify_video
from .video_io import (
    encode_video_ffmpeg,
    probe_video,
    read_all_frames,
    sample_video_frames,
)


def render_candidates_overlay(image_bgr: np.ndarray, det: DetectionResult) -> np.ndarray:
    """Draw top candidates from Methods A, B, C onto a copy of `image_bgr`."""
    vis = image_bgr.copy()
    colors = {"A": (0, 220, 255), "B": (255, 160, 0), "C": (255, 0, 220)}

    for label, cands in (("A", det.candidates_a[:3]), ("B", det.candidates_b[:3]), ("C", det.candidates_c[:3])):
        col = colors[label]
        for idx, c in enumerate(cands, 1):
            cv2.rectangle(vis, (c.x, c.y), (c.x + c.w, c.y + c.h), col, 2)
            cv2.putText(
                vis,
                f"{label}{idx}:{c.score:.2f}",
                (c.x, max(14, c.y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                col,
                1,
                cv2.LINE_AA,
            )

    if det.chosen is not None:
        c = det.chosen
        cv2.rectangle(vis, (c.x - 2, c.y - 2), (c.x + c.w + 2, c.y + c.h + 2), (0, 255, 0), 2)
    return vis


def render_mask_overlay(image_bgr: np.ndarray, hard_mask: np.ndarray) -> np.ndarray:
    """Overlay `hard_mask` in translucent red with green contour on `image_bgr`."""
    vis = image_bgr.copy()
    m = hard_mask > 0
    if m.any():
        overlay = vis.copy()
        overlay[m] = (0, 0, 255)
        vis = cv2.addWeighted(overlay, 0.45, vis, 0.55, 0)
        contours, _ = cv2.findContours(hard_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 255, 0), 1)
    return vis


def render_heatmaps_panel(heatmaps: dict[str, np.ndarray]) -> np.ndarray | None:
    """Combine available Method A heatmaps into a single colorized diagnostic image."""
    if not heatmaps:
        return None
    tiles = []
    for name in ("bottom-right", "bottom-left", "top-right", "top-left", "full"):
        hm = heatmaps.get(name)
        if hm is None or hm.size == 0:
            continue
        norm = np.clip(hm * 255.0, 0, 255).astype(np.uint8)
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
        resized = cv2.resize(colored, (240, 180), interpolation=cv2.INTER_AREA)
        cv2.putText(resized, name, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(resized)
    if not tiles:
        return None
    return np.hstack(tiles)


def format_candidates_table(det: DetectionResult) -> str:
    """Format top candidates with score, corner, and position as Markdown."""
    lines = [
        f"**Status:** {det.reason}",
        "",
        "| Method | Rank | Corner | Position (x, y, w, h) | Center (cx, cy) | Score |",
        "|---|---:|---|---|---|---:|",
    ]
    for m, cands in (("A", det.candidates_a), ("B", det.candidates_b), ("C", det.candidates_c)):
        for i, c in enumerate(cands[:5], 1):
            lines.append(
                f"| {m} | {i} | {c.corner} | ({c.x}, {c.y}, {c.w}, {c.h}) | "
                f"({c.cx:.1f}, {c.cy:.1f}) | {c.score:.3f} |"
            )
    return "\n".join(lines)


def build_image_mask(
    image_bgr: np.ndarray,
    threshold: float = REJECTION_THRESHOLD,
    dilation_px: int = DEFAULT_DILATION_PX,
    feather_px: int = DEFAULT_FEATHER_PX,
    use_profile: bool = True,
) -> tuple[DetectionResult | None, np.ndarray | None, np.ndarray | None, str]:
    """Check resolution profile or run detection + mask refinement on an image."""
    h, w = image_bgr.shape[:2]
    if use_profile:
        prof = load_profile(w, h)
        if prof is not None:
            prof_mask, bbox, score = prof
            hard_m, soft_m = finalize_mask(prof_mask, dilation_px=dilation_px, feather_px=feather_px)
            return None, hard_m, soft_m, f"Loaded saved profile {w}x{h}.npz (bbox={bbox}, score={score:.3f}). Review overlay before running."

    det = detect_image(image_bgr, threshold=threshold)
    if det.accepted and det.chosen is not None:
        hard_m, soft_m = refine_mask_from_candidate(
            image_bgr, det.chosen, initial_mask=det.mask_guess, dilation_px=dilation_px, feather_px=feather_px
        )
        return det, hard_m, soft_m, det.reason
    return det, None, None, det.reason


def run_image_removal(
    image_bgr: np.ndarray,
    hard_mask: np.ndarray,
    feather_px: int = DEFAULT_FEATHER_PX,
    stem: str = "image",
    save_confirmed_profile: bool = True,
    candidate: Candidate | None = None,
) -> tuple[np.ndarray, VerificationReport, Path]:
    """Remove watermark from a single image with LaMa, verify, retry if needed, and save lossless PNG."""
    ensure_dirs()
    h, w = image_bgr.shape[:2]
    hard_m, soft_m = finalize_mask(hard_mask, dilation_px=0, feather_px=feather_px)
    out = inpaint_image_lama(image_bgr, hard_m, soft_m)
    rep = verify_image(image_bgr, out, hard_m, stem=stem)

    if not rep.passed and rep.outside_diff_zero:
        hard_m, soft_m = finalize_mask(hard_m, dilation_px=2, feather_px=feather_px)
        out = inpaint_image_lama(image_bgr, hard_m, soft_m)
        rep = verify_image(image_bgr, out, hard_m, stem=f"{stem}_retry")

    if not rep.outside_diff_zero:
        raise RuntimeError("Integrity check failed: pixels outside mask were modified!")

    if save_confirmed_profile:
        save_profile(w, h, hard_m, candidate=candidate)

    out_path = OUTPUT_DIR / f"{stem}_clean.png"
    cv2.imwrite(str(out_path), out)
    return out, rep, out_path


def run_video_removal(
    video_path: str | Path,
    hard_mask: np.ndarray,
    preview_seconds: float | None = None,
    stem: str = "video",
) -> tuple[Path, VerificationReport]:
    """Remove watermark from video (full or 3-second preview) and encode H.264 CRF 14."""
    ensure_dirs()
    meta = probe_video(video_path)
    max_frames = int(round(preview_seconds * meta.fps)) if preview_seconds else None
    frames = read_all_frames(video_path, max_frames=max_frames)
    if not frames:
        raise RuntimeError("No frames decoded from video.")

    cleaned = remove_watermark_video_frames(frames, hard_mask)
    rep = verify_video(frames, cleaned, hard_mask, stem=stem)
    if not rep.outside_diff_zero:
        raise RuntimeError("Integrity check failed: video pixels outside mask were modified!")

    suffix = f"_preview_{int(preview_seconds)}s" if preview_seconds else "_clean"
    out_path = OUTPUT_DIR / f"{stem}{suffix}.mp4"
    encode_video_ffmpeg(cleaned, video_path, out_path, meta)
    return out_path, rep


def run_batch_zip(
    file_paths: list[str],
    threshold: float = REJECTION_THRESHOLD,
    dilation_px: int = DEFAULT_DILATION_PX,
    feather_px: int = DEFAULT_FEATHER_PX,
) -> tuple[Path, str]:
    """Process a batch of images/videos and package outputs into a ZIP archive."""
    ensure_dirs()
    out_files: list[Path] = []
    logs: list[str] = []

    video_exts = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
    for fp in file_paths:
        p = Path(fp)
        stem = p.stem
        try:
            if p.suffix.lower() in video_exts:
                sampled = sample_video_frames(p, num_samples=75)
                det = detect_video_frames(sampled, threshold=threshold)
                if not det.accepted or det.chosen is None:
                    logs.append(f"[SKIP] {p.name}: {det.reason}")
                    continue
                med = np.median(np.stack(sampled[::3], axis=0), axis=0).astype(np.uint8)
                hard_m, _ = refine_mask_from_candidate(med, det.chosen, det.mask_guess, dilation_px, feather_px)
                out_p, rep = run_video_removal(p, hard_m, preview_seconds=None, stem=stem)
                out_files.append(out_p)
                logs.append(f"[OK] {p.name} -> {out_p.name} (outside_zero={rep.outside_diff_zero})")
            else:
                img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if img is None:
                    logs.append(f"[SKIP] {p.name}: unreadable image")
                    continue
                det, hard_m, _, msg = build_image_mask(img, threshold, dilation_px, feather_px, use_profile=True)
                if hard_m is None:
                    logs.append(f"[SKIP] {p.name}: {msg}")
                    continue
                _, rep, out_p = run_image_removal(
                    img, hard_m, feather_px=feather_px, stem=stem, candidate=det.chosen if det else None
                )
                out_files.append(out_p)
                logs.append(f"[OK] {p.name} -> {out_p.name} (outside_zero={rep.outside_diff_zero})")
        except Exception as exc:
            logs.append(f"[ERROR] {p.name}: {exc}")

    zip_path = OUTPUT_DIR / "automark_batch_output.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for f in out_files:
            zf.write(f, arcname=f.name)

    return zip_path, "\n".join(logs)
