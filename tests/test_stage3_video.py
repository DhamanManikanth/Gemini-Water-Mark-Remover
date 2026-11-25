"""Stage 4 video tests: temporal detection, alpha/logo recovery, flicker check, ffmpeg CRF 14 encode."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from automark.config import OUTPUT_DIR, ensure_dirs
from automark.detect import detect_video_frames
from automark.inpainter_propainter import remove_watermark_video_frames
from automark.mask_refine import refine_mask_from_candidate
from automark.verify import verify_video
from automark.video_io import (
    encode_video_ffmpeg,
    probe_video,
    read_all_frames,
    require_ffmpeg,
    sample_video_frames,
)
from tests.synth_data import make_moving_background_clip


def run_stage3_tests() -> bool:
    print("=== STAGE 4 VIDEO PIPELINE TESTS ===")
    ensure_dirs()
    ffmpeg, _ = require_ffmpeg()

    wm_frames, _, gt_cx, gt_cy, _ = make_moving_background_clip(
        n_frames=70, h=360, w=640, corner="bottom-right", size=32, alpha=0.50, seed=909
    )

    # Write synthetic input MP4 via ffmpeg
    in_mp4 = OUTPUT_DIR / "test_synth_input.mp4"
    out_mp4 = OUTPUT_DIR / "test_synth_clean.mp4"
    cmd = [
        ffmpeg, "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "bgr24",
        "-s", "640x360", "-r", "30/1", "-i", "-",
        "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
        str(in_mp4),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    for fr in wm_frames:
        proc.stdin.write(fr.tobytes())
    proc.stdin.close()
    if proc.stderr:
        proc.stderr.read()
    proc.wait()

    meta = probe_video(in_mp4)
    sampled = sample_video_frames(in_mp4, num_samples=70)
    det = detect_video_frames(sampled)
    if not det.accepted or det.chosen is None:
        print(f"[FAIL] Video detection failed: {det.reason}")
        return False

    err = float(np.hypot(det.chosen.cx - gt_cx, det.chosen.cy - gt_cy))
    print(f"[PASS] Video detection: gt=({gt_cx:.1f},{gt_cy:.1f}) det=({det.chosen.cx:.1f},{det.chosen.cy:.1f}) err={err:.2f}px score={det.chosen.score:.3f} methods={det.chosen.method}")
    if err > 3.0:
        return False

    all_frames = read_all_frames(in_mp4)
    med_frame = np.median(np.stack(sampled[::3], axis=0), axis=0).astype(np.uint8)
    hard_mask, _ = refine_mask_from_candidate(med_frame, det.chosen, initial_mask=det.mask_guess, dilation_px=2, feather_px=1)

    cleaned_frames = remove_watermark_video_frames(all_frames, hard_mask)
    rep = verify_video(all_frames, cleaned_frames, hard_mask, stem="test_video")

    print(
        f"[{'PASS' if rep.passed else 'FAIL'}] Video verify: outside_zero={rep.outside_diff_zero}, "
        f"redet_max={rep.max_redetect_score:.3f}, edge_ratio={rep.edge_ratio:.2f}, "
        f"flicker_ratio={rep.flicker_ratio:.2f}, review={rep.review_crop_path}"
    )
    if not rep.passed:
        return False

    encode_video_ffmpeg(cleaned_frames, in_mp4, out_mp4, meta)
    out_meta = probe_video(out_mp4)
    meta_ok = (out_meta.width == meta.width and out_meta.height == meta.height and out_meta.pix_fmt == meta.pix_fmt)
    print(f"[{'PASS' if meta_ok else 'FAIL'}] Output MP4 metadata: {out_meta.width}x{out_meta.height} {out_meta.fps:.2f}fps {out_meta.pix_fmt} -> {out_mp4}")

    all_ok = rep.passed and meta_ok and err <= 3.0
    print(f"=== STAGE 4 RESULT: {'ALL PASSED' if all_ok else 'FAILED'} ===")
    return all_ok


if __name__ == "__main__":
    ok = run_stage3_tests()
    sys.exit(0 if ok else 1)
