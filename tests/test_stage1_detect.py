"""Stage 1 tests: Method A, Method B, Method C, and 3 px agreement."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from automark.detect import detect_image, detect_video_frames
from tests.synth_data import (
    make_moving_background_clip,
    make_textured_background,
    overlay_synthetic_sparkle,
)


def run_stage1_tests() -> bool:
    print("=== STAGE 1 DETECTION TESTS ===")
    all_passed = True

    image_cases = [
        ("bottom-right", 28, 0.50, 540, 960, 101),
        ("top-left", 36, 0.45, 480, 640, 202),
        ("bottom-left", 24, 0.55, 600, 800, 303),
        ("top-right", 42, 0.42, 720, 1280, 404),
    ]

    for idx, (corner, size, alpha, h, w, seed) in enumerate(image_cases, 1):
        bg = make_textured_background(h, w, seed=seed)
        img, gt_cx, gt_cy, _ = overlay_synthetic_sparkle(bg, corner=corner, size=size, alpha=alpha)
        res = detect_image(img)
        if res.chosen is None or not res.accepted:
            print(f"[FAIL] Image case {idx} ({corner}, {size}px): {res.reason}")
            print(f"  Top A: {[(round(c.cx,1), round(c.cy,1), round(c.score,3), c.w) for c in res.candidates_a[:3]]}")
            print(f"  Top B: {[(round(c.cx,1), round(c.cy,1), round(c.score,3), c.w) for c in res.candidates_b[:3]]}")
            all_passed = False
        else:
            err = float(np.hypot(res.chosen.cx - gt_cx, res.chosen.cy - gt_cy))
            ok = err <= 3.0
            status = "PASS" if ok else "FAIL"
            print(
                f"[{status}] Image case {idx} ({corner}, {size}px, alpha={alpha}): "
                f"gt=({gt_cx:.1f},{gt_cy:.1f}) det=({res.chosen.cx:.1f},{res.chosen.cy:.1f}) "
                f"err={err:.2f}px score={res.chosen.score:.3f} methods={res.chosen.method}"
            )
            if not ok:
                all_passed = False

    # Video moving-background test
    wm_frames, _, gt_cx, gt_cy, _ = make_moving_background_clip(
        n_frames=65, h=360, w=640, corner="bottom-left", size=30, alpha=0.48, seed=505
    )
    vres = detect_video_frames(wm_frames)
    if vres.chosen is None or not vres.accepted:
        print(f"[FAIL] Video case (bottom-left, 30px): {vres.reason}")
        print(f"  Top A: {[(round(c.cx,1), round(c.cy,1), round(c.score,3)) for c in vres.candidates_a[:3]]}")
        print(f"  Top B: {[(round(c.cx,1), round(c.cy,1), round(c.score,3)) for c in vres.candidates_b[:3]]}")
        print(f"  Top C: {[(round(c.cx,1), round(c.cy,1), round(c.score,3)) for c in vres.candidates_c[:3]]}")
        all_passed = False
    else:
        err = float(np.hypot(vres.chosen.cx - gt_cx, vres.chosen.cy - gt_cy))
        ok = err <= 3.0
        status = "PASS" if ok else "FAIL"
        print(
            f"[{status}] Video case (bottom-left, 30px): "
            f"gt=({gt_cx:.1f},{gt_cy:.1f}) det=({vres.chosen.cx:.1f},{vres.chosen.cy:.1f}) "
            f"err={err:.2f}px score={vres.chosen.score:.3f} methods={vres.chosen.method}"
        )
        if not ok:
            all_passed = False

    print(f"=== STAGE 1 RESULT: {'ALL PASSED' if all_passed else 'FAILED'} ===")
    return all_passed


if __name__ == "__main__":
    ok = run_stage1_tests()
    sys.exit(0 if ok else 1)
