"""Stage 2 & 3 tests: Mask refinement, LaMa removal, outside-mask zero diff, re-detection, edge energy."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from automark.detect import detect_image
from automark.inpainter_lama import inpaint_image_lama
from automark.mask_refine import finalize_mask, refine_mask_from_candidate
from automark.verify import verify_image
from tests.synth_data import make_textured_background, overlay_synthetic_sparkle


def run_stage2_tests() -> bool:
    print("=== STAGE 2 & 3 IMAGE REMOVAL & VERIFICATION TESTS ===")
    all_passed = True

    cases = [
        ("bottom-right", 32, 0.55, 512, 768, 701),
        ("top-left", 26, 0.65, 480, 640, 702),
    ]

    for idx, (corner, size, alpha, h, w, seed) in enumerate(cases, 1):
        bg = make_textured_background(h, w, seed=seed)
        img, gt_cx, gt_cy, _ = overlay_synthetic_sparkle(bg, corner=corner, size=size, alpha=alpha)

        det = detect_image(img)
        if not det.accepted or det.chosen is None:
            print(f"[FAIL] Case {idx}: detection failed: {det.reason}")
            all_passed = False
            continue

        hard_mask, soft_mask = refine_mask_from_candidate(
            img, det.chosen, initial_mask=det.mask_guess, dilation_px=2, feather_px=1
        )
        out = inpaint_image_lama(img, hard_mask, soft_mask)
        rep = verify_image(img, out, hard_mask, stem=f"test_img_{idx}")

        # Retry once with enlarged mask (+2 px dilation) if re-detection failed
        if not rep.passed and rep.outside_diff_zero:
            print(f"  Case {idx} retrying with +2px mask dilation...")
            hard_mask, soft_mask = finalize_mask(hard_mask, dilation_px=2, feather_px=1)
            out = inpaint_image_lama(img, hard_mask, soft_mask)
            rep = verify_image(img, out, hard_mask, stem=f"test_img_{idx}_retry")

        status = "PASS" if rep.passed else "FAIL"
        print(
            f"[{status}] Case {idx} ({corner}, {size}px, alpha={alpha}): "
            f"outside_zero={rep.outside_diff_zero}, redet_max={rep.max_redetect_score:.3f}, "
            f"edge_inner={rep.inner_edge_energy:.4f}, edge_ring={rep.ring_edge_energy:.4f}, "
            f"ratio={rep.edge_ratio:.2f}, review={rep.review_crop_path}"
        )
        if not rep.passed:
            all_passed = False

    print(f"=== STAGE 2 & 3 RESULT: {'ALL PASSED' if all_passed else 'FAILED'} ===")
    return all_passed


if __name__ == "__main__":
    ok = run_stage2_tests()
    sys.exit(0 if ok else 1)
