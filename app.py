"""AutoMarkRemover Gradio UI (local only, no external uploads)."""

from __future__ import annotations

from pathlib import Path

import cv2
from fastapi import FastAPI
import gradio as gr
import numpy as np
import torch

from automark.config import (
    DEFAULT_DILATION_PX,
    DEFAULT_FEATHER_PX,
    REJECTION_THRESHOLD,
    ensure_dirs,
    find_ffmpeg_binaries,
    get_device,
    get_vram_gb,
)
from automark.detect import (
    detect_video_frames,
    save_profile,
)
from automark.inpainter_propainter import propainter_available
from automark.mask_refine import finalize_mask, refine_mask_from_candidate
from automark.pipeline import (
    build_image_mask,
    format_candidates_table,
    render_candidates_overlay,
    render_heatmaps_panel,
    render_mask_overlay,
    run_batch_zip,
    run_image_removal,
    run_video_removal,
)
from automark.video_io import probe_video, read_all_frames, sample_video_frames

ensure_dirs()


def _system_status_md() -> str:
    dev = get_device()
    vram = get_vram_gb()
    ffmpeg, ffprobe = find_ffmpeg_binaries()
    gpu_str = f"CUDA (`{torch.cuda.get_device_name(0)}`, {vram:.1f} GB VRAM)" if dev.type == "cuda" else "CPU fallback (no CUDA)"
    ff_str = "OK" if (ffmpeg and ffprobe) else "**MISSING** (install FFmpeg for video)"
    pp_str = "Ready" if propainter_available() else "Not cloned (`third_party/ProPainter` missing; alpha recovery still active)"
    return f"**Device:** {gpu_str} | **FFmpeg/FFprobe:** {ff_str} | **ProPainter:** {pp_str}"


def _bgr_to_rgb(img: np.ndarray | None) -> np.ndarray | None:
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _rgb_to_bgr(img: np.ndarray | None) -> np.ndarray | None:
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def _corner_crop_coords(h: int, w: int, corner: str) -> tuple[int, int, int, int]:
    cw = max(64, int(round(w * 0.35)))
    ch = max(64, int(round(h * 0.35)))
    if corner == "Bottom-Right":
        return w - cw, h - ch, w, h
    if corner == "Bottom-Left":
        return 0, h - ch, cw, h
    if corner == "Top-Right":
        return w - cw, 0, w, ch
    if corner == "Top-Left":
        return 0, 0, cw, ch
    return 0, 0, w, h


def _four_corners_grid(image_bgr: np.ndarray) -> np.ndarray:
    """Create a 2x2 zoomed preview of the four corners for manual inspection."""
    h, w = image_bgr.shape[:2]
    cw, ch = max(48, int(round(w * 0.28))), max(48, int(round(h * 0.28)))
    tl = cv2.resize(image_bgr[0:ch, 0:cw], (240, 180))
    tr = cv2.resize(image_bgr[0:ch, w - cw:w], (240, 180))
    bl = cv2.resize(image_bgr[h - ch:h, 0:cw], (240, 180))
    br = cv2.resize(image_bgr[h - ch:h, w - cw:w], (240, 180))
    for tile, lbl in ((tl, "Top-Left"), (tr, "Top-Right"), (bl, "Bottom-Left"), (br, "Bottom-Right")):
        cv2.putText(tile, lbl, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([np.hstack([tl, tr]), np.hstack([bl, br])])


# ---------------- IMAGE CALLBACKS ----------------

def img_auto_detect(img_rgb, threshold, dilation, feather):
    if img_rgb is None:
        raise gr.Error("Please upload an image first.")
    bgr = _rgb_to_bgr(img_rgb)
    det, hard_m, _, msg = build_image_mask(
        bgr, threshold=threshold, dilation_px=int(dilation), feather_px=int(feather), use_profile=True
    )
    overlay = render_mask_overlay(bgr, hard_m) if hard_m is not None else (
        render_candidates_overlay(bgr, det) if det is not None else bgr
    )
    heatmap = render_heatmaps_panel(det.heatmaps) if det is not None else None
    table = format_candidates_table(det) if det is not None else msg
    state = {"bgr": bgr, "mask": hard_m, "det": det, "crop": (0, 0, bgr.shape[1], bgr.shape[0])}
    return _bgr_to_rgb(overlay), _bgr_to_rgb(heatmap), table, msg, state


def img_open_paint(img_rgb, corner_choice, state):
    if img_rgb is None:
        raise gr.Error("Please upload an image first.")
    bgr = _rgb_to_bgr(img_rgb)
    h, w = bgr.shape[:2]
    x0, y0, x1, y1 = _corner_crop_coords(h, w, corner_choice)
    crop_rgb = _bgr_to_rgb(bgr[y0:y1, x0:x1])
    if state is None:
        state = {"bgr": bgr, "mask": None, "det": None}
    state["bgr"] = bgr
    state["crop"] = (x0, y0, x1, y1)
    corners_preview = _bgr_to_rgb(_four_corners_grid(bgr))
    return {"background": crop_rgb, "layers": [], "composite": crop_rgb}, corners_preview, state, f"Painting on {corner_choice} ({x0},{y0})-({x1},{y1})."


def img_apply_paint(editor_val, dilation, feather, state):
    if state is None or state.get("bgr") is None:
        raise gr.Error("Upload an image and click 'Paint mask' first.")
    if not editor_val or not editor_val.get("layers"):
        raise gr.Error("Draw on the corner canvas before applying the painted mask.")
    bgr = state["bgr"]
    h, w = bgr.shape[:2]
    x0, y0, x1, y1 = state.get("crop", (0, 0, w, h))

    # Extract alpha / non-zero strokes from all layers
    layer_acc = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    for layer in editor_val["layers"]:
        arr = np.array(layer)
        if arr.ndim == 3 and arr.shape[2] == 4:
            m = arr[:, :, 3] > 10
        elif arr.ndim == 3:
            m = arr.max(axis=2) > 10
        else:
            m = arr > 10
        if m.shape[:2] != layer_acc.shape:
            m = cv2.resize(m.astype(np.uint8), (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST) > 0
        layer_acc[m] = 255

    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[y0:y1, x0:x1] = layer_acc
    hard_m, _ = finalize_mask(full_mask, dilation_px=int(dilation), feather_px=int(feather))
    state["mask"] = hard_m
    overlay = render_mask_overlay(bgr, hard_m)
    return _bgr_to_rgb(overlay), state, "Manual mask applied. Review overlay and click 'Run full'."


def img_run_full(feather, state):
    if state is None or state.get("bgr") is None:
        raise gr.Error("Please upload an image and detect or paint a mask first.")
    if state.get("mask") is None or not (state["mask"] > 0).any():
        raise gr.Error("No mask confirmed yet. Click 'Auto-detect' or 'Paint mask' first.")
    bgr = state["bgr"]
    det = state.get("det")
    cand = det.chosen if det is not None else None
    out, rep, out_path = run_image_removal(
        bgr, state["mask"], feather_px=int(feather), stem="gradio_img", save_confirmed_profile=True, candidate=cand
    )
    review_img = cv2.imread(str(rep.review_crop_path)) if rep.review_crop_path else None
    status = " | ".join(rep.messages) + f" | Saved: {out_path}"
    return _bgr_to_rgb(out), str(out_path), _bgr_to_rgb(review_img), status


# ---------------- VIDEO CALLBACKS ----------------

def vid_auto_detect(video_path, threshold, dilation, feather):
    if not video_path:
        raise gr.Error("Please upload a video first.")
    try:
        meta = probe_video(video_path)
    except Exception as exc:
        raise gr.Error(str(exc))
    sampled = sample_video_frames(video_path, num_samples=75)
    det = detect_video_frames(sampled, threshold=threshold)
    med = np.median(np.stack(sampled[:: max(1, len(sampled) // 25)], axis=0), axis=0).astype(np.uint8)

    hard_m = None
    if det.accepted and det.chosen is not None:
        hard_m, _ = refine_mask_from_candidate(
            med, det.chosen, initial_mask=det.mask_guess, dilation_px=int(dilation), feather_px=int(feather)
        )
    overlay = render_mask_overlay(med, hard_m) if hard_m is not None else render_candidates_overlay(med, det)
    heatmap = render_heatmaps_panel(det.heatmaps)
    table = format_candidates_table(det)
    vstate = {"video_path": video_path, "med": med, "mask": hard_m, "det": det, "meta": meta, "crop": (0, 0, med.shape[1], med.shape[0])}
    return _bgr_to_rgb(overlay), _bgr_to_rgb(heatmap), table, det.reason, vstate


def vid_open_paint(video_path, corner_choice, vstate):
    if not video_path:
        raise gr.Error("Please upload a video first.")
    if vstate is None or vstate.get("med") is None:
        sampled = sample_video_frames(video_path, num_samples=25)
        med = np.median(np.stack(sampled, axis=0), axis=0).astype(np.uint8)
        vstate = {"video_path": video_path, "med": med, "mask": None, "det": None}
    med = vstate["med"]
    h, w = med.shape[:2]
    x0, y0, x1, y1 = _corner_crop_coords(h, w, corner_choice)
    vstate["crop"] = (x0, y0, x1, y1)
    crop_rgb = _bgr_to_rgb(med[y0:y1, x0:x1])
    return {"background": crop_rgb, "layers": [], "composite": crop_rgb}, vstate


def vid_apply_paint(editor_val, dilation, feather, vstate):
    if vstate is None or vstate.get("med") is None:
        raise gr.Error("Upload a video and click 'Paint mask' first.")
    if not editor_val or not editor_val.get("layers"):
        raise gr.Error("Draw on the video corner canvas before applying.")
    med = vstate["med"]
    h, w = med.shape[:2]
    x0, y0, x1, y1 = vstate.get("crop", (0, 0, w, h))
    layer_acc = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    for layer in editor_val["layers"]:
        arr = np.array(layer)
        m = (arr[:, :, 3] > 10) if (arr.ndim == 3 and arr.shape[2] == 4) else (arr.max(axis=2) > 10 if arr.ndim == 3 else arr > 10)
        if m.shape[:2] != layer_acc.shape:
            m = cv2.resize(m.astype(np.uint8), (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST) > 0
        layer_acc[m] = 255
    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[y0:y1, x0:x1] = layer_acc
    hard_m, _ = finalize_mask(full_mask, dilation_px=int(dilation), feather_px=int(feather))
    vstate["mask"] = hard_m
    return _bgr_to_rgb(render_mask_overlay(med, hard_m)), vstate, "Manual video mask applied."


def vid_test_one_frame(feather, vstate):
    if vstate is None or vstate.get("mask") is None:
        raise gr.Error("Detect or paint a video mask first.")
    frames = read_all_frames(vstate["video_path"], max_frames=1)
    if not frames:
        raise gr.Error("Could not decode frame from video.")
    out, rep, _ = run_image_removal(frames[0], vstate["mask"], feather_px=int(feather), stem="vid_frame0", save_confirmed_profile=False)
    review_img = cv2.imread(str(rep.review_crop_path)) if rep.review_crop_path else None
    return _bgr_to_rgb(review_img), "One-frame test: " + " | ".join(rep.messages)


def vid_preview_3s(vstate):
    if vstate is None or vstate.get("mask") is None:
        raise gr.Error("Detect or paint a video mask first.")
    out_path, rep = run_video_removal(vstate["video_path"], vstate["mask"], preview_seconds=3.0, stem=Path(vstate["video_path"]).stem)
    review_img = cv2.imread(str(rep.review_crop_path)) if rep.review_crop_path else None
    return str(out_path), _bgr_to_rgb(review_img), "3s Preview: " + " | ".join(rep.messages)


def vid_run_full(vstate):
    if vstate is None or vstate.get("mask") is None:
        raise gr.Error("Detect or paint a video mask first.")
    med = vstate["med"]
    h, w = med.shape[:2]
    det = vstate.get("det")
    save_profile(w, h, vstate["mask"], candidate=det.chosen if det else None)
    out_path, rep = run_video_removal(vstate["video_path"], vstate["mask"], preview_seconds=None, stem=Path(vstate["video_path"]).stem)
    review_img = cv2.imread(str(rep.review_crop_path)) if rep.review_crop_path else None
    return str(out_path), _bgr_to_rgb(review_img), "Full Video: " + " | ".join(rep.messages)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="AutoMarkRemover") as demo:
        gr.Markdown("# AutoMarkRemover\nLocal AI-watermark detector & remover (LaMa + ProPainter, bit-identical outside mask).")
        gr.Markdown(_system_status_md())

        with gr.Row():
            dilation_sl = gr.Slider(0, 6, value=DEFAULT_DILATION_PX, step=1, label="Mask dilation (px)")
            feather_sl = gr.Slider(0, 5, value=DEFAULT_FEATHER_PX, step=1, label="Mask feather (px)")
            thresh_sl = gr.Slider(0.20, 0.80, value=REJECTION_THRESHOLD, step=0.02, label="Detection threshold")

        with gr.Tabs():
            # ---------- IMAGE TAB ----------
            with gr.Tab("Image"):
                img_state = gr.State(None)
                with gr.Row():
                    img_in = gr.Image(type="numpy", label="Upload Image")
                    img_overlay = gr.Image(type="numpy", label="Mask / Candidate Overlay")
                    img_out = gr.Image(type="numpy", label="Cleaned Output (Lossless PNG)")

                with gr.Row():
                    btn_img_detect = gr.Button("Auto-detect", variant="primary")
                    corner_dd = gr.Dropdown(["Bottom-Right", "Bottom-Left", "Top-Right", "Top-Left", "Full"], value="Bottom-Right", label="Zoom corner for Paint")
                    btn_img_paint = gr.Button("Paint mask")
                    btn_img_apply_paint = gr.Button("Apply painted mask")
                    btn_img_run = gr.Button("Run full", variant="primary")

                img_status = gr.Textbox(label="Status", interactive=False)
                with gr.Row():
                    img_editor = gr.ImageEditor(type="numpy", label="Manual Brush (Method D: brush, eraser, zoom, undo)")
                    img_corners_grid = gr.Image(type="numpy", label="4 Corners Zoomed")

                with gr.Accordion("Debug Panel (Candidates, Heatmaps, Review Crop)", open=True):
                    img_cand_md = gr.Markdown("Run Auto-detect to view candidates.")
                    with gr.Row():
                        img_heatmap = gr.Image(type="numpy", label="Match Heatmaps (Method A)")
                        img_review = gr.Image(type="numpy", label="Enlarged Before / After / Mask Review")
                    img_dl = gr.File(label="Download Lossless PNG")

                btn_img_detect.click(
                    img_auto_detect,
                    inputs=[img_in, thresh_sl, dilation_sl, feather_sl],
                    outputs=[img_overlay, img_heatmap, img_cand_md, img_status, img_state],
                )
                btn_img_paint.click(
                    img_open_paint,
                    inputs=[img_in, corner_dd, img_state],
                    outputs=[img_editor, img_corners_grid, img_state, img_status],
                )
                btn_img_apply_paint.click(
                    img_apply_paint,
                    inputs=[img_editor, dilation_sl, feather_sl, img_state],
                    outputs=[img_overlay, img_state, img_status],
                )
                btn_img_run.click(
                    img_run_full,
                    inputs=[feather_sl, img_state],
                    outputs=[img_out, img_dl, img_review, img_status],
                )

            # ---------- VIDEO TAB ----------
            with gr.Tab("Video"):
                vid_state = gr.State(None)
                with gr.Row():
                    vid_in = gr.Video(label="Upload Video")
                    vid_overlay = gr.Image(type="numpy", label="Mask / Candidate Overlay (Median Frame)")
                    vid_out = gr.Video(label="Cleaned Video Output (H.264 CRF 14)")

                with gr.Row():
                    btn_vid_detect = gr.Button("Auto-detect", variant="primary")
                    vid_corner_dd = gr.Dropdown(["Bottom-Right", "Bottom-Left", "Top-Right", "Top-Left", "Full"], value="Bottom-Right", label="Zoom corner")
                    btn_vid_paint = gr.Button("Paint mask")
                    btn_vid_apply_paint = gr.Button("Apply painted mask")
                    btn_vid_one = gr.Button("Test on one frame")
                    btn_vid_3s = gr.Button("3-second preview")
                    btn_vid_run = gr.Button("Run full", variant="primary")

                vid_status = gr.Textbox(label="Video Status", interactive=False)
                vid_editor = gr.ImageEditor(type="numpy", label="Manual Video Brush (zoomed corner)")

                with gr.Accordion("Video Debug Panel", open=True):
                    vid_cand_md = gr.Markdown("Run Auto-detect to view video candidates.")
                    with gr.Row():
                        vid_heatmap = gr.Image(type="numpy", label="Video Match Heatmaps")
                        vid_review = gr.Image(type="numpy", label="Enlarged Before / After / Mask Review")

                btn_vid_detect.click(
                    vid_auto_detect,
                    inputs=[vid_in, thresh_sl, dilation_sl, feather_sl],
                    outputs=[vid_overlay, vid_heatmap, vid_cand_md, vid_status, vid_state],
                )
                btn_vid_paint.click(
                    vid_open_paint,
                    inputs=[vid_in, vid_corner_dd, vid_state],
                    outputs=[vid_editor, vid_state],
                )
                btn_vid_apply_paint.click(
                    vid_apply_paint,
                    inputs=[vid_editor, dilation_sl, feather_sl, vid_state],
                    outputs=[vid_overlay, vid_state, vid_status],
                )
                btn_vid_one.click(
                    vid_test_one_frame,
                    inputs=[feather_sl, vid_state],
                    outputs=[vid_review, vid_status],
                )
                btn_vid_3s.click(
                    vid_preview_3s,
                    inputs=[vid_state],
                    outputs=[vid_out, vid_review, vid_status],
                )
                btn_vid_run.click(
                    vid_run_full,
                    inputs=[vid_state],
                    outputs=[vid_out, vid_review, vid_status],
                )

            # ---------- BATCH TAB ----------
            with gr.Tab("Batch (ZIP output)"):
                batch_files = gr.File(file_count="multiple", label="Upload Images / Videos")
                btn_batch_run = gr.Button("Run full", variant="primary")
                batch_log = gr.Textbox(label="Batch Log", lines=8, interactive=False)
                batch_zip = gr.File(label="Download Batch ZIP")

                btn_batch_run.click(
                    run_batch_zip,
                    inputs=[batch_files, thresh_sl, dilation_sl, feather_sl],
                    outputs=[batch_zip, batch_log],
                )

    return demo


demo = build_ui()
app = gr.mount_gradio_app(FastAPI(), demo, path="/")


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", share=False)
