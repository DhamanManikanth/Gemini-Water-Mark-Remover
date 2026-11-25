"""Video metadata probing (ffprobe), lossless frame extraction, and final CRF 14 H.264 encoding (ffmpeg)."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

from .config import find_ffmpeg_binaries


@dataclass
class VideoMetadata:
    width: int
    height: int
    fps_str: str
    fps: float
    pix_fmt: str
    color_space: str | None
    color_transfer: str | None
    color_primaries: str | None
    color_range: str | None
    has_audio: bool
    nb_frames: int


def require_ffmpeg() -> tuple[str, str]:
    """Return (ffmpeg_path, ffprobe_path) or raise a clear RuntimeError."""
    ffmpeg, ffprobe = find_ffmpeg_binaries()
    if not ffmpeg or not ffprobe:
        raise RuntimeError(
            "ffmpeg and/or ffprobe not found. Please install FFmpeg and ensure it is on PATH "
            "(e.g., `winget install Gyan.FFmpeg`) before processing video."
        )
    return ffmpeg, ffprobe


def probe_video(video_path: str | Path) -> VideoMetadata:
    """Inspect video stream and color metadata using ffprobe."""
    _, ffprobe = require_ffmpeg()
    cmd = [
        ffprobe,
        "-v", "error",
        "-show_streams",
        "-show_format",
        "-of", "json",
        str(video_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(res.stdout)
    streams = info.get("streams", [])
    v_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    if v_stream is None:
        raise RuntimeError(f"No video stream found in {video_path}")
    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    fps_str = v_stream.get("r_frame_rate") or v_stream.get("avg_frame_rate") or "30/1"
    num, _, den = fps_str.partition("/")
    fps = float(num) / float(den) if den and float(den) != 0 else float(num)

    nb_frames = int(v_stream.get("nb_frames") or 0)
    return VideoMetadata(
        width=int(v_stream["width"]),
        height=int(v_stream["height"]),
        fps_str=fps_str,
        fps=fps,
        pix_fmt=v_stream.get("pix_fmt") or "yuv420p",
        color_space=v_stream.get("color_space"),
        color_transfer=v_stream.get("color_transfer"),
        color_primaries=v_stream.get("color_primaries"),
        color_range=v_stream.get("color_range"),
        has_audio=has_audio,
        nb_frames=nb_frames,
    )


def sample_video_frames(video_path: str | Path, num_samples: int = 80) -> list[np.ndarray]:
    """Evenly sample 60-100 BGR frames from `video_path`."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: list[np.ndarray] = []
    if total > 0:
        count = min(total, max(1, num_samples))
        indices = np.linspace(0, total - 1, count, dtype=int)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(frame)
    else:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        if len(frames) > num_samples:
            idxs = np.linspace(0, len(frames) - 1, num_samples, dtype=int)
            frames = [frames[i] for i in idxs]
    cap.release()
    return frames


def read_all_frames(video_path: str | Path, max_frames: int | None = None) -> list[np.ndarray]:
    """Decode video frames in BGR uint8."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()
    return frames


def encode_video_ffmpeg(
    frames_bgr: list[np.ndarray],
    source_video_path: str | Path,
    output_path: str | Path,
    meta: VideoMetadata,
) -> Path:
    """Encode cleaned frames in one final ffmpeg pass: H.264, CRF 14, preserving fps/pix_fmt/color/audio."""
    ffmpeg, _ = require_ffmpeg()
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg, "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{meta.width}x{meta.height}",
        "-r", meta.fps_str,
        "-i", "-",
    ]
    if meta.has_audio:
        cmd.extend(["-i", str(source_video_path), "-map", "0:v:0", "-map", "1:a?", "-c:a", "copy"])
    else:
        cmd.extend(["-map", "0:v:0", "-an"])

    cmd.extend([
        "-c:v", "libx264",
        "-crf", "14",
        "-preset", "slow",
        "-pix_fmt", meta.pix_fmt,
    ])

    if meta.color_space and meta.color_space != "unknown":
        cmd.extend(["-colorspace", meta.color_space])
    if meta.color_transfer and meta.color_transfer != "unknown":
        cmd.extend(["-color_trc", meta.color_transfer])
    if meta.color_primaries and meta.color_primaries != "unknown":
        cmd.extend(["-color_primaries", meta.color_primaries])
    if meta.color_range and meta.color_range != "unknown":
        cmd.extend(["-color_range", meta.color_range])

    cmd.extend(["-movflags", "+faststart", str(out_p)])

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for fr in frames_bgr:
            proc.stdin.write(fr.tobytes())
        proc.stdin.close()
    except BrokenPipeError:
        pass
    stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"ffmpeg encoding failed (code {ret}):\n{stderr}")
    return out_p
