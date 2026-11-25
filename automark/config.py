"""Configuration, device detection, and external tool discovery."""

from __future__ import annotations

import glob
import os
import shutil
from pathlib import Path
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
PROFILES_DIR = ROOT_DIR / "profiles"
OUTPUT_DIR = ROOT_DIR / "output"
REVIEW_DIR = OUTPUT_DIR / "review"
MODELS_DIR = ROOT_DIR / "models"
PROPAINTER_DIR = ROOT_DIR / "third_party" / "ProPainter"

# Detection & verification thresholds (do not lower just to pass tests)
REJECTION_THRESHOLD = 0.42          # Minimum combined score to consider a candidate valid
OUTPUT_MAX_SCORE_THRESHOLD = 0.35   # Max allowed detection score on cleaned output
AGREEMENT_RADIUS_PX = 3.0           # Max distance (px) for two methods to agree
CORNER_FRACTION = 0.30              # 30% x 30% corner search region
MIN_TEMPLATE_PX = 16                # Minimum sparkle template size (px)
MAX_TEMPLATE_FRACTION = 0.15        # Maximum sparkle template size (15% of shorter side)
NUM_TEMPLATE_SCALES = 44            # 40+ sizes

# Mask & removal defaults
DEFAULT_DILATION_PX = 2
DEFAULT_FEATHER_PX = 1
ALPHA_INPAINT_THRESHOLD = 0.85      # Where estimated alpha > 0.85, use LaMa / ProPainter
PROPAINTER_CHUNK_FRAMES = 80
PROPAINTER_OVERLAP_FRAMES = 10
PROPAINTER_LOW_VRAM_CHUNK = 40
PROPAINTER_LOW_VRAM_THRESHOLD_GB = 6.5


def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_vram_gb() -> float:
    """Return total GPU VRAM in GB, or 0.0 on CPU."""
    if not torch.cuda.is_available():
        return 0.0
    props = torch.cuda.get_device_properties(0)
    return float(props.total_memory) / (1024.0 ** 3)


def find_ffmpeg_binaries() -> tuple[str | None, str | None]:
    """Locate ffmpeg and ffprobe executables on PATH or standard WinGet locations."""
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")

    if ffmpeg and ffprobe:
        return ffmpeg, ffprobe

    # Fallback: check common WinGet installation directories on Windows
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    if local_app_data:
        pattern = os.path.join(local_app_data, "Microsoft", "WinGet", "Packages", "*FFmpeg*", "**", "bin")
        for bin_dir in glob.glob(pattern, recursive=True):
            cand_ffmpeg = os.path.join(bin_dir, "ffmpeg.exe")
            cand_ffprobe = os.path.join(bin_dir, "ffprobe.exe")
            if not ffmpeg and os.path.isfile(cand_ffmpeg):
                ffmpeg = cand_ffmpeg
            if not ffprobe and os.path.isfile(cand_ffprobe):
                ffprobe = cand_ffprobe

    return ffmpeg, ffprobe


def ensure_dirs() -> None:
    """Ensure workspace output/profile directories exist."""
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
