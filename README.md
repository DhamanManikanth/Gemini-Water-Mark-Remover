---
title: AutoMarkRemover
emoji: ✨
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.32.0
python_version: "3.11"
app_file: app.py
pinned: false
---

# AutoMarkRemover

Static visible AI-generator watermark detection and removal for images and videos, keeping every pixel outside the mask bit-identical to the source.

- **Detection:** Procedural 4-point sparkle template matching (grayscale + Sobel gradients), corner anomaly scan, temporal video static-overlay detection, and manual brush (Method D).
- **Removal:** LaMa (images) and temporal $\alpha/L$ recovery + ProPainter (video).
- **Verification:** Outside-mask zero-diff check, re-detection check, edge energy & flicker comparison.
