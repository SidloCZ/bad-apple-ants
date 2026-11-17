"""Visible-image checks; valid geographic coverage alone does not prevent flashes."""

from __future__ import annotations

import numpy as np
from PIL import Image
from pathlib import Path
from .data import digest

POLICY = "no-whiteout-v1"


def image_quality(image):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    pixels = np.asarray(image.convert("RGB").resize((96, 72), Image.Resampling.BOX), dtype=np.float32)
    light = pixels.mean(axis=2)
    white = float((pixels.min(axis=2) >= 225).mean())
    bright = float((light >= 210).mean())
    mean = float(light.mean())
    deviation = float(light.std())
    rejected = white >= 0.35 or bright >= 0.60 or (mean >= 170 and deviation < 4)
    return dict(
        policy=POLICY,
        acceptable=not rejected,
        white_fraction=white,
        bright_fraction=bright,
        mean_brightness=mean,
        brightness_std=deviation,
    )


def file_quality(path):
    with Image.open(path) as image:
        return image_quality(image)


def black_placeholder(directory, size, interval, reason):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"black-{size[0]}x{size[1]}.png"
    if not path.is_file():
        Image.new("RGB", size, "black").save(path)
    return dict(
        status="unavailable",
        placeholder=True,
        path=str(path.resolve()),
        image_sha256=digest(path),
        size=list(size),
        interval=interval,
        coverage=0.0,
        clear_coverage=0.0,
        cloud_fraction=0.0,
        scenes=[],
        credit="No usable satellite image; black placeholder",
        quality_policy=POLICY,
        reason=reason,
    )
