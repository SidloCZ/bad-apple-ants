"""Settings of the precision (v4) search: paths, view grid, descriptor size and objective weights."""

import json
from pathlib import Path
import numpy as np

OUTPUT = Path("output/coastline-precision-v4")
BASELINE = Path("output/coastline-fullres-cuda-v2")
DICTIONARY = Path("cache/coastline-precision-dictionary-content")
LEGACY_DICTIONARY = Path("cache/coastline-precision-dictionary")
LEGACY_OUTPUT = Path("output/coastline-precision-v3")


def full_frames(root=Path("cache/coastline-frames-full")):
    """Full-size frame cache written by scripts/prepare_fullres_coastline_frames.py."""
    pointer = root / "current.json"
    if pointer.is_file():
        return root / json.loads(pointer.read_text(encoding="utf-8"))["directory"]
    return root / "aa9e9308939fbb2afe24"  # the published run's source (SHA-256 d8339642...)


FRAMES = full_frames()
WIDTHS = np.geomspace(2.0, 1024.0, 32).tolist()
ANGLES = np.arange(0.0, 360.0, 5.0)
FEATURE_SIZE = (96, 72)
CENTERS = 512000
REGIONAL_TOP = 8
PER_SCALE = 64
LEGACY_PER_SCALE = 32
CONTENT_BORDER_Y = 2
FRAME_EDGE_WEIGHT = 0.10
ASPECT = 1080 / 1444
KEYS = ["longitude", "latitude", "width_km", "rotation_clockwise", "mirror_horizontal"]


def geometry(indices, centers):
    indices = np.asarray(indices, dtype=np.int64)
    return np.column_stack(
        [
            centers[(indices // len(ANGLES)) % len(centers)],
            np.asarray(WIDTHS)[indices // (len(ANGLES) * len(centers))],
            ANGLES[indices % len(ANGLES)],
        ]
    )


def contour_features(features):
    """Area plus signed directional coastline differences, with zero DC."""
    features = np.asarray(features, dtype=np.int16).reshape(-1, FEATURE_SIZE[1], FEATURE_SIZE[0])
    centered = features - np.rint(features.mean(axis=(1, 2), keepdims=True)).astype(np.int16)
    dx = np.zeros_like(features)
    dy = np.zeros_like(features)
    dx[:, :, 1:] = 2 * (features[:, :, 1:] - features[:, :, :-1])
    dy[:, 1:, :] = 2 * (features[:, 1:, :] - features[:, :-1, :])
    return np.concatenate(
        [centered.reshape(len(features), -1), dx.reshape(len(features), -1), dy.reshape(len(features), -1)], axis=1
    ).astype(np.int8)
