"""Keep source resolution while excluding its verified fixed letterbox rows."""

from __future__ import annotations
import numpy as np

SOURCE_SIZE = (1444, 1080)
SOURCE_BORDER_Y = 2


def border_rows(height):
    return round(SOURCE_BORDER_Y * height / SOURCE_SIZE[1])


def content(array, border=None):
    border = border_rows(array.shape[-2]) if border is None else border
    return array[..., border:-border, :] if border else array


def extended(array):
    """Descriptor/FFT padding only; the original frames are never modified."""
    result = np.array(array, copy=True)
    border = border_rows(result.shape[-2])
    if border:
        result[..., :border, :] = result[..., border : border + 1, :]
        result[..., -border:, :] = result[..., -border - 1 : -border, :]
    return result
