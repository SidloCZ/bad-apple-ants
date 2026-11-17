"""Main land/water panel sampled from the native 10 m mask at display size.

The saved native masks are 1444x1080 nearest-neighbour samples. Enlarging them to
the 1920x1440 main panel would add a second resampling and visible stair-steps.
Here every display pixel is sampled from the original 0.3 arc-second grid through
the same view transform that produced the match (rotation and mirroring included):

* zoomed out (several grid cells per display pixel): s x s supersampling of the
  original cells, s <= 4, giving the true covered fraction of each pixel;
* zoomed in (one grid cell spans several display pixels): a Gaussian-weighted
  field (sigma 0.55 cell) of the 4x4 nearest cell centres, contoured at 0.5 with a
  one-pixel anti-aliasing ramp. This draws the 10 m cell boundary as a smooth
  contour instead of enlarged square cells, keeps one-cell-wide channels and
  isolated cells, and adds no detail finer than the source grid.

``nearest`` reproduces the saved native masks pixel for pixel; it is used to
check that the transform is identical to the one used for matching.
"""

from __future__ import annotations
import math
import numpy as np
from .atlas import view_transform

CELL_DEGREES = 0.3 / 3600  # Kindermann global land/water mask grid spacing
SIGMA = 0.55  # Gaussian contour width in grid cells (keeps single-cell features)


def transform_for(record, size):
    return view_transform(
        record["longitude"],
        record["latitude"],
        record["width_km"],
        record["rotation_clockwise"],
        size,
        mirror_horizontal=bool(record.get("mirror_horizontal")),
    )


def pixels_per_cell(transform):
    step_x = math.hypot(transform.a, transform.d) / CELL_DEGREES
    step_y = math.hypot(transform.b, transform.e) / CELL_DEGREES
    return 1.0 / max(step_x, step_y)


class CoverageRenderer:
    """Rust/Rayon renderer over the lossless 10 m RLE globe."""

    def __init__(self, threads=None):
        import ctypes, os
        from .local_map import ROOT

        if threads:
            os.environ["RAYON_NUM_THREADS"] = str(threads)
        self.ctypes = ctypes
        self.library_path = ROOT / "rust/coastline-native/target/release/coastline_native.dll"
        self.dll = ctypes.CDLL(str(self.library_path))
        self.dll.coastline_native_open.argtypes = [ctypes.c_char_p]
        self.dll.coastline_native_open.restype = ctypes.c_void_p
        self.dll.coastline_native_close.argtypes = [ctypes.c_void_p]
        self.dll.coastline_native_render_coverage.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_float),
        ]
        self.dll.coastline_native_render_coverage.restype = ctypes.c_int
        self.manifest = ROOT / "cache/coastline-native-rle/manifest.json"
        self.handle = self.dll.coastline_native_open(str(self.manifest).encode())
        if not self.handle:
            raise ValueError("Cannot open verified native 10 m globe")

    def render(self, transform, size, supersample, ppc=1.0, sigma=SIGMA):
        w, h = size
        ct = self.ctypes
        affine = np.array([transform.a, transform.b, transform.c, transform.d, transform.e, transform.f], np.float64)
        out = np.empty((h, w), np.float32)
        code = self.dll.coastline_native_render_coverage(
            self.handle,
            affine.ctypes.data_as(ct.POINTER(ct.c_double)),
            w,
            h,
            supersample,
            float(ppc),
            float(sigma),
            out.ctypes.data_as(ct.POINTER(ct.c_float)),
        )
        if code:
            raise ValueError(
                f"Native coverage render failed ({code}); missing source coverage"
                if code == 1
                else f"Native coverage render failed ({code})"
            )
        return out

    def close(self):
        if self.handle:
            self.dll.coastline_native_close(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def nearest(renderer, record, size):
    """Pixel-centre nearest neighbour: identical to the saved native masks."""
    return (renderer.render(transform_for(record, size), size, 1) > 0.5).astype(np.uint8)


def coverage(renderer, record, size, max_supersample=4, sigma=None):
    """Fraction of land per display pixel, float32 in [0, 1]."""
    transform = transform_for(record, size)
    ppc = pixels_per_cell(transform)
    if ppc >= 1.0:
        sigma = SIGMA if sigma is None else sigma
        return renderer.render(transform, size, 0, ppc, sigma), dict(
            mode="gaussian-contour", pixels_per_cell=ppc, samples_per_pixel=16, sigma_cells=sigma
        )
    s = int(min(max_supersample, math.ceil(1.0 / ppc)))
    return renderer.render(transform, size, s), dict(
        mode="supersampled-area", pixels_per_cell=ppc, samples_per_pixel=s * s
    )


def srgb_to_linear(c):
    c = np.asarray(c, np.float32) / 255
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.round(255 * np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)).astype(np.uint8)


def colorize(cover, land, water):
    """Blend in linear light so anti-aliased edges keep their apparent weight
    (1024-step lookup table; coverage is quantised to 1/1023)."""
    lut = _palette(tuple(land), tuple(water))
    return lut[np.round(cover * 1023).astype(np.int16)]


from functools import lru_cache


@lru_cache(maxsize=8)
def _palette(land, water):
    lw, ll = srgb_to_linear(water), srgb_to_linear(land)
    return linear_to_srgb(lw + np.linspace(0, 1, 1024, dtype=np.float32)[:, None] * (ll - lw))


def colorize8(cover8, land, water):
    """Colorize stored 8-bit land fraction (linear-light blend, 256-entry table)."""
    lw, ll = srgb_to_linear(water), srgb_to_linear(land)
    lut = linear_to_srgb(lw + np.linspace(0, 1, 256, dtype=np.float32)[:, None] * (ll - lw))
    return lut[cover8]
