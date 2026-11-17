"""Independent GPU video treatment and conservative full-pixel WCAG screening."""

import ctypes, json, os
from pathlib import Path
import numpy as np


def exposure_from_histogram(histogram):
    """Set mean tone-curve output to 0.5 using this frame's luminance only."""
    weights = np.asarray(histogram, np.float64)
    if weights.shape != (512,) or weights.sum() == 0:
        raise ValueError("Empty photographic luminance histogram")
    weights /= weights.sum()
    stops = (np.arange(512) + 0.5) / 32 - 16
    low, high = -21.0, 5.0
    for _ in range(24):
        middle = (low + high) / 2
        mean = weights @ np.clip(0.5 + (stops - middle) * 0.22, 0, 1)
        if mean > 0.5:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def levels_from_histogram(histogram, black_cap=0.01, white_floor=0.06, low=0.005, high=0.995):
    """Black/white relative luminance from this frame's photo pixels only."""
    weights = np.asarray(histogram, np.float64)
    if weights.shape != (512,) or weights.sum() == 0:
        raise ValueError("Empty photographic luminance histogram")
    cumulative = np.cumsum(weights) / weights.sum()
    values = 2.0 ** ((np.arange(512) + 0.5) / 32 - 16)
    black = min(float(values[np.searchsorted(cumulative, low)]), black_cap)
    white = max(float(values[min(511, np.searchsorted(cumulative, high))]), white_floor)
    return black, white


class FlashCuda:
    def __init__(self, width=1280, height=960, window=30, span=0.07):
        root = Path(__file__).resolve().parents[1]
        self.build = json.loads((root / "cuda/video_flash/build/current.json").read_text(encoding="utf-8-sig"))
        cuda = Path(self.build["cuda_root"])
        self.directories = [os.add_dll_directory(str(p)) for p in [cuda / "bin", cuda / "bin/x64"] if p.is_dir()]
        self.dll = ctypes.CDLL(self.build["library"])
        self.dll.video_flash_error.restype = ctypes.c_char_p
        byte = ctypes.POINTER(ctypes.c_uint8)
        self.dll.video_flash_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float]
        self.dll.video_flash_create.restype = ctypes.c_void_p
        self.dll.video_flash_destroy.argtypes = [ctypes.c_void_p]
        self.dll.video_flash_process.argtypes = [ctypes.c_void_p, byte, byte, ctypes.c_float, ctypes.c_float]
        self.dll.video_flash_process_levels.argtypes = [ctypes.c_void_p, byte, byte] + [ctypes.c_float] * 6
        self.dll.video_flash_histogram.argtypes = [ctypes.c_void_p, byte, ctypes.POINTER(ctypes.c_uint32)]
        self.dll.video_flash_shadow_check.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        self.dll.video_flash_audit.argtypes = [ctypes.c_void_p, byte, ctypes.POINTER(ctypes.c_float)]
        self.dll.video_flash_audit_yuv420.argtypes = self.dll.video_flash_audit.argtypes
        self.handle = self.dll.video_flash_create(width, height, window, span)
        if not self.handle:
            raise RuntimeError(self.dll.video_flash_error().decode())
        self.shape = (height, width, 3)
        self.output = np.empty(self.shape, np.uint8)
        self.metrics = np.empty(8, np.float32)
        self.histogram = np.empty(512, np.uint32)
        self.span = span
        self.exposure = None
        self.shadow_stats = np.empty(4, np.uint32)

    def check(self, code):
        if code:
            raise RuntimeError(self.dll.video_flash_error().decode())

    def process(self, frame):
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        if frame.shape != self.shape:
            raise ValueError("Unexpected video dimensions")
        self.check(
            self.dll.video_flash_histogram(
                self.handle,
                frame.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                self.histogram.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            )
        )
        self.exposure = exposure_from_histogram(self.histogram)
        self.check(
            self.dll.video_flash_process(
                self.handle,
                frame.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                self.output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                self.exposure,
                self.span,
            )
        )
        return self.output

    def process_levels(self, frame, floor=0.002, gamma=0.75, saturation=1.15):
        """Black/white-level tone curve; output luminance stays in [floor,floor+span]."""
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        if frame.shape != self.shape:
            raise ValueError("Unexpected video dimensions")
        self.check(
            self.dll.video_flash_histogram(
                self.handle,
                frame.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                self.histogram.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            )
        )
        self.levels = levels_from_histogram(self.histogram)
        self.check(
            self.dll.video_flash_process_levels(
                self.handle,
                frame.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                self.output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                *self.levels,
                gamma,
                floor,
                self.span,
                saturation,
            )
        )
        return self.output

    def shadow_metrics(self):
        """Check the last process() result against its input, entirely on the GPU."""
        self.check(
            self.dll.video_flash_shadow_check(
                self.handle, self.shadow_stats.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
            )
        )
        return self.shadow_stats.copy()

    def audit(self, frame):
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        if frame.shape != self.shape:
            raise ValueError("Unexpected video dimensions")
        self.check(
            self.dll.video_flash_audit(
                self.handle,
                frame.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                self.metrics.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
        )
        return self.metrics.copy()

    def close(self):
        if self.handle:
            self.dll.video_flash_destroy(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def audit_yuv420(self, frame):
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        if frame.size != self.shape[0] * self.shape[1] * 3 // 2:
            raise ValueError("Unexpected YUV420 frame size")
        self.check(
            self.dll.video_flash_audit_yuv420(
                self.handle,
                frame.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                self.metrics.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
        )
        return self.metrics.copy()
