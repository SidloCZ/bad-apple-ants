"""Independent CPU references for GPU treatment and conservative flash bounds."""

import json
from pathlib import Path
import numpy as np
from badapple_coastline.flash_cuda import FlashCuda


def linear(rgb):
    x = np.asarray(rgb, np.float32) / 255
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def cpu_audits(frames, window=30):
    previous = None
    deltas = []
    reds = []
    rows = []
    low = None
    high = None
    for rgb in frames:
        values = linear(rgb)
        y = values @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        low = y.copy() if low is None else np.minimum(low, y)
        high = y.copy() if high is None else np.maximum(high, y)
        deltas.append(np.zeros_like(y) if previous is None else np.abs(y - previous))
        previous = y
        red = values[:, :, 0] >= 0.8 * np.maximum(values.sum(axis=2), 1e-30)
        reds.append(red)
        tv = np.sum(deltas[-window:], axis=0)
        union = np.any(reds[-window - 1 :], axis=0)
        rows.append(
            [
                tv.max(),
                np.count_nonzero((tv >= 0.55) & (high - low >= 0.1)),
                np.count_nonzero(union),
                np.count_nonzero(red),
                (high - low).max(),
                y.min(),
                y.max(),
                y.mean(),
            ]
        )
    return np.asarray(rows)


def run():
    rng = np.random.default_rng(5292)
    frames = rng.integers(0, 256, (70, 12, 16, 3), np.uint8)
    expected = cpu_audits(frames)
    with FlashCuda(16, 12) as gpu:
        actual = np.array([gpu.audit(f) for f in frames])
    np.testing.assert_allclose(actual[:, 0], expected[:, 0], rtol=0, atol=2e-5)
    np.testing.assert_array_equal(actual[:, 1:4], expected[:, 1:4])
    np.testing.assert_allclose(actual[:, 4:], expected[:, 4:], rtol=0, atol=2e-6)
    with FlashCuda(16, 12) as gpu:
        treated = np.array([gpu.process(f).copy() for f in frames])
    yy, xx = np.indices((12, 16))
    photo = (xx * 1280 // 16 >= 20) & (xx * 1280 // 16 < 980) & (yy * 960 // 12 >= 70) & (yy * 960 // 12 < 790)
    np.testing.assert_array_equal(treated[:, ~photo], frames[:, ~photo])
    # Pixels depend on the current image only, independently of ALL earlier images.
    with FlashCuda(16, 12) as gpu:
        reverse = np.array([gpu.process(f).copy() for f in frames[::-1]])[::-1]
    np.testing.assert_array_equal(treated, reverse)
    with FlashCuda(16, 12) as gpu:
        a = gpu.process(frames[0]).copy()
        gpu.process(frames[1])
        b = gpu.process(frames[0]).copy()
    np.testing.assert_array_equal(a, b)
    # Lifting almost-black source/codec chroma must not create blue/magenta specks.
    shadows = np.asarray(
        [[0, 0, 1], [1, 0, 1], [0, 1, 6], [0, 2, 9], [2, 0, 2], [0, 0, 15], [15, 0, 15], [9, 0, 9]], np.uint8
    )
    shadow_frame = np.resize(shadows, (12, 16, 3))
    with FlashCuda(16, 12) as gpu:
        shadow_output = gpu.process(shadow_frame).copy()
        shadow_stats = gpu.shadow_metrics()
    shadow_spread = np.ptp(shadow_output[photo].astype(int), axis=1)
    assert shadow_spread.max() <= 10
    np.testing.assert_array_equal(shadow_stats, [0, 0, int(photo.sum()), int(shadow_spread.max())])
    old_linear = linear(shadow_frame[photo])
    new_linear = linear(shadow_output[photo])
    assert np.max(np.ptp(new_linear, axis=1) - np.ptp(old_linear, axis=1)) < 0.001
    # A moving hard edge is never mixed with its previous position.
    edge = np.full((12, 16, 3), 30, np.uint8)
    edge[:, 8:] = 220
    flat = np.full_like(edge, 80)
    with FlashCuda(16, 12) as gpu:
        gpu.process(edge)
        flat_output = gpu.process(flat).copy()
    assert np.unique(flat_output[photo], axis=0).shape[0] == 1
    # This central sample is in the photographic panel at any resolution.
    luma = linear(treated) @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    assert luma[:, photo].min() > 0.007 and luma[:, photo].max() < 0.080
    assert np.max(np.abs(np.diff(luma[:, 4:8, 4:10], axis=0))) < 0.075
    # Mean luminance is normalized on each current frame, not through averaging frames.
    assert np.max(np.abs(luma[:, photo].mean(axis=1) - 0.043)) < 0.001
    x = linear(treated)[:, 4:8, 4:10]
    ratio = x[:, :, :, 0] / np.maximum(x.sum(axis=3), 1e-30)
    assert ratio.max() < 0.8
    # Black has no saturated-red area; old color is retained for31frames only.
    red = np.zeros((12, 16, 3), np.uint8)
    red[:, :, 0] = 255
    black = np.zeros_like(red)
    with FlashCuda(16, 12) as gpu:
        rows = np.array([gpu.audit(f) for f in [red] + [black] * 32])
    assert rows[0, 2] == 192 and rows[30, 2] == 192 and rows[31, 2] == 0
    # Many small changes can exceed the variation bound without a 0.10 flash.
    narrow = [np.full((12, 16, 3), v, np.uint8) for v in [25, 80] * 20]
    with FlashCuda(16, 12) as gpu:
        small = np.array([gpu.audit(f) for f in narrow])
    assert small[:, 0].max() > 0.55 and small[:, 1].max() == 0 and small[:, 4].max() < 0.1
    # Large opposing changes still trip the conservative potential-flash screen.
    with FlashCuda(16, 12) as gpu:
        large = np.array([gpu.audit(f) for f in [black, red] * 20])
    assert large[:, 1].max() == 192
    # YUV420 path, independently converted using the specified BT.709 limited-range matrix.
    yuv = rng.integers(16, 236, 16 * 12 * 3 // 2, np.uint8)
    y = (yuv[:192].reshape(12, 16).astype(float) - 16) / 219
    cb = (yuv[192:240].reshape(6, 8).repeat(2, 0).repeat(2, 1).astype(float) - 128) / 224
    cr = (yuv[240:].reshape(6, 8).repeat(2, 0).repeat(2, 1).astype(float) - 128) / 224
    rgb = np.clip(np.stack([y + 1.5748 * cr, y - 0.187324 * cb - 0.468124 * cr, y + 1.8556 * cb], axis=2), 0, 1)
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    reds = (lin[:, :, 0] / np.maximum(lin.sum(axis=2), 1e-30) >= 0.8).sum()
    with FlashCuda(16, 12) as gpu:
        values = gpu.audit_yuv420(yuv)
    assert values[1] == 0 and values[2] == values[3] == reds
    result = dict(
        rgb_rolling_variation_reference=True,
        exact_red_and_general_area_counts=True,
        current_frame_order_independence=True,
        no_previous_image_or_edge_residue=True,
        normalized_frame_luminance=True,
        low_contrast_flash_exclusion_and_large_flash_detection=True,
        decoded_luminance_range_and_mean_reference=True,
        saturated_red_quantization_bound=True,
        red_window_expiry=True,
        yuv420_reference=True,
        original_comparison_and_all_nonphoto_pixels_unchanged=True,
        validated_luminance_span=0.07,
    )
    result.update(
        near_black_blue_magenta_regression=True,
        shadow_chroma_not_amplified=True,
        maximum_near_black_color_channel_spread=int(shadow_spread.max()),
    )
    Path("output/flash-cuda-chroma-validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    run()
