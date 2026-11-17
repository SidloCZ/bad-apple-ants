"""Independent CPU references for new CUDA contour distance and ranking."""

import json, struct, time
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_cdt
from scipy.signal import correlate2d
from badapple_coastline.cuda_backend import CudaAtlas, shape_metrics_from_counts
from badapple_coastline.precision_config import contour_features
from badapple_coastline.precision_search import FrameScoreCache, score_batches

rng = np.random.default_rng(7139)
mask = rng.integers(0, 2, (64, 64), dtype=np.uint8)
rows = [np.flatnonzero(np.diff(np.r_[0, row])).astype("<u4") for row in mask]
offsets = np.r_[0, np.cumsum([len(r) for r in rows])].astype("<u8")
path = Path(".tmp/precision-test-native.rle").resolve()
path.parent.mkdir(exist_ok=True)
path.write_bytes(b"LWMRLE01" + struct.pack("<II", 64, 64) + offsets.tobytes() + b"".join(r.tobytes() for r in rows))


def edges(a):
    out = np.zeros_like(a, bool)
    out[:, 1:] |= a[:, 1:] != a[:, :-1]
    out[:, :-1] |= a[:, 1:] != a[:, :-1]
    out[1:] |= a[1:] != a[:-1]
    out[:-1] |= a[1:] != a[:-1]
    return out


def distances(edge):
    return np.minimum(distance_transform_cdt(~edge, metric="taxicab"), 256) if edge.any() else np.full(edge.shape, 256)


with CudaAtlas(load_globe=False) as cuda:
    for row in range(6):
        for col in range(12):
            left = -180 + col * 30
            top = 90 - row * 30
            cuda.check(
                cuda.library.coast_cuda_add_tile(
                    cuda.handle, row * 12 + col, str(path).encode(), left, top - 30, left + 30, top
                )
            )
    cuda.check(cuda.library.coast_cuda_finish_tiles(cuda.handle))
    geos = np.array([[15, 15, 1200, angle] for angle in [0, 15, 67]], np.float64)
    for w, h in [(64, 48), (512, 384), (1444, 1080)]:
        scenes = cuda.sample(geos, (w, h), aspect=1080 / 1444)
        query = scenes[1].copy()
        query[h // 4 : h // 3, w // 4 : w // 3] ^= 1
        start = time.perf_counter()
        metrics = cuda.score_shape(geos, query)
        elapsed = time.perf_counter() - start
        base = cuda.score(geos, query)
        np.testing.assert_array_equal(metrics["counts"], base["counts"])
        qe = edges(query)
        qd = distances(qe)
        weights = np.stack(
            [np.rint(1e6 * np.exp(-np.arange(257) / (tau * w / 1444))).astype(np.uint64) for tau in [8, 32]]
        )
        weights[:, 256] = 0
        for i, scene in enumerate(scenes):
            se = edges(scene)
            sd = distances(se)
            expected = [
                weights[0, qd[se]].sum(),
                weights[0, sd[qe]].sum(),
                weights[1, qd[se]].sum(),
                weights[1, sd[qe]].sum(),
                qd[se].sum(),
                sd[qe].sum(),
            ]
            np.testing.assert_array_equal(metrics["shape_counts"][i, 7:], expected)
        print(
            f"Exact clipped L1 distances and integer soft-contour sums match SciPy: {w}x{h}, {elapsed:.4f}s", flush=True
        )
    for query in [np.zeros((48, 64), np.uint8), np.ones((48, 64), np.uint8)]:
        values = cuda.score_shape(geos, query)
        assert np.isfinite(values["precision_objective"]).all()
        assert (values["soft_contour_f1"] == 0).all()
    mirror_geos = np.column_stack([geos, np.ones(len(geos))])
    normal = cuda.sample(geos, (1444, 1080), aspect=1080 / 1444)
    reflected = cuda.sample(mirror_geos, (1444, 1080), aspect=1080 / 1444)
    np.testing.assert_array_equal(reflected, normal[:, :, ::-1])
    query = normal[1].copy()
    query[20:30, 30:50] ^= 1
    actual = cuda.score_shape(mirror_geos, query)
    expected = cuda.score_shape(geos, query[:, ::-1])
    np.testing.assert_array_equal(actual["shape_counts"], expected["shape_counts"])
    print("Original-size mirrored sampling and every area/contour count match exact horizontal reflection", flush=True)
    views = np.vstack([np.column_stack([geos, np.zeros(len(geos))]), mirror_geos])
    cache = FrameScoreCache(query)
    cached = score_batches(cuda, views, query, cache=cache)
    reference = score_batches(cuda, views, query)
    for name in reference:
        np.testing.assert_array_equal(cached[name], reference[name])
    repeated = np.r_[np.arange(len(views) - 1, -1, -1), 0, 0, 3]
    cached = score_batches(cuda, views[repeated], query, batch=2, cache=cache)
    for name in reference:
        np.testing.assert_array_equal(cached[name], reference[name][repeated])
    assert len(cache.rows) == len(views)
    print(
        "Full-resolution frame score reuse preserves all13 integer counts and every metric, including reordered/repeated normal/mirrored views",
        flush=True,
    )
    w, h, mx, my = 40, 32, 7, 5
    aspect = h / w
    query = cuda.sample(geos[[1]], (w, h), aspect=aspect)[0].astype(np.float32)
    top_geo, top_scores = cuda.refine_topk(mirror_geos, query, (mx, my), 3, 2, aspect)
    sw, sh = w + 2 * mx, h + 2 * my
    expanded = mirror_geos.copy()
    expanded[:, 2] *= sw / w
    scenes = cuda.sample(expanded, (sw, sh), aspect=aspect * sh / h / (sw / w))
    for i, scene in enumerate(scenes):
        intersection = correlate2d(scene.astype(np.float64), query, mode="valid")
        mass = correlate2d(scene.astype(np.float64), np.ones_like(query), mode="valid")
        qm = query.sum()
        lu = mass + qm - intersection
        wu = w * h - intersection
        scores = 0.5 * (
            np.divide(intersection, lu, out=np.ones_like(lu), where=lu > 0)
            + np.divide(w * h - mass - qm + intersection, wu, out=np.ones_like(wu), where=wu > 0)
        )
        for rank in range(3):
            candidate = top_geo[i * 3 + rank]
            a = np.deg2rad(geos[i, 3])
            c = np.cos(a)
            s = np.sin(a)
            dx = (candidate[0] - geos[i, 0]) * 111.32 * np.cos(np.deg2rad(geos[i, 1]))
            dy = (candidate[1] - geos[i, 1]) * 111.32
            x = int(round(-(c * dx - s * dy) * w / geos[i, 2] + mx))
            y = int(round((-s * dx - c * dy) * h / (geos[i, 2] * aspect) + my))
            assert abs(top_scores[i * 3 + rank] - scores.max()) < 2e-5
            assert abs(top_scores[i * 3 + rank] - scores[y, x]) < 2e-5
            scores[max(0, y - 2) : y + 3, max(0, x - 2) : x + 3] = -1
    print(
        "Mirrored multi-peak FFT translations and returned geographic offsets match exhaustive CPU reference",
        flush=True,
    )
    features, valid = cuda.features(geos, size=(96, 72))
    with CudaAtlas(load_globe=False) as contours:
        cuda.queries(features)
        contours.queries(contour_features(features))
        cuda.start_region(3)
        contours.start_region(3)
        cuda.rank(features, valid, offset=17)
        contours.rank(contour_features(features), valid, offset=17)
        expected_area = cuda.region_results()
        expected_contour = contours.region_results()
        cuda.start_region(3)
        contours.start_region(3)
        cuda.rank_native_profiles(contours, geos, offset=17)
        for expected, actual in [(expected_area, cuda.region_results()), (expected_contour, contours.region_results())]:
            np.testing.assert_allclose(actual[0], expected[0], rtol=0, atol=1e-7)
            np.testing.assert_array_equal(actual[1], expected[1])
        print(
            "GPU-resident area/contour descriptors and both regional rankings match CPU descriptor construction exactly",
            flush=True,
        )
    Path("output/coastline-precision-cuda-validation.json").write_text(
        json.dumps(
            dict(
                exact_reference="scipy.ndimage.distance_transform_cdt taxicab, clipped to256; integer exponent weights",
                sizes=[[64, 48], [512, 384], [1444, 1080]],
                integer_count_differences=0,
                mirrored_pixel_differences=0,
                mirrored_fft_reference="exhaustive CPU",
                resident_descriptor_rank_differences=0,
                frame_score_cache_count_and_metric_differences=0,
                full_resolution_seconds=elapsed,
                build=cuda.build,
            ),
            indent=2,
        )
    )
