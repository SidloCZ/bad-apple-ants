import json, struct, time
from pathlib import Path
import numpy as np
from scipy.signal import correlate2d
from scipy.ndimage import maximum_filter
from badapple_coastline.cuda_backend import CudaAtlas

rng = np.random.default_rng(953)
mask = rng.integers(0, 2, (64, 64), dtype=np.uint8)
rows = []
for row in mask:
    transitions = np.flatnonzero(np.diff(np.r_[0, row])).astype("<u4")
    rows.append(transitions)
offsets = np.r_[0, np.cumsum([len(r) for r in rows])].astype("<u8")
path = Path(".tmp/cuda-test-native.rle").resolve()
path.parent.mkdir(exist_ok=True)
path.write_bytes(b"LWMRLE01" + struct.pack("<II", 64, 64) + offsets.tobytes() + b"".join(r.tobytes() for r in rows))

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
    w, h, mx, my = 40, 32, 7, 5
    aspect = h / w
    query = cuda.sample(geos[[1]], (w, h))[0].astype(np.float32)
    shifted, scores = cuda.refine(geos, query, (mx, my), aspect)
    top_geo, top_scores = cuda.refine_topk(geos, query, (mx, my), 3, 2, aspect)
    # Independently reconstruct the same scene and brute-force every translation.
    sw, sh = w + mx * 2, h + my * 2
    scene_geos = geos.copy()
    scene_geos[:, 2] *= sw / w
    scenes = cuda.sample(scene_geos, (sw, sh), aspect=aspect * sh / h / (sw / w))
    for i, scene in enumerate(scenes):
        intersections = correlate2d(scene.astype(np.float64), query, mode="valid")
        mass = correlate2d(scene.astype(np.float64), np.ones_like(query), mode="valid")
        qsum = query.sum()
        lu = mass + qsum - intersections
        wu = w * h - intersections
        land = np.divide(intersections, lu, out=np.ones_like(lu), where=lu > 0)
        water = np.divide(w * h - mass - qsum + intersections, wu, out=np.ones_like(wu), where=wu > 0)
        expected = (land + water) / 2
        assert abs(scores[i] - expected.max()) < 2e-5, (i, scores[i], expected.max())
        remaining = expected.copy()
        for rank in range(3):
            candidate = top_geo[i * 3 + rank]
            a = np.deg2rad(geos[i, 3])
            c = np.cos(a)
            s = np.sin(a)
            dx = (candidate[0] - geos[i, 0]) * 111.32 * np.cos(np.deg2rad(geos[i, 1]))
            dy = (candidate[1] - geos[i, 1]) * 111.32
            x = int(round((c * dx - s * dy) * w / geos[i, 2] + mx))
            y = int(round((-s * dx - c * dy) * h / (geos[i, 2] * aspect) + my))
            assert abs(top_scores[i * 3 + rank] - remaining.max()) < 2e-5
            assert abs(top_scores[i * 3 + rank] - remaining[y, x]) < 2e-5
            remaining[max(0, y - 2) : y + 3, max(0, x - 2) : x + 3] = -1
    print(
        "cuFFT normalized correlation and integral-image translation scores match exhaustive CPU reference", flush=True
    )

    w, h = 1444, 1080
    sampled = cuda.sample(geos, (w, h))
    query = sampled[1].copy()
    query[20:24, 20:26] ^= 1
    start = time.perf_counter()
    metrics = cuda.score(geos, query, tolerance=3)
    elapsed = time.perf_counter() - start

    def edge(a):
        out = np.zeros_like(a, bool)
        out[:, 1:] |= a[:, 1:] != a[:, :-1]
        out[:, :-1] |= a[:, 1:] != a[:, :-1]
        out[1:] |= a[1:] != a[:-1]
        out[:-1] |= a[1:] != a[:-1]
        return out

    qe = edge(query)
    qd = maximum_filter(qe, size=7)
    for i, scene in enumerate(sampled):
        se = edge(scene)
        sd = maximum_filter(se, size=7)
        expected = [scene.sum(), np.sum(scene & query), 0, se.sum(), qe.sum(), np.sum(se & qd), np.sum(qe & sd)]
        assert np.array_equal(metrics["counts"][i], expected), (metrics["counts"][i], expected)
    print("Full 1444x1080 exact land/water/boundary integer counts match CPU reference", elapsed, flush=True)
    Path("output").mkdir(exist_ok=True)
    Path("output/coastline-cuda-refinement-validation.json").write_text(
        json.dumps(
            dict(
                fft_translation_reference="exhaustive scipy.signal.correlate2d",
                full_resolution=[w, h],
                scored_pixels=len(geos) * w * h,
                integer_count_differences=0,
                score_seconds=elapsed,
                build=cuda.build,
            ),
            indent=2,
        )
    )
