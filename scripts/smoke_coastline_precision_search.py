"""Exercise production search stages with a tiny, explicitly synthetic test atlas.

This creates only validation files under .tmp; it never supplies movie frames.
The full original globe can keep running independently in another GPU context.
"""

import json, struct, time
from pathlib import Path
import numpy as np
from unittest.mock import patch
from badapple_coastline.cuda_backend import CudaAtlas
from badapple_coastline.precision_search import Search, vector, geo_key
from badapple_coastline.precision_config import KEYS
from badapple_coastline.data import digest
from PIL import Image

root = Path(".tmp/precision-global-path-smoke")
root.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(19476)
mask = rng.integers(0, 2, (64, 64), dtype=np.uint8)
rows = [np.flatnonzero(np.diff(np.r_[0, row])).astype("<u4") for row in mask]
offsets = np.r_[0, np.cumsum([len(r) for r in rows])].astype("<u8")
path = (root / "validation-only.rle").resolve()
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
    target_geo = np.array([[15.07, 15.04, 1200.0, 15.12, 1.0]])
    target = cuda.sample(target_geo, (1444, 1080), aspect=1080 / 1444)[0]
    target[120:128, 220:228] ^= 1
    base_geo = np.array([[15.0, 15.0, 1200.0, 15.0, 0.0], [15.1, 15.1, 1210.0, 15.0, 0.0]])
    search = Search.__new__(Search)
    search.output = root
    search.raw = np.stack([target * 255, target * 255])
    search.base = [dict(zip(KEYS, geo)) for geo in base_geo]
    search.prior = search.base.copy()
    search.features = np.ones((2, 4), np.float32) / 2
    search.preceding = {}
    search.constant_geos = {}
    search.photos = {geo_key(r): i for i, r in enumerate(search.base)}
    search.settings = {"match_run_id": "synthetic-validation-only"}
    seeds = np.tile(target_geo, (128, 1))
    seeds[:, :2] += rng.normal(0, 0.25, (len(seeds), 2))
    seeds[:, 2] *= rng.uniform(0.93, 1.07, len(seeds))
    seeds[:, 3] += rng.uniform(-5, 5, len(seeds))
    seeds[:, 4] = np.arange(len(seeds)) % 2
    search.global_seeds = lambda frame: seeds
    search.output = root / "cached"
    started = time.perf_counter()
    result = search.global_frame(cuda, 0, search.base.copy())
    global_seconds = time.perf_counter() - started
    started = time.perf_counter()
    refined = search.temporal_frame(cuda, 0, [result, search.base[1]], "temporal-smoke")
    temporal_seconds = time.perf_counter() - started
    baseline = result["baseline_fullres_metrics"]
    for record in [result, refined]:
        exact = cuda.score_structure(np.array([vector(record)]), target, border_y=2)
        np.testing.assert_array_equal(exact["structure_counts"][0], record["structure_counts"])
        assert record["native_miou"] >= baseline["native_miou"] - 1e-12
        assert record["native_objective"] >= baseline["native_objective"] - 1e-12
        assert record["precision_objective"] >= baseline["precision_objective"] - 1e-12
        assert record["structure_objective"] >= baseline["structure_objective"] - 1e-12
        pixels = np.asarray(Image.open(record["native_mask_path"]))
        np.testing.assert_array_equal(pixels, cuda.sample([vector(record)], (1444, 1080))[0] * 255)
        assert digest(Path(record["native_mask_path"])) == record["native_mask_sha256"]
    assert refined["structure_objective"] >= result["structure_objective"] - 1e-12
    search.output = root / "uncached"
    with patch("badapple_coastline.precision_search.FrameScoreCache", return_value=None):
        started = time.perf_counter()
        uncached = search.global_frame(cuda, 0, search.base.copy())
        uncached_global_seconds = time.perf_counter() - started
        started = time.perf_counter()
        uncached_temporal = search.temporal_frame(cuda, 0, [uncached, search.base[1]], "temporal-smoke")
        uncached_temporal_seconds = time.perf_counter() - started
    for cached, plain in [(result, uncached), (refined, uncached_temporal)]:
        left = [cached] + cached["alternatives"]
        right = [plain] + plain["alternatives"]
        assert len(left) == len(right)
        for a, b in zip(left, right):
            np.testing.assert_array_equal(vector(a), vector(b))
            np.testing.assert_array_equal(a["structure_counts"], b["structure_counts"])
            assert a["structure_objective"] == b["structure_objective"]
        assert cached["native_mask_sha256"] == plain["native_mask_sha256"]
    report = dict(
        validation_only=True,
        synthetic_atlas=True,
        source_size=[1444, 1080],
        seed_count=len(seeds),
        integer_count_differences=0,
        native_pixel_differences=0,
        acceptance_floor_regressions=0,
        baseline_miou=baseline["native_miou"],
        global_miou=result["native_miou"],
        temporal_miou=refined["native_miou"],
        global_seconds=global_seconds,
        temporal_seconds=temporal_seconds,
        concurrent_dictionary=True,
        cached_vs_uncached_selected_geometry_differences=0,
        cached_vs_uncached_ranked_alternative_differences=0,
        uncached_global_seconds=uncached_global_seconds,
        uncached_temporal_seconds=uncached_temporal_seconds,
        note="Small test atlas; timings and accuracy are not production estimates or movie results",
        cuda_build=cuda.build,
    )
    (root / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
