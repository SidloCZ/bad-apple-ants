import numpy as np
from badapple_coastline.target_content import content, extended, border_rows
from badapple_coastline.cuda_backend import structure_metrics_from_counts
from badapple_coastline.precision_search import FrameScoreCache, score_batches, admissible


def test_source_letterbox_extension_preserves_original_resolution_and_content():
    source = np.zeros((1080, 1444), np.uint8)
    source[2:-2] = np.arange(1076, dtype=np.uint16)[:, None] % 2
    before = source.copy()
    result = extended(source)
    assert result.shape == source.shape and not np.shares_memory(result, source)
    np.testing.assert_array_equal(source, before)
    np.testing.assert_array_equal(result[2:-2], source[2:-2])
    np.testing.assert_array_equal(result[0], source[2])
    np.testing.assert_array_equal(result[-1], source[-3])
    assert content(source).shape == (1076, 1444)
    assert border_rows(384) == 1 and border_rows(48) == 0


def test_structure_border_errors_use_true_content_edge_lengths():
    row = [80, 50, 0, 20, 20, 10, 10, 12_000_000, 13_000_000, 15_000_000, 15_000_000, 20, 30]
    counts = np.array([row + [0, 0, 22, 0], row + [0, 0, 0, 0]], np.uint64)
    metrics = structure_metrics_from_counts(counts, 100, 64, 48, 2)
    assert metrics["evaluated_pixels"][0] == 64 * 44
    assert metrics["frame_edge_error"][0] == 0.5
    np.testing.assert_allclose(
        metrics["structure_objective"][1] - metrics["structure_objective"][0], 0.05, rtol=0, atol=1e-15
    )


def test_structure_acceptance_is_a_fourth_independent_floor():
    values = dict(
        miou=np.array([0.9, 0.9]),
        objective=np.array([0.8, 0.8]),
        precision_objective=np.array([0.75, 0.75]),
        structure_objective=np.array([0.59, 0.61]),
    )
    baseline = dict(native_miou=0.8, native_objective=0.7, precision_objective=0.65, structure_objective=0.6)
    np.testing.assert_array_equal(admissible(values, baseline), [False, True])


def test_content_cache_preserves_all17counts_and_does_not_rescore_duplicates():
    class FakeCuda:
        def __init__(self):
            self.views = 0

        def score_structure(self, geometry, query, tolerance, border_y, edge_weight):
            self.views += len(geometry)
            h, w = query.shape
            rows = []
            for geo in geometry:
                rows.append(
                    [
                        80,
                        50,
                        0,
                        20,
                        20,
                        10,
                        10,
                        12_000_000,
                        13_000_000,
                        15_000_000,
                        15_000_000,
                        20,
                        30,
                        0,
                        0,
                        int(geo[4]) * 4,
                        0,
                    ]
                )
            return structure_metrics_from_counts(
                np.array(rows, np.uint64), int(content(query, border_y).sum()), w, h, border_y, edge_weight
            )

    target = np.zeros((384, 512), np.uint8)
    target[20:200, 40:250] = 1
    geos = np.array([[12, 30, 20, 15, 0], [12, 30, 20, 15, 1], [12, 30, 20, 15, 0]], float)
    cuda = FakeCuda()
    cache = FrameScoreCache(target, structure=True)
    cached = score_batches(cuda, geos, target, cache=cache, structure=True)
    plain = score_batches(FakeCuda(), geos, target, structure=True)
    for key in plain:
        np.testing.assert_array_equal(cached[key], plain[key])
    assert cuda.views == 2
    other = score_batches(cuda, geos[::-1], target, cache=cache, structure=True)
    for key in plain:
        np.testing.assert_array_equal(other[key], plain[key][::-1])
    assert cuda.views == 2


def test_real_photo_fallback_rejects_a_lower_structure_score(monkeypatch, tmp_path):
    import badapple_coastline.native_imagery as module

    for key in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "GDAL_NUM_THREADS",
        "GDAL_CACHEMAX",
        "COASTLINE_NATIVE_PATCH_CACHE",
        "COASTLINE_CATALOG_READ_CACHE",
        "COASTLINE_CATALOG_WRITE_CACHE",
        "COASTLINE_LANDSAT_CATALOG_CACHE",
    ]:
        monkeypatch.setenv(key, module.os.environ.get(key, ""))
    parent = dict(longitude=12.0, latitude=30.0, width_km=20.0, rotation_clockwise=0.0, satellite={"scenes": []})
    monkeypatch.setattr(module, "baseline_records", lambda path: [parent])
    monkeypatch.setattr(module, "nearest_place", lambda *a: {"name": "test"})
    used = []

    def sentinel(record, *a, **kw):
        used.append(record["longitude"])
        return {"coverage": 1.0, "path": "test", "scenes": []}

    monkeypatch.setattr(module, "sentinel", sentinel)
    baseline = dict(native_miou=0.8, native_objective=0.7, precision_objective=0.65, structure_objective=0.6)
    record = dict(
        parent,
        frame=0,
        native_miou=0.9,
        native_objective=0.85,
        precision_objective=0.8,
        structure_objective=0.5,
        baseline_fullres_metrics=baseline,
    )
    alternate = dict(
        parent,
        longitude=14.0,
        native_miou=0.88,
        native_objective=0.8,
        precision_objective=0.75,
        structure_objective=0.7,
    )
    record["alternatives"] = [alternate]
    result = module.worker(record, str(tmp_path / "imagery"), "irrelevant")
    assert used == [14.0] and result["imagery_candidate_rank"] == 1


def test_white_content_with_black_letterbox_skips_descriptor_search(tmp_path):
    from badapple_coastline.precision_search import Search
    from badapple_coastline.precision_config import KEYS

    class FakeCuda:
        def score_structure(self, geometry, query, tolerance, border_y, edge_weight):
            n = len(geometry)
            pixels = 1444 * 1076
            rows = np.tile(np.array([pixels, pixels, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], np.uint64), (n, 1))
            return structure_metrics_from_counts(rows, pixels, 1444, 1080, 2, edge_weight)

    source = np.ones((1080, 1444), np.uint8) * 255
    source[:2] = 0
    source[-2:] = 0
    search = Search.__new__(Search)
    geo = np.array([[134.0, -25.0, 2.0, 0.0, 0.0]])
    search.base = [dict(zip(KEYS, geo[0]))]
    search.raw = source[None]
    search.constant_geos = {1: geo}
    search.finish = lambda *a, **kw: {"native_miou": 1.0}
    result = search.global_frame(FakeCuda(), 0, search.base)
    assert result["constant_content"] and result["native_miou"] == 1


def test_constant_photo_workers_reuse_serially_warmed_real_view(monkeypatch, tmp_path):
    import hashlib, json
    from PIL import Image
    from badapple_coastline.data import digest
    import badapple_coastline.native_imagery as module

    monkeypatch.setattr(module, "SIZE", (80, 60))
    monkeypatch.setattr(module, "resolve_interval", lambda r: "test-interval")
    monkeypatch.setattr(
        module,
        "earth_catalog",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("Cached view should not query network")),
    )
    record = dict(longitude=134.0, latitude=-25.0, width_km=2.0, rotation_clockwise=0.0, constant_content=True)
    key = hashlib.sha256(
        json.dumps([[134.0, -25.0, 2.0, 0.0], (80, 60), "test-interval", "EarthSearch-native-s2-v1"]).encode()
    ).hexdigest()[:24]
    directory = tmp_path / "process-123"
    directory.mkdir()
    y, x = np.indices((60, 80))
    pixels = np.stack([30 + x % 80, 70 + y % 50, 30 + x % 40], axis=2).astype(np.uint8)
    path = directory / f"{key}.png"
    Image.fromarray(pixels).save(path)
    metadata = dict(path=str(path), image_sha256=digest(path), coverage=1.0, size=[80, 60], scenes=[])
    (directory / f"{key}.json").write_text(json.dumps(metadata))
    actual = module.sentinel(record, tmp_path / "process-456")
    assert actual["path"] == str(path) and actual["image_sha256"] == metadata["image_sha256"]
