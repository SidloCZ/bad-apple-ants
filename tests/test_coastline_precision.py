import numpy as np
from badapple_coastline.cuda_backend import shape_metrics_from_counts
from badapple_coastline.precision_config import geometry, WIDTHS, ANGLES, contour_features, FEATURE_SIZE
from badapple_coastline.precision_search import diverse, admissible, vector, both_handedness


def test_same_area_better_contour_wins_and_empty_cases_are_defined():
    rows = np.array(
        [
            [80, 75, 0, 20, 20, 10, 10, 18_000_000, 18_000_000, 19_000_000, 19_000_000, 0, 0],
            [80, 75, 0, 20, 20, 10, 10, 5_000_000, 5_000_000, 10_000_000, 10_000_000, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        np.uint64,
    )
    metrics = shape_metrics_from_counts(rows, 0, 100)
    assert metrics["miou"][0] == metrics["miou"][1]
    assert metrics["precision_objective"][0] > metrics["precision_objective"][1]
    assert metrics["precision_objective"][2] == 1
    assert metrics["precision_objective"][3] == -1


def test_large_dictionary_identifier_decodes_without_overflow():
    centers = np.array([[12.0, 30.0], [14.0, 35.0]])
    i = (len(WIDTHS) * len(centers) * len(ANGLES)) - 1
    view = geometry([i], centers)[0]
    np.testing.assert_array_equal(view[:2], centers[-1])
    assert view[2] == WIDTHS[-1] and view[3] == ANGLES[-1]


def test_three_independent_acceptance_floors_are_enforced():
    metrics = dict(
        miou=np.array([0.81, 0.79, 0.81, 0.81]),
        objective=np.array([0.71, 0.71, 0.69, 0.71]),
        precision_objective=np.array([0.65, 0.65, 0.65, 0.63]),
    )
    expected = admissible(metrics, dict(native_miou=0.80, native_objective=0.70, precision_objective=0.64))
    np.testing.assert_array_equal(expected, [True, False, False, False])


def test_geographic_diversity_wraps_the_dateline_and_handles_empty_input():
    geos = np.array([[179.999, 30, 10, 0], [-179.999, 30, 10, 0], [140, 30, 10, 0]], float)
    np.testing.assert_array_equal(diverse(geos, np.array([0.9, 0.8, 0.7]), 3), [0, 2])
    assert len(diverse(np.empty((0, 4)), np.empty(0), 3)) == 0


def test_directional_features_have_no_false_constant_boundary():
    values = np.full((2, np.prod(FEATURE_SIZE)), 16, np.int8)
    assert not contour_features(values).any()


def test_reusing_another_frames_photo_keeps_its_actual_geography(monkeypatch, tmp_path):
    from PIL import Image
    from badapple_coastline.data import digest
    from badapple_coastline.atlas import corners, view_transform
    import badapple_coastline.native_imagery as module

    monkeypatch.setattr(module, "SIZE", (80, 60))
    # Production workers are isolated processes; this in-process test must
    # restore their environment so later raw-source tests do not share caches.
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
    parents = []
    for i, lon in enumerate([12.0, 20.0]):
        path = tmp_path / f"photo{i}.png"
        Image.new("RGB", (80, 60), (60 + i * 30, 70, 80)).save(path)
        parents.append(
            dict(
                longitude=lon,
                latitude=30.0,
                width_km=20.0,
                rotation_clockwise=0.0,
                landsat_replacement=bool(i),
                satellite=dict(
                    path=str(path),
                    image_sha256=digest(path),
                    coverage=1.0,
                    size=[80, 60],
                    scenes=[],
                    footprint_lonlat=corners(view_transform(lon, 30, 20, 0, (80, 60)), (80, 60)),
                ),
            )
        )
    monkeypatch.setattr(module, "baseline_records", lambda path: parents)
    monkeypatch.setattr(module, "nearest_place", lambda *a: {"name": "test"})
    record = dict(
        longitude=20.0,
        latitude=30.0,
        width_km=20.0,
        rotation_clockwise=0.0,
        frame=0,
        native_objective=0.9,
        native_miou=0.9,
        precision_objective=0.9,
        baseline_fullres_metrics=dict(native_objective=0.8, native_miou=0.8, precision_objective=0.8),
        baseline_photo_frame=1,
    )
    actual = module.worker(record, str(tmp_path / "imagery"), "irrelevant")
    assert actual["satellite"]["image_sha256"] == parents[1]["satellite"]["image_sha256"]
    assert actual["satellite"]["path"] == parents[1]["satellite"]["path"]
    assert actual["landsat_replacement"]
    assert actual["longitude"] == 20.0


def test_default_production_stops_after_monochrome_without_any_photo_stage(monkeypatch, tmp_path):
    import json
    import badapple_coastline.precision_pipeline as module

    monkeypatch.setattr(module, "OUTPUT", tmp_path)
    for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS", "GDAL_NUM_THREADS"]:
        monkeypatch.setenv(key, module.os.environ.get(key, ""))
    commands = []
    monkeypatch.setattr(module.subprocess, "run", lambda args, **kwargs: commands.append(args))
    module.run()
    assert [c[3] for c in commands] == [
        "badapple_coastline.precision_dictionary",
        "badapple_coastline.precision_search",
        "badapple_coastline.precision_monochrome",
    ]
    result = json.loads((tmp_path / "pipeline-status.json").read_text())
    assert result["phase"] == "complete-monochrome" and result["new_satellite_downloads"] is False


def test_mirrored_candidates_are_distinct_and_old_records_default_to_normal():
    record = dict(longitude=12.0, latitude=30.0, width_km=20.0, rotation_clockwise=15.0)
    geos = both_handedness(np.array([vector(record)]))
    np.testing.assert_array_equal(geos[:, 4], [0, 1])
    np.testing.assert_array_equal(diverse(geos, np.array([0.9, 0.8]), 2), [0, 1])


def test_explicit_satellite_request_completes_monochrome_before_real_imagery(monkeypatch, tmp_path):
    import json
    import badapple_coastline.precision_pipeline as module

    monkeypatch.setattr(module, "OUTPUT", tmp_path)
    for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS", "GDAL_NUM_THREADS"]:
        monkeypatch.setenv(key, module.os.environ.get(key, ""))
    commands = []
    monkeypatch.setattr(module.subprocess, "run", lambda args, **kwargs: commands.append(args))
    module.run(workers=48, include_satellite_photos=True)
    names = [c[3] if c[2] == "-m" else c[2] for c in commands]
    assert names == [
        "badapple_coastline.precision_dictionary",
        "badapple_coastline.precision_search",
        "badapple_coastline.precision_monochrome",
        "badapple_coastline.native_imagery",
        "badapple_coastline.retry_native_primary",
        "badapple_coastline.native_finalize",
        "scripts/preview_fullres_coastline.py",
        "scripts/build_coastline_youtube.py",
    ]
    assert commands[3][-2:] == ["--workers", "48"]
    assert json.loads((tmp_path / "pipeline-status.json").read_text())["phase"] == "complete"


def test_mirrored_existing_photo_and_geographic_corners_agree(monkeypatch, tmp_path):
    from PIL import Image
    from badapple_coastline.atlas import view_transform, corners
    from badapple_coastline.data import digest
    from badapple_coastline.landsat import key_for
    import badapple_coastline.native_imagery as module

    monkeypatch.setattr(module, "SIZE", (80, 60))
    y, x = np.indices((60, 80))
    pixels = np.stack([x * 2, y * 2, np.full_like(x, 40)], axis=2).astype(np.uint8)
    source = tmp_path / "source.png"
    Image.fromarray(pixels).save(source)
    parent = dict(
        longitude=12.0,
        latitude=30.0,
        width_km=20.0,
        rotation_clockwise=15.0,
        satellite=dict(path=str(source), image_sha256=digest(source), coverage=1.0, scenes=[]),
    )
    mirrored = dict(parent, mirror_horizontal=True)
    result = module.baseline_crop(mirrored, parent, tmp_path / "cache")
    np.testing.assert_array_equal(np.asarray(Image.open(result["path"])), pixels[:, ::-1])
    original = corners(view_transform(12, 30, 20, 15, (80, 60)), (80, 60))
    expected = [original[i] for i in [1, 0, 3, 2]]
    np.testing.assert_allclose(result["footprint_lonlat"], expected, rtol=0, atol=1e-12)
    assert key_for(parent) != key_for(mirrored)
    assert key_for(parent) == key_for(dict(parent, mirror_horizontal=False))


def test_frame_score_cache_reuses_only_exact_geometry_and_preserves_all_metrics():
    from badapple_coastline.precision_search import FrameScoreCache, score_batches
    from badapple_coastline.cuda_backend import CudaAtlas

    class FakeCuda:
        def __init__(self):
            self.scored = []

        def score_shape(self, geometry, target, tolerance):
            _, source, mirrors = CudaAtlas.split_geometry(geometry)
            self.scored.extend([tuple(row) + (int(mirror),) for row, mirror in zip(source, mirrors)])
            rows = []
            for row, mirror in zip(source, mirrors):
                value = 10 + int(row[0] > 12) + int(mirror)
                rows.append([value, 5, 0, 8, 6, 4, 3, 2_000_000, 3_000_000, 4_000_000, 5_000_000, 12, 14])
            return shape_metrics_from_counts(np.array(rows, np.uint64), int(target.sum()), target.size)

    target = np.zeros((48, 64), np.uint8)
    target[8:30, 10:40] = 1
    a = [12.0, 30.0, 20.0, 15.0, 0.0]
    b = [12.0, 30.0, 20.0, 15.0, 1.0]
    c = [np.nextafter(12.0, 13.0), 30.0, 20.0, 15.0, 0.0]
    views = np.array([a, b, a, c])
    reference = score_batches(FakeCuda(), views, target, batch=2)
    cuda = FakeCuda()
    cache = FrameScoreCache(target)
    actual = score_batches(cuda, views, target, batch=2, cache=cache)
    for key in reference:
        np.testing.assert_array_equal(actual[key], reference[key])
    assert len(cuda.scored) == 3
    actual = score_batches(cuda, views[::-1], target, batch=1, cache=cache)
    for key in reference:
        np.testing.assert_array_equal(actual[key], reference[key][::-1])
    assert len(cuda.scored) == 3
    # Legacy four-column geometry and explicit mirror=False identify the same
    # native view, while a single representable-coordinate step stays distinct.
    score_batches(cuda, np.array([a[:4]]), target, cache=cache)
    assert len(cuda.scored) == 3


def test_frame_score_cache_rejects_a_different_target():
    import pytest
    from badapple_coastline.precision_search import FrameScoreCache, score_batches

    target = np.zeros((48, 64), np.uint8)
    cache = FrameScoreCache(target)
    target[0, 0] = 1
    with pytest.raises(ValueError, match="another target"):
        score_batches(None, np.array([[12.0, 30.0, 20.0, 15.0]]), target, cache=cache)
