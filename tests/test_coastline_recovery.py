"""Recovery of real-image reuse and completed monochrome production."""

import json
import numpy as np
import pytest
from PIL import Image
from badapple_coastline.data import digest


@pytest.fixture
def imagery(monkeypatch):
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
    return module


def test_subpixel_geographic_roundoff_does_not_reject_an_existing_photo(imagery, tmp_path):
    from badapple_coastline.atlas import view_transform, corners

    source = tmp_path / "real-photo-fixture.png"
    Image.new("RGB", imagery.SIZE, (60, 80, 100)).save(source)
    parent = dict(
        longitude=-75.6075487880461,
        latitude=-5.11226578745735,
        width_km=3.0303,
        rotation_clockwise=100.85,
        satellite=dict(path=str(source), image_sha256=digest(source), coverage=1.0, scenes=[]),
    )
    child = dict(parent, longitude=parent["longitude"] - 3e-11)
    transform = view_transform(
        child["longitude"], child["latitude"], child["width_km"], child["rotation_clockwise"], imagery.SIZE
    )
    result = imagery.baseline_crop(child, parent, tmp_path / "crop")
    actual = np.asarray(Image.open(result["path"]))
    assert (actual == np.array([60, 80, 100], np.uint8)).all()
    assert result["coverage"] == 1.0 and result["footprint_lonlat"] == corners(transform, imagery.SIZE)
    with pytest.raises(ValueError, match="outside"):
        imagery.baseline_crop(dict(child, width_km=3.04), parent, tmp_path / "outside")


@pytest.mark.parametrize("sentinel_available", [True, False])
def test_failed_photo_reuse_retrieves_same_view_before_giving_up(imagery, monkeypatch, tmp_path, sentinel_available):
    import badapple_coastline.landsat as landsat

    parent = dict(longitude=12.0, latitude=30.0, width_km=20.0, rotation_clockwise=0.0, satellite={"scenes": []})
    monkeypatch.setattr(imagery, "baseline_records", lambda path: [parent])
    monkeypatch.setattr(imagery, "nearest_place", lambda *a: {"name": "test"})

    def unusable(*a, **kw):
        raise ValueError("Baseline crop would require pixels outside its real image")

    monkeypatch.setattr(imagery, "baseline_crop", unusable)
    calls = []

    def sentinel(view, *a, **kw):
        calls.append(("sentinel", view["longitude"], view["width_km"]))
        if not sentinel_available:
            raise RuntimeError("No clear Sentinel view")
        return dict(path="actual-sentinel-fixture", coverage=1.0, scenes=[])

    def fetch(view, *a, **kw):
        calls.append(("landsat", view["longitude"], view["width_km"]))
        return dict(path="actual-landsat-fixture", coverage=1.0, scenes=[])

    monkeypatch.setattr(imagery, "sentinel", sentinel)
    monkeypatch.setattr(landsat, "fetch_window", fetch)
    record = dict(
        parent,
        frame=0,
        baseline_photo_frame=0,
        native_objective=0.9,
        native_miou=0.9,
        precision_objective=0.9,
        structure_objective=0.8,
        baseline_fullres_metrics=dict(
            native_objective=0.8, native_miou=0.8, precision_objective=0.8, structure_objective=0.7
        ),
    )
    result = imagery.worker(record, str(tmp_path / "imagery"), "irrelevant")
    assert calls == [("sentinel", 12.0, 20.0)] + ([] if sentinel_available else [("landsat", 12.0, 20.0)])
    assert result["imagery_candidate_rank"] == 0 and result["landsat_replacement"] == (not sentinel_available)
    assert result["imagery_errors"][0]["source"] == "baseline reuse"
    assert "baseline_photo_frame" not in result


def test_alternative_view_does_not_inherit_another_candidates_photo_parent(imagery, monkeypatch, tmp_path):
    parent = dict(longitude=12.0, latitude=30.0, width_km=20.0, rotation_clockwise=0.0, satellite={"scenes": []})
    monkeypatch.setattr(imagery, "baseline_records", lambda path: [parent])
    monkeypatch.setattr(imagery, "nearest_place", lambda *a: {"name": "test"})

    def sentinel(view, *a, **kw):
        assert view["longitude"] == 14.0 and "baseline_photo_frame" not in view
        return dict(path="actual-alternative-fixture", coverage=1.0, scenes=[])

    monkeypatch.setattr(imagery, "sentinel", sentinel)
    record = dict(
        parent,
        frame=0,
        baseline_photo_frame=0,
        native_objective=0.5,
        native_miou=0.5,
        precision_objective=0.5,
        baseline_fullres_metrics=dict(native_objective=0.8, native_miou=0.8, precision_objective=0.8),
    )
    record["alternatives"] = [
        dict(parent, longitude=14.0, native_objective=0.9, native_miou=0.9, precision_objective=0.9)
    ]
    result = imagery.worker(record, str(tmp_path / "imagery"), "irrelevant")
    assert result["imagery_candidate_rank"] == 1 and "baseline_photo_frame" not in result


def completed_monochrome(output):
    source = output / "original.mp4"
    source.write_bytes(b"original fixture")
    settings = dict(
        frames=2, match_run_id="completed-search-fixture", video_source=str(source), video_source_sha256=digest(source)
    )
    summary = dict(frames=2, match_run_id=settings["match_run_id"])
    for name, key in [
        ("badapple-land-water.mp4", "movie_sha256"),
        ("badapple-land-water-lossless.mkv", "lossless_sha256"),
        ("badapple-land-water-comparison.mp4", "comparison_sha256"),
    ]:
        path = output / name
        path.write_bytes(name.encode())
        summary[key] = digest(path)
    for key in ["lossless_pixel_audit", "mp4_silhouette_audit", "comparison_left_audit", "comparison_right_audit"]:
        summary[key] = dict(frames=2, decode_errors=0, pixel_hash_differences=0)
    for name, value in [
        ("run.json", settings),
        ("monochrome-summary.json", summary),
        ("monochrome-native-verification.json", dict(frames=2, match_run_id=settings["match_run_id"])),
    ]:
        (output / name).write_text(json.dumps(value))
    (output / "matches.jsonl").write_text("completed fixture\n")


def test_satellite_resume_preserves_completed_search_and_monochrome(monkeypatch, tmp_path):
    import badapple_coastline.precision_pipeline as module

    completed_monochrome(tmp_path)
    monkeypatch.setattr(module, "OUTPUT", tmp_path)
    for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS", "GDAL_NUM_THREADS"]:
        monkeypatch.setenv(key, module.os.environ.get(key, ""))
    commands = []
    monkeypatch.setattr(module.subprocess, "run", lambda args, **kwargs: commands.append(args))
    module.run(include_satellite_photos=True, resume_satellite=True)
    assert [c[3] if c[2] == "-m" else c[2] for c in commands] == [
        "badapple_coastline.native_imagery",
        "badapple_coastline.retry_native_primary",
        "badapple_coastline.native_finalize",
        "scripts/preview_fullres_coastline.py",
        "scripts/build_coastline_youtube.py",
    ]


def test_satellite_resume_rejects_a_changed_monochrome_movie(tmp_path):
    import badapple_coastline.precision_pipeline as module

    completed_monochrome(tmp_path)
    (tmp_path / "badapple-land-water.mp4").write_bytes(b"changed")
    with pytest.raises(ValueError, match="movie changed"):
        module.verify_satellite_resume(tmp_path)
