import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("rasterio")

from badapple_coastline.atlas import Atlas, corners, prepare_atlas, view_transform
from badapple_coastline.matching import fft_best_translation
from badapple_coastline.satellite import fetch_window


def test_fft_mse_matches_brute_force_and_excludes_missing_pixels():
    rng = np.random.default_rng(62)
    scene = rng.integers(0, 2, (17, 23)).astype(np.float32)
    query = scene[8:13, 11:18].copy()
    x, y, score = fft_best_translation(scene, query)
    assert (x, y) == (11, 8)
    assert score == pytest.approx(0, abs=1e-6)
    valid = np.ones(scene.shape, bool)
    valid[8, 11] = False
    actual = fft_best_translation(scene, query, valid)
    scores = np.full((13, 17), np.inf)
    for y in range(13):
        for x in range(17):
            if valid[y : y + 5, x : x + 7].all():
                scores[y, x] = np.mean((scene[y : y + 5, x : x + 7] - query) ** 2)
    assert actual[2] == pytest.approx(scores.min(), abs=1e-6)
    assert scores[actual[1], actual[0]] == pytest.approx(scores.min())


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_constant_frames_do_not_need_division_by_variance(value):
    scene = np.full((12, 15), value, dtype=np.float32)
    x, y, score = fft_best_translation(scene, np.full((4, 5), value, dtype=np.float32))
    assert np.isfinite(score) and score == pytest.approx(0, abs=1e-6)
    _, _, missing = fft_best_translation(scene, scene[:4, :5], np.zeros(scene.shape, bool))
    assert missing == np.inf


def test_geographic_transform_preserves_center_scale_and_clockwise_rotation():
    for angle in [0, 30, 90, 270]:
        transform = view_transform(139.7, 35.5, 48, angle, (32, 24))
        assert transform @ (16, 12) == pytest.approx((139.7, 35.5))
        left, right = transform @ (0, 12), transform @ (32, 12)
        east = (right[0] - left[0]) * 111.32 * np.cos(np.deg2rad(35.5))
        north = (right[1] - left[1]) * 111.32
        assert np.hypot(east, north) == pytest.approx(48)
        assert north == pytest.approx(-48 * np.sin(np.deg2rad(angle)), abs=1e-8)
        assert len(corners(transform, (32, 24))) == 4


def test_fft_iou_matches_direct_window_scores():
    rng = np.random.default_rng(271)
    scene = rng.integers(0, 2, (13, 17)).astype(np.float32)
    query = rng.random((4, 6)).astype(np.float32)
    x, y, distance = fft_best_translation(scene, query, metric="iou")
    brute = []
    for row in range(10):
        for col in range(12):
            patch = scene[row : row + 4, col : col + 6]
            intersect = float((patch * query).sum())
            brute.append(1 - intersect / float(patch.sum() + query.sum() - intersect))
    assert distance == pytest.approx(min(brute), abs=1e-6)
    for v in [0.0, 1.0]:
        assert fft_best_translation(np.full((10, 12), v, np.float32), np.full((3, 4), v, np.float32), metric="iou")[
            2
        ] == pytest.approx(0, abs=1e-5)


def test_scale_search_retains_distinct_rotations_and_geographic_locations():
    from badapple_coastline.matching import _diverse_indices

    candidates = np.array([[0, 0, 24, 0], [0, 0, 24, 30], [0, 0, 24, 90], [1, 0, 24, 0], [0, 0, 48, 0]])
    assert _diverse_indices(range(5), candidates, 4) == [0, 2, 3, 4]


def test_native_ranking_never_offers_a_worse_satellite_fallback():
    from badapple_coastline.improvement import retain_improvements, mask_iou

    def candidate(lon, score):
        return dict(longitude=lon, latitude=0, width_km=24, rotation_clockwise=0, native_iou=score)

    baseline = candidate(0, 0.7)
    selected = retain_improvements(
        baseline, [candidate(1, 0.9), candidate(2, 0.6), candidate(3, 0.8), candidate(1, 0.9), candidate(4, 0.7)]
    )
    assert [r["longitude"] for r in selected] == [1, 3, 0]
    assert min(r["native_iou"] for r in selected) == baseline["native_iou"]
    assert mask_iou(np.zeros((3, 4)), np.zeros((3, 4))) == 1
    assert mask_iou(np.eye(4), np.eye(4)) == 1
    assert mask_iou(np.eye(4), 1 - np.eye(4)) == 0


def test_each_scale_gets_candidates_and_duplicate_silhouettes_share_refinement(monkeypatch, tmp_path):
    import badapple_coastline.matching as module
    from badapple_coastline.data import digest

    atlas = tmp_path / "atlas"
    atlas.mkdir()
    (atlas / "atlas.json").write_text("{}")
    dictionary = tmp_path / "dictionary"
    dictionary.mkdir()
    (dictionary / "dictionary.json").write_text(
        json.dumps({"atlas": str(atlas), "atlas_sha256": digest(atlas / "atlas.json"), "global_coverage": True})
    )
    candidates = np.array([[0, 0, 6, 0], [1, 0, 6, 0], [0, 0, 768, 0], [1, 0, 768, 0]], np.float64)
    np.save(dictionary / "candidates.npy", candidates)
    query = np.eye(24, 32, dtype=np.float32)
    np.save(dictionary / "features.npy", np.stack([module.feature(query)] * 4))
    from types import SimpleNamespace

    monkeypatch.setattr(module, "Atlas", lambda path: SimpleNamespace(path=path))
    calls = []

    def refinement(atlas, q, candidate, metric):
        calls.append(candidate[2])
        return {"distance": 0.2, "width_km": float(candidate[2])}

    monkeypatch.setattr(module, "refine", refinement)
    result = module.match_queries(np.stack([query, query]), dictionary, per_scale=1, workers=2)
    assert sorted(calls) == [6, 768]
    assert [r["frame"] for r in result] == [0, 1]
    assert {result[0]["width_km"], result[0]["alternatives"][0]["width_km"]} == {6, 768}


def test_rust_refinement_matches_python_reference_and_recorded_geography(tmp_path):
    from badapple_coastline import rust_backend
    from badapple_coastline.matching import refine

    try:
        rust_backend.artifacts()
    except FileNotFoundError:
        pytest.skip("Optional Rust engine not built")
    atlas_dir = tmp_path / "atlas"
    atlas_dir.mkdir()
    rng = np.random.default_rng(417)
    np.save(atlas_dir / "land.npy", rng.integers(0, 2, (360, 720), dtype=np.uint8))
    (atlas_dir / "atlas.json").write_text(json.dumps({"resolution_degrees": 0.5}))
    atlas = Atlas(atlas_dir)
    candidate = (10.0, 20.0, 2000.0, 30.0)
    query, _ = atlas.sample(*candidate)
    rotations = (-7.5, 0.0, 7.5)
    scales = (0.85, 1.0, 1.18)
    python = refine(atlas, query, candidate, rotations=rotations, scales=scales)
    rust = rust_backend.refine_batch(atlas, query, [candidate], rotations=rotations, scales=scales)[0]
    assert rust["iou"] == pytest.approx(python["iou"], abs=1e-6)
    assert rust["mse"] == pytest.approx(python["mse"], abs=1e-6)
    sampled, valid = atlas.sample(rust["longitude"], rust["latitude"], rust["width_km"], rust["rotation_clockwise"])
    assert valid.all() and np.mean((sampled - query) ** 2) == pytest.approx(rust["mse"], abs=1e-6)
    assert len(rust["footprint_lonlat"]) == 4


def test_partial_atlas_is_unknown_not_sea_and_global_build_refuses_missing(tmp_path):
    import rasterio
    from rasterio.transform import from_origin

    data, dest = tmp_path / "source", tmp_path / "atlas"
    data.mkdir()
    name = "tile.tif"
    with rasterio.open(
        data / name,
        "w",
        driver="GTiff",
        width=60,
        height=60,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_origin(120, 60, 0.5, 0.5),
    ) as ds:
        values = np.ones((60, 60), np.uint8)
        values[:, :10] = 0
        ds.write(values, 1)
    (data / "landWater2020.vrt").write_text(
        "<VRTDataset><VRTRasterBand><SimpleSource><SourceFilename>tile.tif</SourceFilename></SimpleSource><SimpleSource><SourceFilename>missing.tif</SourceFilename></SimpleSource></VRTRasterBand></VRTDataset>"
    )
    with pytest.raises(FileNotFoundError, match="Missing"):
        prepare_atlas(data, dest, 0.5)
    result = prepare_atlas(data, dest, 0.5, allow_partial=True)
    assert result["global_coverage"] is False
    atlas = Atlas(dest)
    values, valid = atlas.sample(135, 45, 20, 0)
    assert valid.all() and (values == 1).all()
    _, valid = atlas.sample(0, 0, 20, 0)
    assert not valid.any()
    from PIL import Image

    native = atlas.native_mask(135, 45, 20, 30, size=(40, 30))
    assert (np.array(Image.open(native)) == 255).all()
    water = atlas.native_mask(122, 45, 20, 30, size=(40, 30))
    assert (np.array(Image.open(water)) == 0).all()
    with pytest.raises(ValueError, match="unacquired"):
        atlas.native_mask(0, 0, 20, 0, size=(40, 30))


def fake_item(identifier, date):
    return {
        "id": identifier,
        "collection": "sentinel-2-l2a",
        "properties": {"datetime": date, "eo:cloud_cover": 1, "s2:mgrs_tile": "54SUE"},
        "assets": {"visual": {"href": identifier + "-rgb"}, "SCL": {"href": identifier + "-scl"}},
    }


def test_satellite_mosaic_uses_real_pixels_actual_year_and_cache_checksum(monkeypatch, tmp_path):
    import badapple_coastline.satellite as module
    from PIL import Image

    items = [fake_item("A", "2020-04-01T00:00:00Z"), fake_item("B", "2021-05-01T00:00:00Z")]
    monkeypatch.setattr(module, "request_json", lambda *args: {"features": items})
    calls = []

    def project(asset, transform, size, bands, resampling):
        calls.append(asset["href"])
        if asset["href"].endswith("scl"):
            scl = np.full((1, 3, 4), 6, np.uint8)
            if asset["href"].startswith("A"):
                scl[0, 0, 0] = 9
            return scl
        return np.full((3, 3, 4), 100 if asset["href"].startswith("A") else 200, np.uint8)

    monkeypatch.setattr(module, "project_asset", project)
    record = {"longitude": 139, "latitude": 35, "width_km": 24, "rotation_clockwise": 30}
    metadata = fetch_window(record, tmp_path, size=(4, 3))
    assert metadata["coverage"] == 1
    assert metadata["credit"] == "Contains modified Copernicus Sentinel data 2020-2021"
    image = np.array(Image.open(metadata["path"]))
    assert (image[0, 0] == 200).all() and (image[1, 1] == 100).all()
    count = len(calls)
    fetch_window(record, tmp_path, size=(4, 3))
    assert len(calls) == count
    monkeypatch.setenv("COASTLINE_SHARED_WINDOW_CACHE", str(tmp_path))
    shared = fetch_window(record, tmp_path / "worker", size=(4, 3))
    assert shared["path"] == metadata["path"] and len(calls) == count
    Path(metadata["path"]).write_bytes(b"modified file")
    fetch_window(record, tmp_path, size=(4, 3))
    assert len(calls) > count
    count = len(calls)
    Path(metadata["path"]).with_suffix(".json").write_text("{interrupted")
    fetch_window(record, tmp_path, size=(4, 3))
    assert len(calls) > count


def test_missing_optical_coverage_never_creates_output_photo(monkeypatch, tmp_path):
    import badapple_coastline.satellite as module

    monkeypatch.setattr(module, "request_json", lambda *args: {"features": []})
    with pytest.raises(RuntimeError, match="no synthetic fill"):
        fetch_window({"longitude": 139, "latitude": 35, "width_km": 24, "rotation_clockwise": 0}, tmp_path, size=(4, 3))
    assert not list(tmp_path.glob("*.png"))
    failure = json.loads(next(tmp_path.glob("*.failure.json")).read_text())
    assert failure["coverage"] == 0


def test_wide_view_catalog_follows_pages_until_spatial_coverage_is_complete(monkeypatch, tmp_path):
    import badapple_coastline.satellite as module
    from rasterio.transform import from_origin

    def item(name, left, right):
        return {
            "id": name,
            "geometry": {"type": "Polygon", "coordinates": [[[left, 0], [right, 0], [right, 6], [left, 6], [left, 0]]]},
        }

    calls = []

    def request(url, body):
        calls.append((url, body))
        if len(calls) == 1:
            return {
                "features": [item("west", 0, 4)],
                "links": [
                    {
                        "rel": "next",
                        "href": module.STAC + "/search",
                        "method": "POST",
                        "body": {"next": "observed-token"},
                    }
                ],
            }
        assert body["next"] == "observed-token"
        assert "assets.visual.href" in body["fields"]["include"]
        return {"features": [item("east", 4, 8)], "links": []}

    monkeypatch.setattr(module, "request_json", request)
    transform = from_origin(0, 6, 0.25, 0.25)
    result = module.catalog([0, 0, 8, 6], "2020/2021", 25, tmp_path, viewport=transform)
    assert [i["id"] for i in result["features"]] == ["west", "east"]
    assert len(calls) == 2
    module.catalog([0, 0, 8, 6], "2020/2021", 25, tmp_path, viewport=transform)
    assert len(calls) == 2


def test_improved_imagery_fallback_reports_the_actual_native_gain(monkeypatch, tmp_path):
    import badapple_coastline.pipeline as module
    from PIL import Image
    from badapple_coastline.data import digest

    output = tmp_path / "output"
    output.mkdir()
    (output / "run.json").write_text(json.dumps({"match_run_id": "wide", "global_search_coverage": True}))

    def candidate(lon, iou):
        return dict(
            longitude=lon,
            latitude=35,
            width_km=24,
            rotation_clockwise=0,
            mse=0.1,
            iou=iou,
            native_iou=iou,
            native_mse=0.1,
        )

    record = dict(
        candidate(1, 0.9),
        frame=0,
        match_run_id="wide",
        baseline_native_iou=0.7,
        native_iou_gain=0.2,
        alternatives=[candidate(0, 0.7)],
    )
    record.update(native_mask_path="primary-view-only.png", native_mask_sha256="primary-mask-hash")
    (output / "matches.jsonl").write_text(json.dumps(record) + "\n")
    photo = tmp_path / "real.png"
    Image.new("RGB", (4, 3), (10, 20, 30)).save(photo)
    monkeypatch.setattr(module, "fetch_places", lambda path: None)
    monkeypatch.setattr(module, "nearest_place", lambda *args: {"name": "Test"})

    def fetch(record, *args, **kwargs):
        if record["longitude"] == 1:
            raise RuntimeError("Unavailable wide view")
        return {
            "path": str(photo),
            "image_sha256": digest(photo),
            "coverage": 1,
            "scenes": [{"datetime": "2020-06-01T00:00:00Z"}],
        }

    monkeypatch.setattr(module, "fetch_window", fetch)
    module.imagery(output, tmp_path / "cache", size=(4, 3), workers=1)
    final = module.read_records(output / "frames.jsonl")[0]
    assert final["imagery_candidate_rank"] == 1
    assert final["native_iou"] == 0.7 and final["native_iou_gain"] == 0
    assert "native_mask_path" not in final and "native_mask_sha256" not in final
    assert json.loads((output / "imagery-status.json").read_text())["actual_regressed_frames"] == 0


def test_image_processes_keep_private_caches_and_merge_in_frame_order(monkeypatch, tmp_path):
    import badapple_coastline.process_images as module
    import badapple_coastline.pipeline as pipeline

    output = tmp_path / "output"
    output.mkdir()
    (output / "run.json").write_text(json.dumps({"match_run_id": "unit-run"}))
    records = [{"frame": i, "match_run_id": "unit-run"} for i in range(5)]
    (output / "matches.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    environments = []

    class Process:
        def __init__(self, command, env, **kwargs):
            environments.append(env)
            shard = Path(command[command.index("--output") + 1])
            subset = pipeline.read_records(shard / "matches.jsonl")
            (shard / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reversed(subset)))

        def poll(self):
            return 0

        def wait(self):
            return 0

        def terminate(self):
            raise AssertionError("Completed process should not be terminated")

    monkeypatch.setattr(module.subprocess, "Popen", Process)

    def normalize(path, *args, **kwargs):
        assert pipeline.read_records(path / "frames.jsonl") == records
        assert kwargs["workers"] == 1
        return {"frames": 5}

    monkeypatch.setattr(pipeline, "imagery", normalize)
    assert module.retrieve(output, tmp_path / "cache", (4, 3), "auto", 2, 2) == {"frames": 5}
    assert len({env["COASTLINE_SCENE_CACHE"] for env in environments}) == 2
    assert all(env["COASTLINE_IMAGE_CHILD"] == "1" and env["GDAL_NUM_THREADS"] == "1" for env in environments)


def test_cloud_fallback_retains_real_photo_and_reports_cloud_area(monkeypatch, tmp_path):
    import badapple_coastline.satellite as module
    from PIL import Image

    monkeypatch.setattr(module, "request_json", lambda *args: {"features": [fake_item("A", "2020-06-01T00:00:00Z")]})

    def project(asset, *args):
        return np.full((1, 3, 4), 9, np.uint8) if asset["href"].endswith("scl") else np.full((3, 3, 4), 123, np.uint8)

    monkeypatch.setattr(module, "project_asset", project)
    r = fetch_window({"longitude": 139, "latitude": 35, "width_km": 24, "rotation_clockwise": 0}, tmp_path, size=(4, 3))
    assert r["coverage"] == 1 and r["cloud_fraction"] == 1 and r["clear_coverage"] == 0
    assert (np.array(Image.open(r["path"])) == 123).all()


def test_retry_honors_server_rate_limit(monkeypatch):
    import io
    import urllib.error
    import badapple_coastline.data as module

    sleeps = []
    calls = []

    def opening(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise urllib.error.HTTPError("https://example.com", 429, "Too Many Requests", {"Retry-After": "3"}, None)
        return io.BytesIO(b'{"ok":true}')

    monkeypatch.setattr(module.urllib.request, "urlopen", opening)
    monkeypatch.setattr(module.time, "sleep", sleeps.append)
    assert module.request_json("https://example.com") == {"ok": True}
    assert sleeps == [3]


def test_resume_ignores_only_an_unfinished_trailing_json_record(tmp_path):
    from badapple_coastline.pipeline import read_records

    partial = tmp_path / "frames.partial.jsonl"
    partial.write_bytes(b'{"frame":0}\n{"frame":2}\n{"frame":')
    assert read_records(partial) == [{"frame": 0}, {"frame": 2}]
    complete = tmp_path / "frames.jsonl"
    complete.write_bytes(partial.read_bytes())
    with pytest.raises(json.JSONDecodeError):
        read_records(complete)


def test_reused_scene_cache_preserves_geography_and_pixel_sampling(monkeypatch, tmp_path):
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from badapple_coastline.satellite import cached_asset

    source = tmp_path / "source.tif"
    monkeypatch.setenv("COASTLINE_SCENE_CACHE", str(tmp_path / "private"))
    values = np.arange(128 * 128, dtype=np.uint8).reshape(128, 128)
    with rasterio.open(
        source,
        "w",
        driver="GTiff",
        width=128,
        height=128,
        count=1,
        dtype="uint8",
        crs="EPSG:32654",
        transform=from_origin(300000, 3900000, 10, 10),
        nodata=0,
    ) as ds:
        ds.write(values, 1)
    for _ in range(3):
        assert cached_asset(str(source), 64, Resampling.nearest, tmp_path / "cache") == str(source)
    cached = cached_asset(str(source), 64, Resampling.nearest, tmp_path / "cache")
    assert Path(cached).parent == tmp_path / "private"
    with rasterio.open(source) as original, rasterio.open(cached) as reduced:
        assert original.bounds == reduced.bounds and original.crs == reduced.crs
        assert reduced.width == reduced.height == 64
        np.testing.assert_array_equal(
            reduced.read(1), original.read(1, out_shape=(64, 64), resampling=Resampling.nearest)
        )


@pytest.mark.skipif(not __import__("shutil").which("ffmpeg"), reason="FFmpeg unavailable")
def test_coastline_encoding_preserves_fractional_rate_frame_count_and_audio(tmp_path):
    import subprocess
    from types import SimpleNamespace
    from PIL import Image
    from badapple_media import FeatureConfig
    from badapple_media import extract_frames
    from badapple_coastline.data import digest
    from badapple_coastline.render import render

    source = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=96x72:rate=30000/1001",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-frames:v",
            "4",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            str(source),
        ],
        check=True,
    )
    frames = extract_frames(source, tmp_path / "frames", FeatureConfig())
    photo = tmp_path / "photo.png"
    Image.new("RGB", (160, 120), (50, 80, 110)).save(photo)
    records = [
        {
            "latitude": 35.0,
            "longitude": 139.0,
            "width_km": 24.0,
            "rotation_clockwise": 0.0,
            "mse": 0.05,
            "place": {"name": "Test settlement", "country": "Test country", "distance_km": 12},
            "satellite": {
                "path": str(photo),
                "coverage": 1,
                "image_sha256": digest(photo),
                "credit": "Contains modified Copernicus Sentinel data 2020",
                "scenes": [{"datetime": "2020-06-01T00:00:00Z"}],
            },
        }
        for _ in range(4)
    ]
    atlas = SimpleNamespace(
        metadata={"global_coverage": False}, sample=lambda *args: (np.zeros((195, 260), np.float32), None)
    )
    result = render(source, frames, records, atlas, tmp_path / "coastline.mp4")
    assert result["frames"] == 4 and result["fps"] == "30000/1001" and result["has_audio"]
    assert result["duration"] == pytest.approx(4 * 1001 / 30000, abs=0.001)
    assert (tmp_path / "coastline.poster.png").is_file()
