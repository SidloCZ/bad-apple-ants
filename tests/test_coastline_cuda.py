"""Metric and real-photo geometry contracts, independent of installed GPU hardware."""

import numpy as np
import pytest
from PIL import Image
from badapple_coastline.cuda_backend import metrics_from_counts
from badapple_coastline.data import digest


def test_balanced_colors_can_prefer_the_actual_dark_figure_over_white_background():
    counts = np.array([[95, 88, 0, 20, 20, 10, 10], [80, 80, 0, 20, 20, 10, 10]], np.uint64)
    scores = metrics_from_counts(counts, 90, 100)
    assert scores["land_iou"][0] > scores["land_iou"][1]
    assert scores["water_iou"][1] > scores["water_iou"][0]
    assert scores["miou"][1] > scores["miou"][0]
    assert scores["objective"][1] > scores["objective"][0]


def test_empty_regions_and_contours_are_defined_and_missing_data_is_rejected():
    counts = np.array([[0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 4, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0]], np.uint64)
    scores = metrics_from_counts(counts, 0, 100)
    assert scores["miou"][0] == scores["boundary_f1"][0] == scores["objective"][0] == 1
    assert scores["boundary_f1"][1] == 0
    assert scores["objective"][2] == -1


def test_existing_photograph_is_cropped_by_its_geographic_transform(monkeypatch, tmp_path):
    import badapple_coastline.native_imagery as module

    monkeypatch.setattr(module, "SIZE", (80, 60))
    source = tmp_path / "photo.png"
    x, y = np.meshgrid(np.arange(80), np.arange(60))
    pixels = np.stack([x * 2, y * 2, np.full_like(x, 40)], axis=2).astype(np.uint8)
    Image.fromarray(pixels).save(source)
    parent = dict(
        longitude=12.0,
        latitude=30.0,
        width_km=20.0,
        rotation_clockwise=20.0,
        satellite=dict(path=str(source), image_sha256=digest(source), coverage=1.0, scenes=[]),
    )
    child = dict(parent, width_km=10.0)
    result = module.baseline_crop(child, parent, tmp_path / "crop")
    actual = np.asarray(Image.open(result["path"]))
    expected = np.asarray(Image.fromarray(pixels).resize((80, 60), Image.Resampling.BICUBIC, box=(20, 15, 60, 45)))
    assert np.abs(actual.astype(int) - expected.astype(int)).max() <= 1
    assert result["baseline_photo_sha256"] == digest(source)
    with pytest.raises(ValueError, match="outside"):
        module.baseline_crop(dict(child, width_km=40), parent, tmp_path / "outside")


def test_multiband_sentinel_visual_preserves_uint8_and_disables_asset_as_band(monkeypatch, tmp_path):
    from rasterio.io import MemoryFile
    from rasterio.enums import Resampling
    from urllib.parse import urlsplit, parse_qs
    from badapple_coastline.atlas import view_transform
    import badapple_coastline.landsat as module

    monkeypatch.delenv("COASTLINE_NATIVE_PATCH_CACHE", raising=False)
    transform = view_transform(12, 30, 20, 0, (8, 6))
    pixels = np.arange(144, dtype=np.uint8).reshape(3, 6, 8)
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff", height=6, width=8, count=3, dtype="uint8", crs="EPSG:4326", transform=transform
        ) as target:
            target.write(pixels)
        raw = memory.read()
    requested = []

    def request(url):
        requested.append(url)
        return raw

    monkeypatch.setattr(module, "request_crop", request)
    actual = module.read_bands(
        dict(id="unit-test"),
        transform,
        (8, 6),
        ("visual",),
        Resampling.bilinear,
        tmp_path,
        collection="sentinel-2-l2a",
        dtype="uint8",
        expected_count=3,
    )
    np.testing.assert_array_equal(actual, pixels)
    params = parse_qs(urlsplit(requested[0]).query)
    assert params["asset_as_band"] == ["false"] and params["unscale"] == ["false"]


def test_native_crops_are_losslessly_compressed_and_reused(monkeypatch, tmp_path):
    from rasterio.io import MemoryFile
    from rasterio.enums import Resampling
    import rasterio
    from badapple_coastline.atlas import view_transform
    import badapple_coastline.landsat as module

    transform = view_transform(12, 30, 20, 0, (8, 6))
    pixels = np.arange(144, dtype=np.uint16).reshape(3, 6, 8) + 12000
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff", height=6, width=8, count=3, dtype="uint16", crs="EPSG:4326", transform=transform
        ) as source:
            source.write(pixels)
        raw = memory.read()
    calls = []

    def request(url):
        calls.append(url)
        return raw

    monkeypatch.setattr(module, "request_crop", request)
    for _ in range(2):
        actual = module.read_native_crop(
            dict(id="unit-test"),
            transform,
            (8, 6),
            module.BANDS,
            Resampling.bilinear,
            tmp_path,
            module.COLLECTION,
            "uint16",
        )
        np.testing.assert_array_equal(actual, pixels)
    assert len(calls) == 1
    with rasterio.open(next(tmp_path.glob("*.tif"))) as source:
        np.testing.assert_array_equal(source.read(), pixels)
        assert source.transform == transform and source.crs.to_epsg() == 4326
        assert source.compression.value == "DEFLATE"


def test_nonoverlapping_scene_skips_network_requests(monkeypatch, tmp_path):
    from rasterio.enums import Resampling
    from badapple_coastline.atlas import view_transform
    import badapple_coastline.landsat as module

    def unexpected(url):
        raise AssertionError("A nonoverlapping scene must not be downloaded")

    monkeypatch.setattr(module, "request_crop", unexpected)
    item = dict(
        id="elsewhere",
        geometry={"type": "Polygon", "coordinates": [[[40, 40], [41, 40], [41, 41], [40, 41], [40, 40]]]},
    )
    actual = module.read_native_crop(
        item,
        view_transform(12, 30, 20, 0, (8, 6)),
        (8, 6),
        ("qa_pixel",),
        Resampling.nearest,
        tmp_path,
        module.COLLECTION,
        "uint16",
    )
    assert not actual.any()


def test_remaining_snow_must_stay_on_land_and_clouds_must_avoid_water_and_shore():
    from badapple_coastline.native_imagery import residual_is_safe

    water = np.zeros((8, 8), bool)
    water[4:] = True
    critical = water.copy()
    critical[3] = True
    covered = np.ones((8, 8), bool)
    snow = np.zeros_like(covered)
    covered[1, 1] = False
    assert residual_is_safe(covered, snow, water, critical)
    covered[3, 1] = False
    assert not residual_is_safe(covered, snow, water, critical)
    snow[3, 1] = True
    assert residual_is_safe(covered, snow, water, critical)
    covered[5, 1] = False
    snow[5, 1] = True
    assert not residual_is_safe(covered, snow, water, critical)


def test_small_catalog_footprint_gaps_still_require_and_can_use_complete_real_pixels(monkeypatch, tmp_path):
    import badapple_coastline.native_imagery as module
    from badapple_coastline.atlas import view_transform, corners

    monkeypatch.setattr(module, "SIZE", (8, 6))
    record = dict(longitude=12.0, latitude=30.0, width_km=20.0, rotation_clockwise=0.0)
    t = view_transform(12, 30, 20, 0, (8, 6))
    p = np.array(corners(t, (8, 6)))
    left, bottom = p.min(axis=0)
    right, top = p.max(axis=0)
    left += (right - left) * 0.01
    item = dict(
        id="real-source",
        collection="sentinel-2-l2a",
        properties={"datetime": "2020-08-01T00:00:00Z", "eo:cloud_cover": 0},
        assets={"SCL": {"href": "qa"}, "visual": {"href": "rgb"}},
        geometry={
            "type": "Polygon",
            "coordinates": [[[left, bottom], [right, bottom], [right, top], [left, top], [left, bottom]]],
        },
    )
    monkeypatch.setattr(module, "earth_catalog", lambda *a, **kw: {"features": [item]})
    calls = []

    def project(asset, transform, size, bands, resampling, force_native=False):
        assert force_native
        calls.append(asset["href"])
        return np.full((len(bands), 6, 8), 4 if asset["href"] == "qa" else 80, np.uint8)

    monkeypatch.setattr(module, "project_asset", project)
    result = module.sentinel(record, tmp_path)
    assert calls == ["qa", "rgb"] and result["coverage"] == 1
    assert (np.asarray(Image.open(result["path"])) == 80).all()
