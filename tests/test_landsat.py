import numpy as np
from pathlib import Path
import pytest
from PIL import Image

from badapple_coastline.landsat import fetch_window, natural_color, qa_masks


def test_landsat_uint16_scale_offset_and_fixed_srgb_transfer():
    dn = np.array([0, 7273, 18639, 43636], np.uint16)
    rgb = natural_color(dn)
    assert rgb.dtype == np.uint8 and rgb[0] == 0 and rgb[1] <= 1 and rgb[-1] == 255
    expected = (1.055 * (18639 * 0.0000275 - 0.2) ** (1 / 2.4) - 0.055) * 255
    assert abs(int(rgb[2]) - expected) < 0.51


def test_blob_signing_reuses_container_token_and_refreshes_expired_token(monkeypatch):
    import badapple_coastline.satellite as module

    monkeypatch.setattr(module, "_sas_tokens", {})
    calls = []

    def token(url):
        calls.append(url)
        return {"msft:expiry": "2035-01-01T00:00:00Z", "token": "sp=r&sig=unit-test"}

    monkeypatch.setattr(module, "request_json", token)
    first = "https://unit.blob.core.windows.net/landsat-c2/first.TIF"
    second = "https://unit.blob.core.windows.net/landsat-c2/second.TIF"
    assert module.signed_href(first) == first + "?sp=r&sig=unit-test"
    assert module.signed_href(second) == second + "?sp=r&sig=unit-test"
    assert len(calls) == 1
    module._sas_tokens[("unit", "landsat-c2")]["msft:expiry"] = "2000-01-01T00:00:00Z"
    module.signed_href(second)
    assert len(calls) == 2
    signed = first + "?sp=r&sig=already-signed"
    assert module.signed_href(signed) == signed and len(calls) == 2


def test_qa_keeps_water_and_rejects_fill_cloud_snow_without_truncating_bits():
    qa = np.array([0, 1, 64, 192, 64 | 8, 64 | 32, 64 | 16, 21824], np.uint16)
    valid, clear = qa_masks(qa)
    assert valid.tolist() == [False, False, True, True, True, True, True, True]
    assert clear.tolist() == [False, False, True, True, False, False, False, True]


def test_landsat_replaces_cloud_pixels_with_another_real_scene(monkeypatch, tmp_path):
    import badapple_coastline.landsat as module

    def item(name):
        return dict(
            id=name,
            collection=module.COLLECTION,
            geometry={"type": "Polygon", "coordinates": [[[0, 0], [20, 0], [20, 80], [0, 80], [0, 0]]]},
            properties={"datetime": "2020-07-01T00:00:00Z", "eo:cloud_cover": 0, "platform": "landsat-8"},
            assets={k: {"href": name + "-" + k} for k in ("red", "green", "blue", "qa_pixel")},
        )

    monkeypatch.setattr(module, "catalog", lambda *args: [item("first"), item("second")])

    def project(item, transform, size, assets, resampling, cache):
        first = item["id"] == "first"
        if assets == ("qa_pixel",):
            qa = np.full((1, 3, 4), 21824, np.uint16)
            if first:
                qa[0, 0, 0] |= 8
            return qa
        return np.full((3, 3, 4), 12000 if first else 16000, np.uint16)

    monkeypatch.setattr(module, "read_bands", project)
    record = dict(longitude=10, latitude=60, width_km=12, rotation_clockwise=0)
    result = fetch_window(record, tmp_path, size=(4, 3))
    image = np.asarray(Image.open(result["path"]))
    assert result["coverage"] == 1 and result["clear_coverage"] == 1
    assert result["source_resolution_m"] == 30
    assert image[0, 0, 0] == natural_color(np.array([16000], np.uint16))[0]
    assert image[1, 1, 0] == natural_color(np.array([12000], np.uint16))[0]
    assert len(result["scenes"]) == 2 and "Geological Survey" in result["credit"]


def test_official_crop_keeps_uint16_pixels_and_geographic_transform(monkeypatch, tmp_path):
    from rasterio.io import MemoryFile
    from rasterio.enums import Resampling
    from badapple_coastline.atlas import view_transform
    import badapple_coastline.landsat as module

    transform = view_transform(10, 60, 12, 0, (4, 3))
    pixels = np.arange(36, dtype=np.uint16).reshape(3, 3, 4) + 12000

    def tiff(data):
        with MemoryFile() as memory:
            with memory.open(
                driver="GTiff", height=3, width=4, count=3, dtype=data.dtype, crs="EPSG:4326", transform=transform
            ) as sink:
                sink.write(data)
            return memory.read()

    requested = []

    def request(url):
        requested.append(url)
        return tiff(pixels)

    monkeypatch.setattr(module, "request_crop", request)
    actual = module.read_bands({"id": "test"}, transform, (4, 3), module.BANDS, Resampling.bilinear, tmp_path)
    np.testing.assert_array_equal(actual, pixels)
    assert "unscale=false" in requested[0] and "return_mask=false" in requested[0]
    module.read_bands({"id": "test"}, transform, (4, 3), module.BANDS, Resampling.bilinear, tmp_path)
    assert len(requested) == 1
    bad = tmp_path / "bad"
    bad.mkdir()
    monkeypatch.setattr(module, "request_crop", lambda url: tiff(np.uint8(pixels)))
    with pytest.raises(ValueError, match="uint16"):
        module.read_bands({"id": "test"}, transform, (4, 3), module.BANDS, Resampling.bilinear, bad)


def test_landsat_alternative_keeps_iou_floor_and_updates_view_metadata(monkeypatch, tmp_path):
    import badapple_coastline.landsat as module

    def view(lon, iou):
        return dict(longitude=lon, latitude=60, width_km=12, rotation_clockwise=0, native_iou=iou)

    original = dict(
        view(1, 0.9),
        frame=8,
        baseline_native_iou=0.75,
        native_mask_path="primary.png",
        native_mask_sha256="primary",
        native_pixels_verified=True,
    )
    seen = []

    def fetch(record, *args, **kwargs):
        seen.append(record["longitude"])
        return {"path": "real.tif"}

    monkeypatch.setattr(module, "fetch_window", fetch)
    monkeypatch.setattr(module, "nearest_place", lambda *args: {"name": "Alternative location"})
    actual = module.worker(original, tmp_path, 24, "2020/2021", [view(2, 0.7), view(3, 0.8)])
    assert seen == [3] and actual["longitude"] == 3
    assert actual["landsat_original_geometry"]["longitude"] == 1
    assert actual["native_iou_gain"] == pytest.approx(0.05) and actual["landsat_alternative"]
    assert "native_mask_path" not in actual and "native_pixels_verified" not in actual


def test_shared_photo_preserves_each_frames_own_scores_and_timestamp():
    from badapple_coastline.landsat import apply_image_result

    record = dict(frame=2, timestamp=0.2, native_iou=0.9, baseline_native_iou=0.85)
    result = dict(
        frame=1,
        timestamp=0.1,
        native_iou=0.8,
        baseline_native_iou=0.75,
        satellite={"path": "shared-photo.png"},
        landsat_replacement=True,
    )
    actual = apply_image_result(record, result)
    assert all(actual[k] == record[k] for k in record)
    assert actual["satellite"] == result["satellite"]
    with pytest.raises(ValueError, match="different frame"):
        apply_image_result(record, dict(result, landsat_alternative=True))


def test_landsat_fills_only_real_missing_sentinel_pixels(monkeypatch, tmp_path):
    import badapple_coastline.landsat as module
    from badapple_coastline.data import digest

    item = dict(
        id="landsat-real",
        collection=module.COLLECTION,
        geometry={"type": "Polygon", "coordinates": [[[0, 0], [20, 0], [20, 80], [0, 80], [0, 0]]]},
        properties={"datetime": "2020-07-01T00:00:00Z", "eo:cloud_cover": 0, "platform": "landsat-8"},
        assets={k: {"href": "real-" + k} for k in ("red", "green", "blue", "qa_pixel")},
    )
    monkeypatch.setattr(module, "catalog", lambda *args: [item])

    def read(item, transform, size, assets, resampling, cache):
        if assets == ("qa_pixel",):
            return np.full((1, 3, 4), 21824, np.uint16)
        return np.full((3, 3, 4), 16000, np.uint16)

    monkeypatch.setattr(module, "read_bands", read)
    photo = tmp_path / "sentinel.png"
    pixels = np.full((3, 4, 3), 80, np.uint8)
    pixels[0, 0] = 0
    Image.fromarray(pixels).save(photo)
    background = dict(
        path=str(photo), image_sha256=digest(photo), scenes=[dict(id="sentinel-real", collection="sentinel-2-l2a")]
    )
    for name in ["validity", "cloud", "snow"]:
        values = np.zeros((3, 4), np.uint8)
        if name == "validity":
            values[:] = 255
            values[0, 0] = 0
        path = tmp_path / f"{name}.png"
        Image.fromarray(values).save(path)
        background[name + "_mask_path"] = str(path)
        background[name + "_mask_sha256"] = digest(path)
    record = dict(longitude=10, latitude=60, width_km=12, rotation_clockwise=0)
    result = fetch_window(record, tmp_path / "mixed", size=(4, 3), background=background)
    actual = np.asarray(Image.open(result["path"]))
    np.testing.assert_array_equal(actual[1:], pixels[1:])
    np.testing.assert_array_equal(actual[0, 1:], pixels[0, 1:])
    assert (actual[0, 0] == natural_color(np.array([16000], np.uint16))[0]).all()
    assert result["coverage"] == 1 and result["mixed_sensors"]
    assert Path(result["path"]).parent == tmp_path / "mixed"
    assert all(
        digest(Path(background[n + "_mask_path"])) == background[n + "_mask_sha256"]
        for n in ["validity", "cloud", "snow"]
    )
    assert result["sentinel_pixel_fraction"] == pytest.approx(11 / 12)
    assert [s["id"] for s in result["scenes"]] == ["sentinel-real", "landsat-real"]
