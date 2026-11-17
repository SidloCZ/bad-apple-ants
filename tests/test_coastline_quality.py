import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from badapple_coastline.quality import POLICY, black_placeholder, file_quality, image_quality


def test_visible_quality_rejects_whiteout_but_preserves_dark_real_water(tmp_path):
    assert not image_quality(Image.new("RGB", (40, 30), "white"))["acceptable"]
    assert not image_quality(Image.new("RGB", (40, 30), (190, 190, 190)))["acceptable"]
    assert image_quality(Image.new("RGB", (40, 30), (5, 25, 45)))["acceptable"]
    placeholder = black_placeholder(tmp_path, (40, 30), "2020/2021", "No usable view")
    assert placeholder["coverage"] == 0 and placeholder["status"] == "unavailable"
    assert not np.asarray(Image.open(placeholder["path"])).any()
    assert file_quality(placeholder["path"])["acceptable"]


def test_white_cached_scene_is_replaced_and_invalid_scl_cannot_supply_pixels(monkeypatch, tmp_path):
    import badapple_coastline.satellite as module
    from badapple_coastline.data import digest

    record = dict(longitude=10, latitude=60, width_km=12, rotation_clockwise=0)
    interval = module.resolve_interval(record)
    key = module.window_key(record, (4, 3), interval, 25.0)
    cached = tmp_path / f"{key}.png"
    Image.new("RGB", (4, 3), "white").save(cached)
    (tmp_path / f"{key}.json").write_text(json.dumps(dict(path=str(cached), coverage=1, image_sha256=digest(cached))))

    def item(name):
        return dict(
            id=name,
            collection="sentinel-2-l2a",
            properties={"datetime": "2020-07-01T00:00:00Z", "eo:cloud_cover": 0},
            assets={"visual": {"href": name + "-rgb"}, "SCL": {"href": name + "-scl"}},
        )

    monkeypatch.setattr(
        module, "request_json", lambda *args: {"features": [item("snow"), item("invalid"), item("summer")]}
    )
    calls = []

    def project(asset, *args):
        name = asset["href"].split("-")[0]
        calls.append(asset["href"])
        if asset["href"].endswith("scl"):
            return np.full((1, 3, 4), {"snow": 11, "invalid": 255, "summer": 4}[name], np.uint8)
        return np.full((3, 3, 4), 255 if name == "snow" else 100, np.uint8)

    monkeypatch.setattr(module, "project_asset", project)
    result = module.fetch_window(record, tmp_path, size=(4, 3), quality_policy=POLICY)
    assert [s["id"] for s in result["scenes"]] == ["summer"]
    assert "invalid-rgb" not in calls
    assert result["clear_coverage"] == 1 and result["quality_policy"] == POLICY
    assert (np.asarray(Image.open(result["path"])) == 100).all()


def test_failed_image_candidates_become_explicit_black_placeholder(monkeypatch, tmp_path):
    import badapple_coastline.pipeline as module

    output = tmp_path / "output"
    output.mkdir()
    settings = dict(
        match_run_id="test", global_search_coverage=True, image_quality_policy=POLICY, black_on_image_failure=True
    )
    (output / "run.json").write_text(json.dumps(settings))
    record = dict(frame=0, match_run_id="test", longitude=10, latitude=60, width_km=12, rotation_clockwise=0, mse=0.1)
    (output / "matches.jsonl").write_text(json.dumps(record) + "\n")
    monkeypatch.setattr(module, "fetch_places", lambda *args: None)
    monkeypatch.setattr(module, "nearest_place", lambda *args: dict(name="Test", country="Test", distance_km=1))

    def unavailable(*args, **kwargs):
        raise RuntimeError("No usable imagery")

    monkeypatch.setattr(module, "fetch_window", unavailable)
    module.imagery(output, tmp_path / "cache", size=(4, 3), workers=1)
    result = module.read_records(output / "frames.jsonl")[0]
    assert result["longitude"] == 10 and result["imagery_candidate_rank"] == -1
    assert result["satellite"]["placeholder"] and result["satellite"]["scenes"] == []
    assert not np.asarray(Image.open(result["satellite"]["path"])).any()
    assert json.loads((output / "imagery-status.json").read_text())["black_placeholder_frames"] == 1


def test_footer_attribution_pixels_stay_fixed_when_scene_dates_and_values_change(tmp_path):
    from badapple_coastline.render import compose

    photo = Image.new("RGB", (4, 3), (10, 30, 50))
    record = dict(
        place=dict(name="Long settlement name", country="Country", distance_km=9),
        latitude=1,
        longitude=2,
        width_km=6,
        rotation_clockwise=0,
        mse=0.1,
        satellite=dict(scenes=[{"datetime": "2020-07-01"}], cloud_fraction=0),
    )
    first = np.asarray(compose(photo, np.zeros((3, 4)), np.zeros((3, 4)), record, 0, 2, True))
    record.update(
        latitude=-85,
        longitude=-179,
        width_km=480,
        rotation_clockwise=359,
        satellite=dict(scenes=[], cloud_fraction=0.9, placeholder=True),
    )
    second = np.asarray(compose(photo, np.zeros((3, 4)), np.zeros((3, 4)), record, 1, 2, True))
    assert np.array_equal(first[910:], second[910:])
    assert np.array_equal(first[848:875, 20:60], second[848:875, 20:60])


def test_later_worker_photo_can_rescue_placeholder_without_reusing_wrong_native_mask(monkeypatch):
    import badapple_coastline.repair as module

    def geometry(lon, iou):
        return dict(longitude=lon, latitude=60, width_km=12, rotation_clockwise=0, native_iou=iou)

    missing = dict(
        geometry(1, 0.9),
        frame=0,
        baseline_native_iou=0.7,
        native_mask_path="view-one.png",
        native_mask_sha256="view-one",
        satellite={"placeholder": True},
    )
    valid = dict(geometry(2, 0.8), frame=1, satellite={"path": "view-two.png"})
    matches = {0: dict(geometry(1, 0.9), alternatives=[geometry(2, 0.8)])}
    monkeypatch.setattr(module, "nearest_place", lambda *args: {"name": "View two"})
    records = [missing, valid]
    assert module.reuse_repaired_photos(records, matches, [valid]) == 1
    rescued = records[0]
    assert rescued["longitude"] == 2 and rescued["native_iou_gain"] == pytest.approx(0.1)
    assert rescued["satellite"] == valid["satellite"] and "native_mask_path" not in rescued
    assert records[1] is valid
