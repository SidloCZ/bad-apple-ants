import json
import numpy as np
import pytest
from PIL import Image
from badapple_coastline.footprint_map import FootprintHistory, footprint, polygons


@pytest.fixture
def land(tmp_path):
    path = tmp_path / "land.geojson"
    path.write_text(
        json.dumps(
            dict(
                type="FeatureCollection",
                features=[
                    dict(
                        type="Feature",
                        properties={},
                        geometry=dict(
                            type="Polygon", coordinates=[[[-150, 60], [-80, 60], [-80, 20], [-150, 20], [-150, 60]]]
                        ),
                    )
                ],
            )
        )
    )
    return path


def record(lon, frame=0):
    p = [[lon - 0.5, 0.5], [lon + 0.5, 0.5], [lon + 0.5, -0.5], [lon - 0.5, -0.5]]
    return dict(
        frame=frame,
        longitude=lon,
        latitude=0.0,
        width_km=111.0,
        rotation_clockwise=0.0,
        satellite=dict(footprint_lonlat=p, image_sha256=f"photo-{frame}"),
    )


def test_past_boxes_persist_and_future_locations_are_not_painted(land):
    state = FootprintHistory(land_path=land)
    first = record(-120, 0)
    second = record(120, 1)
    third = record(0, 2)
    a = np.asarray(state.panel(first, 0))
    base = np.asarray(state.base.convert("RGB").resize(state.size, Image.Resampling.LANCZOS))
    np.testing.assert_array_equal(a[58:73, 208:225], base[58:73, 208:225])
    b = np.asarray(state.panel(second, 1))
    c = np.asarray(state.panel(third, 2))
    np.testing.assert_array_equal(b[57:74, 34:52], c[57:74, 34:52])
    assert np.any(b[57:74, 34:52] != base[57:74, 34:52])
    assert len(state.views) == 3 and len(state.seen) == 3
    with pytest.raises(ValueError, match="video order"):
        state.panel(first, 4)


def test_map_uses_delivered_photo_corners_and_reflection_preserves_geographic_area(land):
    actual = record(100)
    actual["footprint_lonlat"] = record(-120)["satellite"]["footprint_lonlat"]
    np.testing.assert_array_equal(footprint(actual), actual["satellite"]["footprint_lonlat"])
    reflected = record(100)
    reflected["mirror_horizontal"] = True
    reflected["satellite"]["footprint_lonlat"] = reflected["satellite"]["footprint_lonlat"][::-1]
    normal = FootprintHistory(land_path=land).panel(actual, 0)
    mirrored = FootprintHistory(land_path=land).panel(reflected, 0)
    np.testing.assert_array_equal(np.asarray(normal), np.asarray(mirrored))


def test_dateline_footprint_is_split_at_both_edges_instead_of_crossing_map_center():
    p = np.array([[179, 10], [-179, 10], [-179, 9], [179, 9]], float)
    rings = polygons(p, (360, 180))
    assert len(rings) == 2
    assert all(np.ptp(r[:, 0]) < 3 for r in rings)
    assert all(r[:, 0].max() <= 2 or r[:, 0].min() >= 358 for r in rings)


def test_locator_box_enlargement_keeps_center_and_exports_true_coordinates(land, tmp_path):
    r = record(12)
    r["satellite"]["footprint_lonlat"] = [[12, 10], [12.001, 10], [12.001, 9.999], [12, 9.999]]
    p = footprint(r)
    normal = polygons(p, (260, 130))[0]
    shown = polygons(p, (260, 130), 6)[0]
    np.testing.assert_allclose(shown.mean(axis=0), normal.mean(axis=0), rtol=0, atol=1e-12)
    assert np.ptp(shown, axis=0).max() == pytest.approx(6.0)
    state = FootprintHistory(land_path=land)
    state.panel(r, 0)
    state.panel(dict(r, frame=1), 1)
    path = tmp_path / "footprints.json"
    metadata = state.save(path)
    saved = json.loads(path.read_text())
    np.testing.assert_array_equal(saved["views"][0]["corners_lonlat"], p)
    assert metadata["persistent_history"] and saved["frames"] == 2 and len(state.seen) == 1


def test_map_layout_preserves_photograph_reference_panels_and_fixed_footer(land):
    from badapple_coastline.render import compose

    r = record(12)
    r.update(place=dict(name="Test", country="Test country", distance_km=0), native_miou=0.8, mse=0.1)
    r["satellite"].update(scenes=[{"datetime": "2021-06-01T00:00:00Z"}], cloud_fraction=0.0)
    image = Image.new("RGB", (1444, 1080), (60, 80, 100))
    target = np.zeros((1080, 1444), np.float32)
    old = compose(image, target, 1 - target, r, 0, 2, True)
    panel = FootprintHistory(land_path=land).panel(r, 0)
    new = compose(image, target, 1 - target, r, 0, 2, True, world_map_panel=panel)
    assert new.size == (1280, 960)
    for box in [(20, 70, 980, 790), (1000, 92, 1260, 287), (1000, 352, 1260, 547), (0, 800, 1280, 960)]:
        np.testing.assert_array_equal(np.asarray(new.crop(box)), np.asarray(old.crop(box)))
