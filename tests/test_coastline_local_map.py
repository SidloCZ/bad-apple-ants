import numpy as np
import pytest
from PIL import Image
from pyproj.enums import TransformDirection
from badapple_coastline.atlas import view_transform, corners
from badapple_coastline.local_map import geometry, map_xy, centre, LocalMaps, LocalMapCache, SIZE, HEADER


def record(lon=20, lat=10, width=2, angle=30, mirror=False):
    points = corners(view_transform(lon, lat, width, angle, (1444, 1080), mirror_horizontal=mirror), (1444, 1080))
    return dict(
        longitude=-80,
        latitude=-30,
        mirror_horizontal=mirror,
        satellite=dict(footprint_lonlat=points, image_sha256=f"{lon}-{lat}-{width}-{angle}"),
    )


class FlatSampler:
    def sample(self, longitude, latitude):
        return np.zeros(np.shape(longitude), np.uint8)


@pytest.mark.parametrize(
    "lon,lat,width,angle", [(179.99, 0, 20, 67), (-179.99, 45, 30, 125), (35, 83.2, 800, 90), (70, -75, 400, 33)]
)
def test_metric_local_geometry_handles_seam_poles_and_actual_photo_corners(lon, lat, width, angle):
    r = record(lon, lat, width, angle, True)
    p = np.asarray(r["satellite"]["footprint_lonlat"])
    transform, x, y, span = geometry(p)
    assert abs((x - lon + 180) % 360 - 180) < 1 and abs(y - lat) < 1
    xy = map_xy(transform, p, span)
    assert ((xy > 0) & (xy < np.array(SIZE))).all()
    map_x, map_y = transform.transform(x, y + 0.001)
    assert map_y > 0 and abs(map_x) < 1
    # Equal metric x/y scales; inverse projection returns true corner coordinates.
    east = (xy[:, 0] / SIZE[0] - 0.5) * span * 1000
    north = (SIZE[1] / 2 - xy[:, 1]) / SIZE[0] * span * 1000
    a, b = transform.transform(east, north, direction=TransformDirection.INVERSE)
    np.testing.assert_allclose((np.asarray(a) - p[:, 0] + 180) % 360 - 180, 0, atol=1e-7)
    np.testing.assert_allclose(b, p[:, 1], atol=1e-7)


def test_local_history_keeps_all_nearby_past_views_without_future_or_300_limit():
    records = [record(lon=20 + i * 0.00001) for i in range(307)]
    maps = LocalMaps(records, sampler=FlatSampler())
    first, info = maps.render(0)
    assert info["nearby_past_unique_views"] == 0
    last, info = maps.render(305)
    assert info["nearby_past_unique_views"] == 305
    assert info["photograph_sha256"] == records[305]["satellite"]["image_sha256"]
    assert first.shape == last.shape == (SIZE[1] + HEADER, SIZE[0], 3)


def test_sidebar_preserves_original_and_photo_and_moves_true_world_history(tmp_path):
    panel = np.full((1, SIZE[1] + HEADER, SIZE[0], 3), 22, np.uint8)
    path = tmp_path / "local.u8"
    panel.tofile(path)
    cache = LocalMapCache(path, dict(frames=1, visited_counts=[1]))
    rng = np.random.default_rng(4)
    frame = rng.integers(0, 256, (960, 1280, 3), np.uint8)
    before = frame.copy()
    actual = cache.compose(frame, 0, True)
    np.testing.assert_array_equal(actual[92:287, 1000:1260], before[92:287, 1000:1260])
    np.testing.assert_array_equal(actual[70:790, 20:980], before[70:790, 20:980])
    np.testing.assert_array_equal(actual[608:738, 1000:1260], before[608:738, 1000:1260])
    np.testing.assert_array_equal(actual[352:547, 1000:1260], before[352:547, 1000:1260])
    assert actual[352:547, 1000:1260].shape == actual[92:287, 1000:1260].shape
    np.testing.assert_array_equal(actual[780:942, 1000:1260], panel[0, HEADER:])
    expected_footer = np.asarray(Image.fromarray(before[800:960]).resize((960, 120), Image.Resampling.LANCZOS))
    np.testing.assert_array_equal(actual[810:930, 20:980], expected_footer)
    assert 980 < 1000 and 930 < 960  # Footer ends before the right column and canvas bottom.
    assert np.all(actual[800:960, :20] == [8, 15, 23])
    outside = np.ones((960, 1280), bool)
    outside[294:960, 1000:1260] = False
    outside[800:960] = False
    np.testing.assert_array_equal(actual[outside], before[outside])
