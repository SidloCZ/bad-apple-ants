import math

import numpy as np
import pytest

from badapple_moon.geometry import apply_quadratic, fit_quadratic, ortho_forward, ortho_inverse
from badapple_moon.render import crop_key, merge_matches
from badapple_moon.sun import camera_line, local_sun


@pytest.mark.parametrize("lon0,lat0", [(0.0, 0.0), (140.86, -20.34), (-115.8, 86.6), (179.5, -89.0)])
def test_orthographic_round_trip(lon0, lat0):
    rng = np.random.default_rng(0)
    x, y = rng.uniform(-30, 30, 200), rng.uniform(-30, 30, 200)
    lon, lat = ortho_inverse(x, y, lon0, lat0)
    x2, y2 = ortho_forward(lon, lat, lon0, lat0)
    np.testing.assert_allclose(x2, x, atol=1e-9)
    np.testing.assert_allclose(y2, y, atol=1e-9)


def test_orthographic_centre_and_axes():
    assert ortho_forward(10.0, 20.0, 10.0, 20.0) == pytest.approx((0.0, 0.0))
    x, y = ortho_forward(10.01, 20.0, 10.0, 20.0)
    assert x > 0 and abs(y) < 1e-3  # east is +x
    x, y = ortho_forward(10.0, 20.01, 10.0, 20.0)
    assert y > 0 and abs(x) < 1e-9  # north is +y


def test_quadratic_fit_is_exact_for_quadratic_maps():
    rng = np.random.default_rng(1)
    x, y = rng.uniform(-5, 5, 100), rng.uniform(-5, 5, 100)
    u = 3 + 2 * x - y + 0.1 * x * x + 0.05 * x * y
    v = -1 + 0.5 * y + 0.02 * y * y
    coefficients, residual = fit_quadratic(x, y, u, v)
    assert residual < 1e-9
    u2, v2 = apply_quadratic(coefficients, np.array([1.5]), np.array([-2.0]))
    assert u2[0] == pytest.approx(3 + 3 + 2 + 0.225 - 0.15)
    assert v2[0] == pytest.approx(-1 - 1 + 0.08)


def test_local_sun_elevation_and_time_of_day():
    elevation, _ = local_sun(0.0, 0.0, [0.0, 0.0])
    assert elevation == pytest.approx(90.0)
    elevation, morning = local_sun(0.0, -45.0, [0.0, 0.0])  # Sun 45 degrees to the east
    assert elevation == pytest.approx(45.0) and morning
    elevation, morning = local_sun(0.0, 45.0, [0.0, 0.0])
    assert elevation == pytest.approx(45.0) and not morning
    # Near the pole the Sun stays within about the sub-solar latitude of the horizon.
    elevation, _ = local_sun(-88.56, 10.0, [10.0, -1.5])
    assert abs(elevation) < 3.0


def test_camera_line_never_shows_negative_zero():
    assert camera_line(-0.04, True).endswith("SUN  0.0°")
    assert "EVENING" in camera_line(12.0, False)


def candidate(raster, score, a=0.0):
    return {"raster": raster, "source": "tc", "mirror": False, "score32": score, "a": a, "b": 0.0,
            "width_km": 3.0, "theta_deg": 0.0, "lon": 0.0, "lat": 0.0}  # fmt: skip


def test_merge_keeps_the_best_candidate_of_any_run():
    run1 = {"frames": [0, 1], "results": [
        {"unique": 0, "kind": "match", "best": candidate("A", 0.8), "alternatives": [candidate("A", 0.8)], "qa": {"rejected": []}},
        {"unique": 1, "kind": "dark", "best": candidate("D", 0.0)},
    ]}  # fmt: skip
    run2 = {"frames": [0, 1], "results": [
        {"unique": 0, "kind": "match", "best": candidate("B", 0.9), "alternatives": [candidate("B", 0.9), candidate("A", 0.8)]},
        {"unique": 1, "kind": "dark", "best": candidate("E", 0.0)},
    ]}  # fmt: skip
    merged = merge_matches([run1, run2])
    first = merged["results"][0]
    assert first["best"]["raster"] == "B" and "qa" not in first
    assert [c["raster"] for c in first["alternatives"]] == ["B", "A"]  # duplicates collapse by crop key
    assert merged["results"][1]["best"]["raster"] == "D"  # single-colour frames keep the first run


def test_crop_key_depends_on_pose_only():
    assert crop_key(candidate("A", 0.1)) == crop_key(candidate("A", 0.9))
    assert crop_key(candidate("A", 0.1)) != crop_key(candidate("A", 0.1, a=math.pi))
