"""Tests for 4K rendering helpers, mini world map projection, and binomial taxon formatting."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from badapple_ants.render import (
    MAP_H,
    MAP_W,
    RESOLUTION_SCALES,
    _build_glow_dot_sprite,
    _build_world_map_base,
    _draw_map_dot,
    _format_taxon,
    _project_lat_lon,
    _split_taxon_runs,
)


def test_project_lat_lon_bounds_and_center() -> None:
    map_w, map_h = MAP_W * 3, MAP_H * 3
    cx, cy = _project_lat_lon(0.0, 0.0, map_w, map_h, margin=0.0)
    assert abs(cx - 0.5 * (map_w - 1)) < 1e-6
    assert abs(cy - 0.5 * (map_h - 1)) < 1e-6

    mx, my = _project_lat_lon(90.0, -180.0, map_w, map_h, margin=6.0)
    assert mx == 6.0 and my == 6.0

    sx, sy = _project_lat_lon(-95.0, 200.0, map_w, map_h, margin=6.0)
    assert sx == float(map_w - 1 - 6.0)
    assert sy == float(map_h - 1 - 6.0)


def test_world_map_base_and_glow_dot_4k() -> None:
    scale = RESOLUTION_SCALES["4k"]
    assert scale == 3
    map_w, map_h = MAP_W * scale, MAP_H * scale
    assets_dir = Path("assets")
    base = _build_world_map_base(assets_dir, map_w, map_h, scale)
    assert base.size == (336, 168)
    assert base.mode == "RGB"

    sprite = _build_glow_dot_sprite(scale)
    assert sprite.mode == "RGBA"
    assert sprite.width == sprite.height
    center = sprite.width // 2
    r, g, b, a = sprite.getpixel((center, center))
    assert (r, g, b, a) == (255, 252, 230, 255)

    canvas = Image.new("RGB", (400, 220), (18, 19, 22))
    canvas.paste(base, (20, 20))
    _draw_map_dot(canvas, 20, 20, map_w, map_h, scale, {"lat": 50.0755, "lon": 14.4378}, sprite)
    # Missing coords should be a safe no-op.
    _draw_map_dot(canvas, 20, 20, map_w, map_h, scale, {"lat": None, "lon": None}, sprite)


def test_split_taxon_runs_and_format_taxon() -> None:
    assert _split_taxon_runs({"species": "Camponotus pennsylvanicus", "genus": "Camponotus"}) == [
        ("Camponotus pennsylvanicus", True)
    ]
    assert _split_taxon_runs({"species": "", "genus": "Formica"}) == [
        ("Formica", True),
        (" sp.", False),
    ]
    assert _split_taxon_runs({"species": "", "genus": ""}) == [("Formicidae", False)]
    assert _format_taxon({"species": "Lasius niger", "genus": "Lasius"}) == "Lasius niger"
    assert _format_taxon({"species": "", "genus": "Lasius"}) == "Lasius sp."
    assert _format_taxon({"species": "", "genus": ""}) == "Formicidae"
