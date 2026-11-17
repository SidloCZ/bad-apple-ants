import io
from pathlib import Path
import json
import numpy as np
import pytest
from PIL import Image
from badapple_coastline.flash_reduction import output_spec, fill_frame, source_frame_map
from badapple_coastline.footprint_map import FootprintHistory


def test_exact_30p_preserves_every_source_frame_once_without_additions():
    result = output_spec(6955)
    assert result["fps"] == "30/1" and result["frames"] == 6955
    assert result["progressive"] and result["drop_frame_timecode"] is False
    assert result["duration_seconds"] == pytest.approx(231.8333333333)
    assert result["audio_tempo"] is None and result["repeated_display_frames"] == result["dropped_frames"] == 0
    np.testing.assert_array_equal(source_frame_map(6955), np.arange(6955))
    assert result["content_box"] == [0, 0, 2560, 1920]
    assert (result["width"], result["height"]) == (2560, 1920)


def test_reusable_frame_reader_detects_truncation_and_clean_eof():
    buffer = np.empty(9, np.uint8)
    source = io.BytesIO(bytes(range(9)))
    assert fill_frame(source, buffer) and buffer.tolist() == list(range(9))
    assert fill_frame(source, buffer) is False
    with pytest.raises(ValueError, match="Truncated"):
        fill_frame(io.BytesIO(b"123"), buffer)


def test_unicode_audio_mode_matches_existing_renderer_without_speed_filter():
    from badapple_coastline.flash_reduction import audio_command

    command = audio_command(Path("video.mp4"), Path("original.mp4"), Path("final.mp4"), 6955, "unicode")
    assert "-af" not in command and not any("atempo" in value for value in command)
    assert command[command.index("-c:a") + 1] == "aac" and command[command.index("-b:a") + 1] == "192k"
    assert float(command[command.index("-t") + 1]) == pytest.approx(6955 / 30)
    assert command[command.index("-c:v") + 1] == "copy"


def test_history_only_accumulation_matches_rendering_every_preview(tmp_path):
    land = tmp_path / "land.geojson"
    land.write_text(json.dumps({"type": "FeatureCollection", "features": []}))

    def record(i):
        x = -100 + i * 60
        return dict(
            frame=i, satellite=dict(footprint_lonlat=[[x, 10], [x + 1, 10], [x + 1, 9], [x, 9]], image_sha256=str(i))
        )

    a = FootprintHistory(land_path=land)
    b = FootprintHistory(land_path=land)
    for i in range(3):
        a.panel(record(i), i)
        b.advance(record(i), i)
    np.testing.assert_array_equal(np.asarray(a.panel(record(3), 3)), np.asarray(b.panel(record(3), 3)))


@pytest.mark.skipif(
    not __import__("badapple_coastline.footprint_map", fromlist=["LAND"]).LAND.is_file(),
    reason="Natural Earth not downloaded (python -m badapple_coastline fetch --natural-earth-only)",
)
def test_fixed_credits_keep_one_real_photo_and_both_original_credit_cards(monkeypatch, tmp_path):
    import badapple_coastline.flash_reduction as module

    monkeypatch.setattr(module, "CREDIT_START", 3)
    monkeypatch.setattr(module, "CREDIT_PHOTO", 4)
    monkeypatch.setattr(module, "LOGO_START", 6)
    monkeypatch.setattr(module, "LOGO_SOURCE", 7)

    def capture(command, **kwargs):
        value = 60 if "00004" in command[-1] else 120
        Image.new("RGB", (1280, 960), (value, 80, 100)).save(command[-1])

    monkeypatch.setattr(module.subprocess, "run", capture)
    rows = [
        dict(
            frame=i,
            mirror_horizontal=bool(i % 2),
            satellite=dict(image_sha256=str(i), footprint_lonlat=[[i, 10], [i + 0.1, 10], [i + 0.1, 9], [i, 9]]),
        )
        for i in range(8)
    ]
    a, b, info = module.prepare_credits(Path("fixture.mp4"), rows, tmp_path)
    np.testing.assert_array_equal(a[70:790, 20:980], b[70:790, 20:980])
    assert np.all(a[100:280, 1000:1260, 0] == 60) and np.all(b[100:280, 1000:1260, 0] == 120)
    assert info["output_mirror_switches"] == 0 and info["geographical_view_fixed"]
