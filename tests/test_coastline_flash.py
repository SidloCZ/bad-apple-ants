import io
from pathlib import Path
import json
import numpy as np
import pytest
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


def test_trim_audio_mode_reencodes_without_speed_filter():
    from badapple_coastline.flash_reduction import audio_command

    command = audio_command(Path("video.mp4"), Path("original.mp4"), Path("final.mp4"), 6955, "trim")
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
