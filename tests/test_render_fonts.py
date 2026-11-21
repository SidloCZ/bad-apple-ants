"""Tests for script fallback font rendering on non-Latin author names."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from badapple_ants.render import FontChain


def test_font_chain_fallback_renders_non_latin(tmp_path: Path) -> None:
    assets_dir = Path("assets")
    if not (assets_dir / "fonts" / "NotoSans-Regular.ttf").is_file():
        return

    chain = FontChain.load(assets_dir, "NotoSans-Regular.ttf", 14)
    primary = chain.fonts[0]
    assert chain.resolve("A") is primary

    # Verify fallback activates for CJK characters missing in NotoSans-Regular
    has_cjk_fallback = len(chain.fonts) > 1 and chain.resolve("张") is not primary
    img = Image.new("RGB", (288, 32), (18, 19, 22))
    draw = ImageDraw.Draw(img)

    text = "张三 / 김 / あ | CC-BY-NC-4.0"
    truncated = chain.truncate(draw, text, 288)
    assert truncated == text
    assert chain.textlength(draw, text) > 0

    chain.draw_text(draw, (4, 6), text, fill=(156, 163, 175))
    arr = np.asarray(img)
    assert arr.max() > 100

    if has_cjk_fallback:
        # Compare rendered CJK glyphs against forced-primary tofu boxes
        tofu_img = Image.new("RGB", (288, 32), (18, 19, 22))
        tofu_draw = ImageDraw.Draw(tofu_img)
        assert isinstance(primary, ImageFont.FreeTypeFont)
        tofu_draw.text((4, 6), text, fill=(156, 163, 175), font=primary)
        assert not np.array_equal(arr, np.asarray(tofu_img))


def test_font_chain_truncation_within_width() -> None:
    chain = FontChain.load(Path("assets"), "NotoSans-Regular.ttf", 12)
    img = Image.new("RGB", (200, 32), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    long_name = "蚂蚁摄影师_" * 15 + "| CC-BY-4.0"
    out = chain.truncate(draw, long_name, 120)
    assert out.endswith("...")
    assert chain.textlength(draw, out) <= 120.5
