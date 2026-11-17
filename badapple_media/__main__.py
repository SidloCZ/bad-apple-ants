"""Fetch the shared inputs: the source animation and the two UI fonts.

    python -m badapple_media fetch [--assets assets] [--no-video]

Every file is pinned to one upstream commit and checked by SHA-256. The animation
is the 1444x1080, 30000/1001 fps, 6,955-frame encode kept in a public GitHub
repository; any other encode changes the frame count or the pixels, and the
pipelines check the source hash they were run with. The video and its audio are
used locally only; nothing here redistributes them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

VIDEO = (
    "https://raw.githubusercontent.com/bad-apple-lab/Bad-Apple/80e39ed58d754e4de420b7a1c474e53e20a71ba3/badapple.mp4",
    "badapple.mp4",
    "d8339642fd3303a0c4971e1564515e82f69ef61b700ce17b681debb19404c23a",
)
NOTO = "https://raw.githubusercontent.com/notofonts/notofonts.github.io/412a7edb2c8c535d1066157e5b17e661fd5591fe/fonts/"
FONTS = [
    (NOTO + "NotoSans/full/ttf/NotoSans-Regular.ttf", "fonts/noto/NotoSans/NotoSans-Regular.ttf",
     "f5f552c8c5edb61fe6efb824baf4d4de47b1a8689ab4925ff43f7bd6a4ebece5"),
    (NOTO + "NotoSansMono/unhinted/ttf/NotoSansMono-Regular.ttf", "fonts/noto/NotoSansMono/NotoSansMono-Regular.ttf",
     "87f8ce0522a6c99b743ee5fc75b4073cfdd575639119672828b7b9944b65b4f4"),
    (NOTO + "LICENSE", "licenses/Noto-OFL.txt", "f2095b08bed08b23a6fe26112fcd679a2bee3f002eef077eb05d215ed1051bd8"),
]  # fmt: skip


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, path: Path, expected: str) -> dict:
    if not (path.is_file() and sha256(path) == expected):
        path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(url, headers={"User-Agent": "bad-apple-earth-moon/1.0"})
        with urllib.request.urlopen(request, timeout=300) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError(f"Checksum mismatch: {url}")
        partial = path.with_suffix(path.suffix + ".part")
        partial.write_bytes(data)
        partial.replace(path)
        print(f"download: {path} ({len(data) / 1e6:.1f} MB), verified", flush=True)
    return {"path": str(path), "url": url, "sha256": expected}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m badapple_media")
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("fetch", help="Download the source animation and UI fonts (pinned, SHA-256 checked)")
    command.add_argument("--assets", type=Path, default=Path("assets"))
    command.add_argument("--no-video", action="store_true", help="Fonts only (bring your own copy of the video)")
    args = parser.parse_args(argv)
    files = [] if args.no_video else [VIDEO]
    files += FONTS
    result = [fetch(url, args.assets / relative, expected) for url, relative, expected in files]
    record = args.assets / "source.json"
    if record.is_file():  # keep entries of files not fetched this time (e.g. the video with --no-video)
        fetched = {entry["path"] for entry in result}
        result = [e for e in json.loads(record.read_text(encoding="utf-8")) if e["path"] not in fetched] + result
    record.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
