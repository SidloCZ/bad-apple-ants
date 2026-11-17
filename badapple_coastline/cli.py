from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
import urllib.error

from badapple_media import parse_size
from .atlas import Atlas, prepare_atlas
from .data import fetch_mask, fetch_natural_earth
from .matching import build_dictionary
from .pipeline import imagery, match, render_existing


def main(argv=None):
    parser = argparse.ArgumentParser(description="Bad Apple silhouettes matched to real Sentinel-2 land/water views")
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser(
        "fetch",
        help="Download and checksum the CC BY 4.0 land/water mask (7.3 GB globally) and the Natural Earth layers",
    )
    fetch.add_argument("--data", type=Path, default=Path("assets/coastline"))
    fetch.add_argument("--tiles", nargs="+", help="Omit to fetch the whole globe")
    fetch.add_argument("--workers", type=int, default=3)
    fetch.add_argument("--natural-earth-only", action="store_true", help="Only the Natural Earth layers (a few MB)")
    atlas = sub.add_parser("atlas", help="Downsample original tiles into a geographic search atlas")
    atlas.add_argument("--data", type=Path, default=Path("assets/coastline"))
    atlas.add_argument("--atlas", type=Path, default=Path("cache/coastline-atlas"))
    atlas.add_argument("--resolution", type=float, default=0.01)
    atlas.add_argument(
        "--allow-partial", action="store_true", help="Explicit regional pilot; missing areas stay unknown"
    )
    native_cache = sub.add_parser("native-cache", help="Build a lossless Rust/Rayon cache of all original 10 m pixels")
    native_cache.add_argument("--atlas", type=Path, default=Path("cache/coastline-atlas-fine"))
    native_cache.add_argument("--output", type=Path, default=Path("cache/coastline-native-rle"))
    build = sub.add_parser("build", help="Sample real coastal windows at multiple rotations/scales")
    build.add_argument("--atlas", type=Path, default=Path("cache/coastline-atlas"))
    build.add_argument("--dictionary", type=Path, default=Path("cache/coastline-dictionary"))
    build.add_argument("--centers", type=int, default=8000)
    build.add_argument("--widths", nargs="+", type=float, default=[24.0, 48.0, 96.0])
    build.add_argument("--angle-step", type=int, default=30)
    build.add_argument("--bounds", type=float, nargs=4, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    for name in ["match", "run"]:
        p = sub.add_parser(
            name, help="Match silhouettes" if name == "match" else "Match, retrieve actual imagery, render"
        )
        p.add_argument("--input", type=Path, default=Path("assets/badapple.mp4"))
        p.add_argument("--dictionary", type=Path, default=Path("cache/coastline-dictionary"))
        p.add_argument("--output", type=Path, default=Path("output/coastline"))
        p.add_argument("--fps")
        p.add_argument("--max-frames", type=int)
        p.add_argument("--start-seconds", type=float, default=0.0)
        p.add_argument("--duration", type=float)
        p.add_argument("--top-k", type=int, default=3)
        p.add_argument("--metric", choices=["iou", "mse"], default="iou")
        p.add_argument("--backend", choices=["rust", "python"], default="rust")
        p.add_argument("--per-scale", type=int, default=4, help="Rust: candidate count per geographic scale")
        if name == "run":
            p.add_argument("--image-size", type=parse_size, default=(960, 720))
            p.add_argument("--encoder", choices=["libx264", "h264_nvenc"], default="libx264")
    images = sub.add_parser("imagery", help="Resume cloud-screened Sentinel-2 retrieval for existing matches")
    images.add_argument("--output", type=Path, default=Path("output/coastline"))
    images.add_argument("--cache", type=Path, default=Path("cache/coastline-satellite"))
    images.add_argument("--size", type=parse_size, default=(960, 720))
    images.add_argument(
        "--interval", default="auto", help="Default: local summer in 2020 (southern summer through 2021)"
    )
    images.add_argument("--max-scenes", type=int, default=12)
    images.add_argument("--workers", type=int, default=4)
    rendering = sub.add_parser("render", help="Encode existing complete imagery, coordinates and attribution")
    rendering.add_argument("--output", type=Path, default=Path("output/coastline"))
    rendering.add_argument("--encoder", choices=["libx264", "h264_nvenc"], default="libx264")
    improvement = sub.add_parser(
        "improve", help="Wider scale search, keeping the baseline native-mask IoU for every frame"
    )
    improvement.add_argument("--baseline", type=Path, default=Path("output/coastline"))
    improvement.add_argument("--dictionary", type=Path, default=Path("cache/coastline-dictionary-wide"))
    improvement.add_argument("--output", type=Path, default=Path("output/coastline-wide-rust"))
    improvement.add_argument("--per-scale", type=int, default=4)
    improvement.add_argument("--native-candidates", type=int, default=12)
    improvement.add_argument("--refine-finalists", type=int, default=4)
    improvement.add_argument("--workers", type=int, default=8)
    improvement.add_argument("--backend", choices=["rust", "python"], default="rust")
    args = parser.parse_args(argv)
    try:
        if args.command == "fetch":
            result = fetch_natural_earth(args.data)
            if not args.natural_earth_only:
                result.update(fetch_mask(args.data, args.tiles, args.workers))
        elif args.command == "atlas":
            result = prepare_atlas(args.data, args.atlas, args.resolution, args.allow_partial)
        elif args.command == "native-cache":
            from .native_rle import build_cache

            result = build_cache(args.atlas, args.output)
        elif args.command == "build":
            if args.angle_step < 1 or args.angle_step > 360:
                raise ValueError("Angle step must be between 1 and 360")
            result = build_dictionary(
                Atlas(args.atlas),
                args.dictionary,
                args.centers,
                args.widths,
                tuple(range(0, 360, args.angle_step)),
                bounds=args.bounds,
            )
        elif args.command in {"match", "run"}:
            result = match(
                args.input,
                args.dictionary,
                args.output,
                args.fps,
                args.max_frames,
                args.start_seconds,
                args.duration,
                args.top_k,
                args.metric,
                args.backend,
                args.per_scale,
            )
            if args.command == "run":
                imagery(args.output, size=args.image_size)
                result = render_existing(args.output, args.encoder)
        elif args.command == "imagery":
            result = imagery(args.output, args.cache, args.size, args.interval, args.max_scenes, args.workers)
        elif args.command == "improve":
            from .improvement import improve

            result = improve(
                args.baseline,
                args.dictionary,
                args.output,
                args.per_scale,
                args.native_candidates,
                args.refine_finalists,
                args.workers,
                args.backend,
            )
        else:
            result = render_existing(args.output, args.encoder)
        print(json.dumps(result, ensure_ascii=True, indent=2))
    except (ValueError, FileNotFoundError, RuntimeError, urllib.error.URLError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Error: {exc}\n")
