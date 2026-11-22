"""CLI entry point for the Bad Apple!! Ants prototype."""

from __future__ import annotations

import argparse
from pathlib import Path

from .ant_gate import MIN_ANT_SCORE
from .catalog import build_catalog, build_catalog_dwca
from .download import download_thumbs, download_winners
from .media import fetch_media
from .render import MAIN_MODES, render_video
from .search import (
    DEFAULT_DEDUP_L1,
    MAX_CENTER_OFFSET,
    MIN_FIT_RATIO,
    MIN_SHARPNESS_RATIO,
    run_search,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m badapple_ants")
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch-media", help="Download pinned Bad Apple!! video and UI fonts")
    p_fetch.add_argument("--assets", type=Path, default=Path("assets"))

    p_cat = sub.add_parser("catalog", help="Fetch iNaturalist + AntWeb ant image catalog via GBIF")
    p_cat.add_argument("--output", type=Path, default=Path("cache/catalog.jsonl"))
    p_cat.add_argument("--inat", type=int, default=6500)
    p_cat.add_argument("--antweb", type=int, default=3500)

    p_dwca = sub.add_parser(
        "catalog-dwca",
        help="Build a species-balanced catalog from a local GBIF Darwin Core Archive zip",
    )
    p_dwca.add_argument("--zip", type=Path, default=None, dest="zip_path")
    p_dwca.add_argument("--output", type=Path, default=Path("cache/catalog.jsonl"))
    p_dwca.add_argument("--inat", type=int, default=150000)
    p_dwca.add_argument("--antweb", type=int, default=85000)
    p_dwca.add_argument("--mcz", type=int, default=10000)
    p_dwca.add_argument("--nmnh", type=int, default=5000)
    p_dwca.add_argument("--max-per-occ", type=int, default=2, help="Max photos per occurrence (0 = unlimited)")
    p_dwca.add_argument("--commercial-only", action="store_true")
    p_dwca.add_argument("--no-fill", action="store_true")

    p_dl = sub.add_parser("download-thumbs", help="Download low-res thumbnails and pack search cache")
    p_dl.add_argument("--catalog", type=Path, default=Path("cache/catalog.jsonl"))
    p_dl.add_argument("--cache", type=Path, default=Path("cache"))
    p_dl.add_argument("--canvas-size", type=int, default=192)
    p_dl.add_argument("--concurrency", type=int, default=32)

    p_search = sub.add_parser("search", help="Run GPU NCC search and 64x48 refinement")
    p_search.add_argument("--video", type=Path, default=Path("assets/badapple.mp4"))
    p_search.add_argument("--cache", type=Path, default=Path("cache"))
    p_search.add_argument("--output", type=Path, default=Path("output/proto"))
    p_search.add_argument("--start", type=float, default=25.0)
    p_search.add_argument("--duration", type=float, default=25.0)
    p_search.add_argument("--cooldown", type=int, default=15)
    p_search.add_argument("--max-reuse", type=int, default=6)
    p_search.add_argument("--min-scale", type=float, default=MIN_FIT_RATIO)
    p_search.add_argument("--min-sharpness", type=float, default=MIN_SHARPNESS_RATIO)
    p_search.add_argument("--max-center-offset", type=float, default=MAX_CENTER_OFFSET)
    p_search.add_argument("--min-ant-score", type=float, default=MIN_ANT_SCORE)
    p_search.add_argument("--dedup-l1", type=float, default=DEFAULT_DEDUP_L1)
    p_search.add_argument("--max-images", type=int, default=0)

    p_win = sub.add_parser("fetch-winners", help="Download full-resolution photos for winning matches")
    p_win.add_argument("--matches", type=Path, default=Path("output/proto/matches.json"))
    p_win.add_argument("--cache", type=Path, default=Path("cache"))
    p_win.add_argument("--concurrency", type=int, default=12)

    p_ren = sub.add_parser("render", help="Render 1280x720 video, credits, and diagnostic stills")
    p_ren.add_argument("--matches", type=Path, default=Path("output/proto/matches.json"))
    p_ren.add_argument("--video", type=Path, default=Path("assets/badapple.mp4"))
    p_ren.add_argument("--assets", type=Path, default=Path("assets"))
    p_ren.add_argument("--cache", type=Path, default=Path("cache"))
    p_ren.add_argument("--output", type=Path, default=Path("output/proto"))
    p_ren.add_argument("--main-mode", choices=MAIN_MODES, default="grayscale-stretch")

    p_all = sub.add_parser("run-all", help="Run every prototype stage end-to-end")
    p_all.add_argument("--assets", type=Path, default=Path("assets"))
    p_all.add_argument("--cache", type=Path, default=Path("cache"))
    p_all.add_argument("--output", type=Path, default=Path("output/proto"))
    p_all.add_argument("--inat", type=int, default=6500)
    p_all.add_argument("--antweb", type=int, default=3500)
    p_all.add_argument("--start", type=float, default=25.0)
    p_all.add_argument("--duration", type=float, default=25.0)
    p_all.add_argument("--cooldown", type=int, default=15)
    p_all.add_argument("--max-reuse", type=int, default=6)
    p_all.add_argument("--min-scale", type=float, default=MIN_FIT_RATIO)
    p_all.add_argument("--min-sharpness", type=float, default=MIN_SHARPNESS_RATIO)
    p_all.add_argument("--max-center-offset", type=float, default=MAX_CENTER_OFFSET)
    p_all.add_argument("--min-ant-score", type=float, default=MIN_ANT_SCORE)
    p_all.add_argument("--dedup-l1", type=float, default=DEFAULT_DEDUP_L1)
    p_all.add_argument("--main-mode", choices=MAIN_MODES, default="grayscale-stretch")

    args = parser.parse_args(argv)

    if args.command == "fetch-media":
        fetch_media(args.assets)
    elif args.command == "catalog":
        build_catalog(args.output, inat_target=args.inat, antweb_target=args.antweb)
    elif args.command == "catalog-dwca":
        build_catalog_dwca(
            args.output,
            zip_path=args.zip_path,
            inat_target=args.inat,
            antweb_target=args.antweb,
            mcz_target=args.mcz,
            nmnh_target=args.nmnh,
            commercial_only=args.commercial_only,
            fill_remainder=not args.no_fill,
            max_per_occurrence=args.max_per_occ,
        )
    elif args.command == "download-thumbs":
        download_thumbs(
            args.catalog,
            args.cache,
            canvas_size=args.canvas_size,
            concurrency=args.concurrency,
        )
    elif args.command == "search":
        run_search(
            args.video,
            args.cache,
            args.output,
            start_s=args.start,
            duration_s=args.duration,
            cooldown=args.cooldown,
            max_reuse=args.max_reuse,
            min_scale=args.min_scale,
            min_sharpness=args.min_sharpness,
            max_center_offset=args.max_center_offset,
            min_ant_score=args.min_ant_score,
            dedup_l1=args.dedup_l1,
            max_images=args.max_images,
        )
    elif args.command == "fetch-winners":
        download_winners(args.matches, args.cache, concurrency=args.concurrency)
    elif args.command == "render":
        render_video(
            args.matches,
            args.video,
            args.assets,
            args.cache,
            args.output,
            main_mode=args.main_mode,
        )
    elif args.command == "run-all":
        fetch_media(args.assets)
        cat_path = args.cache / "catalog.jsonl"
        if not cat_path.is_file():
            build_catalog(cat_path, inat_target=args.inat, antweb_target=args.antweb)
        download_thumbs(cat_path, args.cache)
        run_search(
            args.assets / "badapple.mp4",
            args.cache,
            args.output,
            start_s=args.start,
            duration_s=args.duration,
            cooldown=args.cooldown,
            max_reuse=args.max_reuse,
            min_scale=args.min_scale,
            min_sharpness=args.min_sharpness,
            max_center_offset=args.max_center_offset,
            min_ant_score=args.min_ant_score,
            dedup_l1=args.dedup_l1,
        )
        download_winners(args.output / "matches.json", args.cache)
        render_video(
            args.output / "matches.json",
            args.assets / "badapple.mp4",
            args.assets,
            args.cache,
            args.output,
            main_mode=args.main_mode,
        )


if __name__ == "__main__":
    main()
