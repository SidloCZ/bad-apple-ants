from __future__ import annotations

import argparse
from pathlib import Path

WAC_URL = (
    "https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif"
)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Bad Apple!! but every frame is the closest moon surface")
    sub = parser.add_subparsers(dest="command", required=True)
    catalog = sub.add_parser("catalog", help="Harvest a Kaguya TC STAC catalog (mono, stereo or sp)")
    catalog.add_argument("--series", choices=["mono", "stereo", "sp"], default="mono")
    catalog.add_argument("--output", type=Path)
    tc = sub.add_parser("ingest-tc", help="Fetch one <=95 m/px overview per TC image")
    tc.add_argument("--catalog", type=Path, default=Path("cache/moon/tc_catalog.jsonl"))
    tc.add_argument("--output", type=Path, default=Path("cache/moon/tiles"))
    tc.add_argument("--workers", type=int, default=48)
    tc.add_argument("--limit", type=int)
    tc.add_argument("--part", type=int, default=0, help="ingest catalog rows part::parts")
    tc.add_argument("--parts", type=int, default=1)
    tc.add_argument("--max-base-m", type=float, default=95.0, help="coarsest overview pixel to accept (m)")
    tc.add_argument("--ids", type=Path, help="only these image ids (one per line, e.g. results/moon/rasters.txt)")
    glob = sub.add_parser("ingest-global", help="Decimate the WAC global mosaic into the 800 m search base")
    glob.add_argument("--output", type=Path, default=Path("cache/moon/tiles"))
    sub.add_parser("fetch-names", help="Download the IAU/USGS lunar place names used in the captions")
    sub.add_parser("fetch-wac", help="Download the LROC WAC 100 m global mosaic (6 GB, resumable)")
    qa = sub.add_parser("qa-images", help="List low-signal TC observations to exclude from the search")
    qa.add_argument("--tiles", type=Path, default=Path("cache/moon/tiles"))
    merge = sub.add_parser(
        "merge", help="Merge refined candidates of several runs over the same queries (best score32 wins)"
    )
    merge.add_argument("inputs", type=Path, nargs="+")
    merge.add_argument("--out", type=Path, default=Path("output/moon/matches.json"))
    for name, help_text in (
        ("crops", "Cut each matched surface at full resolution (1944x1458)"),
        ("encode", "Caption the crops and encode the final videos with the original audio"),
        ("preview", "Contact sheet of frames vs matched surface"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--matches", type=Path, default=Path("output/moon/matches.json"))
        p.add_argument(
            "--index", type=Path, nargs="*", help="TC index files (default: every cache/moon/tiles*/tc_index.jsonl)"
        )
        p.add_argument("--crops", type=Path, default=Path("cache/moon/crops"))
        p.add_argument("--video", type=Path, default=Path("assets/badapple.mp4"))
        p.add_argument("--output", type=Path, default=Path("output/moon"))
        p.add_argument("--workers", type=int, default=24)
        p.add_argument("--encoder", choices=["av1_nvenc", "libx264", "h264_nvenc"], default="av1_nvenc")
        p.add_argument("--limit", type=int)
        p.add_argument("--frames", type=int, nargs="*")
        p.add_argument("--size", default="2560x1920", help="Video size WxH (aspect 4:3 or wider)")
        p.add_argument(
            "--fallback", type=Path, help="crops: earlier QA-checked matches.json used when every alternative fails"
        )
    fix = sub.add_parser("refooter", help="Rewrite the footer's Sun elevation in a finished video (no crops needed)")
    fix.add_argument("--video", type=Path, default=Path("output/moon/badapple-moon.mp4"))
    fix.add_argument("--frames", type=Path, default=Path("output/moon/frames.json"))
    fix.add_argument("--out", type=Path, default=Path("output/moon/badapple-moon.sunfix.mp4"))
    fix.add_argument("--bitrate", default="60M")
    args = parser.parse_args(argv)
    if args.command == "refooter":
        import json
        from .render import refooter
        from .sun import load_subsolar

        log = json.loads(args.frames.read_text(encoding="utf-8"))
        subsolar = load_subsolar({e["raster"] for e in log if e["raster"].startswith("TC")})
        refooter(args.video, args.frames, args.out, subsolar, args.bitrate)
        return
    if args.command == "fetch-wac":
        from .download import download

        download(WAC_URL, Path("assets/moon/raw/wac_100m.tif"))
        return
    if args.command == "fetch-names":
        from .names import fetch_names

        print(fetch_names())
        return
    if args.command == "catalog":
        from .catalog import TC_EXTRA, TC_MONO, harvest

        if args.series == "mono":
            harvest(TC_MONO, args.output or Path("cache/moon/tc_catalog.jsonl"))
        else:
            harvest(
                TC_EXTRA[args.series], args.output or Path(f"cache/moon/tc_catalog_{args.series}.jsonl"), tile_deg=10
            )
    elif args.command == "ingest-tc":
        from .ingest import ingest_tc

        ids = set(args.ids.read_text(encoding="utf-8").split()) if args.ids else None
        ingest_tc(args.catalog, args.output, args.workers, args.limit, args.part, args.parts, args.max_base_m, ids)
    elif args.command == "ingest-global":
        from .ingest import ingest_global

        ingest_global(args.output)
    elif args.command == "qa-images":
        from .ingest import qa_images

        qa_images(args.tiles)
    elif args.command == "merge":
        import json
        from .render import merge_matches

        runs = [json.loads(p.read_text(encoding="utf-8")) for p in args.inputs]
        args.out.write_text(json.dumps(merge_matches(runs)), encoding="utf-8")
    elif args.command in ("crops", "encode", "preview"):
        import json
        from . import render

        matches = json.loads(args.matches.read_text(encoding="utf-8"))
        records = render.load_tc_records(args.index or None)
        if args.command == "preview":
            render.preview(matches, records, args.video, args.output / "preview.jpg", args.frames or [])
        elif args.command == "crops":
            fallback = json.loads(args.fallback.read_text(encoding="utf-8")) if args.fallback else None
            render.make_crops(matches, records, args.crops, args.workers, args.matches, fallback)
        else:
            width, height = (int(v) for v in args.size.lower().split("x"))
            render.encode(
                matches, records, args.crops, args.video, args.output, args.encoder, args.limit, (width, height)
            )


if __name__ == "__main__":
    main()
