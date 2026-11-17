"""Repair only visibly unusable views; preserve all other satellite photographs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .pipeline import imagery, nearest_place, read_records, write_json
from .quality import POLICY, file_quality
from .satellite import resolve_interval, window_key


def reuse_repaired_photos(records, matches, donors):
    """A later worker may retrieve an exact view an earlier worker could not."""
    key = lambda r: window_key(r, (960, 720), resolve_interval(r), 25.0)
    available = {key(r): r["satellite"] for r in donors if not r["satellite"].get("placeholder")}
    reused = 0
    for index, record in enumerate(records):
        if not record["satellite"].get("placeholder"):
            continue
        source = matches[record["frame"]]
        candidates = [record] + sorted(
            [source] + source.get("alternatives", []), key=lambda r: r["native_iou"], reverse=True
        )
        for candidate in candidates:
            photo = available.get(key(candidate))
            if photo is None:
                continue
            success = dict(record, **{k: v for k, v in candidate.items() if k not in ("satellite", "alternatives")})
            if key(candidate) != key(record) and "native_mask_path" not in candidate:
                success.pop("native_mask_path", None)
                success.pop("native_mask_sha256", None)
            success.update(
                satellite=photo,
                cached_repair_reuse=True,
                imagery_candidate_rank=0 if key(candidate) == key(record) else 1,
                place=nearest_place(
                    Path("assets/coastline/places-10m.geojson"), success["longitude"], success["latitude"]
                ),
            )
            success["native_iou_gain"] = success["native_iou"] - success["baseline_native_iou"]
            records[index] = success
            reused += 1
            break
    return reused


def repair(source: Path, output: Path, cache: Path, workers=16, max_scenes=12):
    source, output, cache = source.resolve(), output.resolve(), cache.resolve()
    if source == output:
        raise ValueError("Repair output must preserve the original sequence in a separate directory")
    output.mkdir(parents=True, exist_ok=True)
    settings = json.loads((source / "run.json").read_text(encoding="utf-8"))
    records = read_records(source / "frames.jsonl")
    baseline = read_records(Path(settings["baseline"]) / "frames.jsonl")
    matches = {r["frame"]: r for r in read_records(source / "matches.jsonl")}
    photos = {r["satellite"]["image_sha256"]: r["satellite"]["path"] for r in records + baseline}
    print(f"Checking {len(photos)} distinct photographs", flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        quality = dict(zip(photos, pool.map(file_quality, photos.values())))
    bad = [r for r in records if not quality[r["satellite"]["image_sha256"]]["acceptable"]]
    print(f"Repairing {len(bad)}/{len(records)} frames", flush=True)
    key = lambda r: window_key(r, (960, 720), resolve_interval(r), 25.0)
    available = {key(r) for r in records + baseline if quality[r["satellite"]["image_sha256"]]["acceptable"]}
    requests = []
    for original in bad:
        candidates = [matches[original["frame"]]] + matches[original["frame"]].get("alternatives", [])
        candidates = sorted(candidates, key=lambda r: r["native_iou"], reverse=True)
        # First retry this exact view in summer. A known good alternative is
        # next, then two fresh views that still satisfy the native IoU floor.
        selected, seen = [], {key(original)}
        for cached_only, limit in [(True, 1), (False, 2)]:
            added = 0
            for candidate in candidates:
                identity = key(candidate)
                if identity in seen or (cached_only and identity not in available):
                    continue
                selected.append({k: v for k, v in candidate.items() if k != "alternatives"})
                seen.add(identity)
                added += 1
                if added >= limit:
                    break
        request = {k: v for k, v in original.items() if k not in ("satellite", "place", "alternatives")}
        request.update(
            alternatives=selected, repair_exclude_scene_ids=[s["id"] for s in original["satellite"]["scenes"]]
        )
        requests.append(request)
    settings.update(
        image_quality_policy=POLICY,
        black_on_image_failure=True,
        repaired_from=str(source),
        repair_max_scenes=max_scenes,
    )
    subset = output / "repair"
    subset.mkdir(exist_ok=True)
    write_json(subset / "run.json", settings)
    (subset / "matches.jsonl").write_text("".join(json.dumps(r) + "\n" for r in requests), encoding="utf-8")
    shared = {Path(r["satellite"]["path"]).parent for r in records + baseline}
    shared.add(Path("cache/coastline-satellite").resolve())
    os.environ["COASTLINE_SHARED_WINDOW_CACHE"] = os.pathsep.join(map(str, sorted(shared)))
    os.environ["COASTLINE_SCENE_READ_CACHE"] = os.pathsep.join(
        str(p / "scene-cache") for p in sorted(shared) if (p / "scene-cache").is_dir()
    )
    os.environ["COASTLINE_CATALOG_READ_CACHE"] = os.pathsep.join(
        str(p / "catalogs") for p in sorted(shared) if (p / "catalogs").is_dir()
    )
    if requests:
        imagery(subset, cache, (960, 720), "auto", max_scenes, workers)
        replacement = {r["frame"]: r for r in read_records(subset / "frames.jsonl")}
    else:
        replacement = {}
    merged = [replacement.get(r["frame"], r) for r in records]
    donors = [r for r in records + baseline if quality[r["satellite"]["image_sha256"]]["acceptable"]]
    reused = reuse_repaired_photos(merged, matches, donors + merged)
    for r in merged:
        satellite = r["satellite"]
        if satellite.get("placeholder"):
            continue
        visible = quality.get(satellite["image_sha256"]) or file_quality(satellite["path"])
        if not visible["acceptable"]:
            raise RuntimeError(f"Frame {r['frame']} still has whiteout")
    (output / "frames.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in merged), encoding="utf-8"
    )
    shutil.copyfile(source / "matches.jsonl", output / "matches.jsonl")
    write_json(output / "run.json", settings)
    report = dict(
        requested=len(records),
        complete=len(merged),
        failures=[],
        repair_requested=len(bad),
        unchanged_photos=sum(
            a["satellite"]["image_sha256"] == b["satellite"]["image_sha256"] for a, b in zip(records, merged)
        ),
        black_placeholder_frames=sum(bool(r["satellite"].get("placeholder")) for r in merged),
        cached_repair_reuse=reused,
        changed_views=sum(key(a) != key(b) for a, b in zip(records, merged)),
        quality_policy=POLICY,
    )
    write_json(output / "imagery-status.json", report)
    write_json(output / "repair-report.json", dict(report, repaired_frames=sorted(replacement)))
    credits = (source / "CREDITS.txt").read_text(encoding="utf-8")
    credits += (
        "\nWhiteout repair: prefer local summer; reject snowy/whiteout imagery. Only unusable photographs were replaced.\n"
        f"Unretrievable views: {report['black_placeholder_frames']} black placeholders, explicitly marked NO IMAGE. These are not satellite photographs.\n"
        "Per-frame source scenes and coordinates remain in frames.jsonl.\n"
    )
    (output / "CREDITS.txt").write_text(credits, encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("output/coastline-wide-rust"))
    parser.add_argument("--output", type=Path, default=Path("output/coastline-wide-rust-fixed"))
    parser.add_argument("--cache", type=Path, default=Path("cache/coastline-satellite-repair"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-scenes", type=int, default=12)
    args = parser.parse_args()
    repair(**vars(args))
