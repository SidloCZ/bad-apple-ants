"""Export the per-frame results of the published videos into results/ (small, text, no images).

    python scripts/export_results.py

Reads the outputs of a finished run:
  output/coastline-precision-v4-2560x1920-30p-native-av1/frames.jsonl  (satellite video)
  output/landwater-osm/frames.csv                                     (country / continent per frame)
  output/moon/matches.json, output/moon/frames.json                   (moon video)
and writes
  results/satellite/frames.csv   one row per frame: view, score, photo scenes and dates, place
  results/moon/frames.csv        one row per frame: image, pose, NCC, feature, Sun
  results/moon/matches.json      the chosen pose of every distinct frame; copy it to output/moon/matches.json
                                 and run `python -m badapple_moon crops` (docs/moon.md)
  results/moon/rasters.txt       the Kaguya TC images those poses use (for `ingest-tc --ids`)
"""

from __future__ import annotations

import csv
import json
from fractions import Fraction
from pathlib import Path

SATELLITE = Path("output/coastline-precision-v4-2560x1920-30p-native-av1/frames.jsonl")
LANDWATER = Path("output/landwater-osm/frames.csv")
MOON = Path("output/moon")
RESULTS = Path("results")


def write_csv(path: Path, header: list[str], rows: list[list]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    print(f"{path}: {len(rows)} rows, {path.stat().st_size / 1e6:.2f} MB")


def number(value, digits):
    return "" if value is None else f"{value:.{digits}f}"


def satellite():
    records = [json.loads(line) for line in SATELLITE.read_text(encoding="utf-8").splitlines() if line.strip()]
    with LANDWATER.open(encoding="utf-8-sig", newline="") as stream:
        places = {int(row["frame"]) - 1: row for row in csv.DictReader(stream)}  # that CSV counts from 1
    header = [
        "frame", "time_s", "longitude", "latitude", "width_km", "height_km", "rotation_clockwise_deg", "mirrored",
        "miou", "precision_objective",
        "corner1_lon", "corner1_lat", "corner2_lon", "corner2_lat", "corner3_lon", "corner3_lat", "corner4_lon", "corner4_lat",
        "photo_collection", "photo_scenes", "photo_dates", "photo_cloud_fraction",
        "country", "territory", "continent", "nearest_place", "nearest_place_country", "nearest_place_km", "note",
    ]  # fmt: skip
    rows = []
    for r in records:
        frame = r["frame"]
        photo = r["satellite"]
        scenes = photo["scenes"]
        scenes = json.loads(scenes) if isinstance(scenes, str) else scenes
        corners = [v for corner in r["footprint_lonlat"] for v in corner]
        place = places.get(frame, {})
        rows.append(
            [
                frame, f"{frame / 30:.4f}",  # the satellite video is 30p; the moon video keeps 30000/1001
                number(r["longitude"], 6), number(r["latitude"], 6), number(r["width_km"], 4), number(r["height_km"], 4),
                number(r["rotation_clockwise"], 3), "yes" if r.get("mirror_horizontal") else "no",
                number(r.get("native_miou"), 4), number(r.get("precision_objective"), 4),
                *[number(v, 6) for v in corners],
                ";".join(sorted({s.get("collection", "") for s in scenes})),
                ";".join(s["id"] for s in scenes),
                ";".join(sorted({s["datetime"][:10] for s in scenes})),
                number(photo.get("cloud_fraction"), 4),
                place.get("country", ""), place.get("territory", ""), place.get("continent", ""),
                r["place"]["name"], r["place"]["country"], f"{r['place']['distance_km']:.0f}",
                place.get("note", ""),
            ]
        )  # fmt: skip
    write_csv(RESULTS / "satellite/frames.csv", header, rows)


def moon():
    matches = json.loads((MOON / "matches.json").read_text(encoding="utf-8"))
    frames = json.loads((MOON / "frames.json").read_text(encoding="utf-8"))
    keep = ["raster", "source", "mirror", "score32", "score64", "level", "a", "b", "width_km", "theta_deg", "lon", "lat"]
    compact = {
        "input": matches["input"],
        "rate": matches["rate"],
        "frames": matches["frames"],
        "note": "Chosen pose per distinct frame after full-resolution QA; alternatives omitted.",
        "results": [
            {"unique": r["unique"], "kind": r["kind"], "best": {k: r["best"].get(k) for k in keep}} for r in matches["results"]
        ],
    }
    path = RESULTS / "moon/matches.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(compact, separators=(",", ":")), encoding="utf-8")
    print(f"{path}: {len(compact['results'])} poses, {path.stat().st_size / 1e6:.2f} MB")
    rasters = sorted({r["best"]["raster"] for r in compact["results"] if r["best"]["source"] == "tc"})
    (RESULTS / "moon/rasters.txt").write_text("\n".join(rasters) + "\n", encoding="utf-8")
    print(f"results/moon/rasters.txt: {len(rasters)} TC images")
    poses = {r["unique"]: r["best"] for r in compact["results"]}
    rate = Fraction(matches["rate"])
    header = [
        "frame", "time_s", "kind", "image", "source", "latitude", "longitude", "width_km", "rotation_deg", "mirrored",
        "ncc", "feature", "feature_detail", "camera",
    ]  # fmt: skip
    rows = []
    for e in frames:
        best = poses[e["unique"]]
        caption = e["caption"]
        rows.append(
            [
                e["frame"], f"{e['frame'] / rate:.4f}", e["kind"], best["raster"], best["source"],
                number(best["lat"], 6), number(best["lon"], 6), number(best["width_km"], 4), number(best["theta_deg"], 3),
                "yes" if best["mirror"] else "no", number(e["ncc"], 4),
                " ".join(caption["feature"].split()), " ".join(caption["feature_detail"].split()), " ".join(caption["camera"].split()),
            ]
        )  # fmt: skip
    write_csv(RESULTS / "moon/frames.csv", header, rows)


if __name__ == "__main__":
    satellite()
    moon()
