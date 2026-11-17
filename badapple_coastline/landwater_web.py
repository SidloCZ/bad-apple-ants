"""Export the place-map website data: every view's drawn extent, country,
continent, frames and a small original-vs-mask thumbnail.

Per-frame data follow the published video https://youtu.be/nw02Enk8o1I (the
2560x1920/30p satellite edition): its own frames.jsonl, frame n at (n-1)/30 s,
the same capture-date and imagery text as on screen, and its ending credits
(frames 6,563-6,955), which hold one fixed real view with no match score."""

from __future__ import annotations
import argparse, csv, io, json, shutil, subprocess, time
from pathlib import Path
import numpy as np
from PIL import Image
from badapple_media import ffmpeg_binary
from .pipeline import read_records
from .flash_reduction import SOURCE, CREDIT_START, CREDIT_PHOTO
from .footprint_map import footprint
from .native_2560 import capture_dates, source_label
from .landwater_osm import LAND, WATER, CONTINENT_COLORS, OUTPUT
from .geostats import CONTINENT_ORDER, OPEN_OCEAN, assign

ROOT = Path(__file__).resolve().parents[1]
# The site itself lives in its own repository (github.com/ikuto32/bad-apple-earth-map);
# pass --site <that checkout>/web to update it in place.
SITE = ROOT / "output/landwater-site/web"
THUMB = (160, 120)
YOUTUBE_ID = "nw02Enk8o1I"  # the published satellite edition the site follows
VIDEO_RECORDS = (
    ROOT / "output/coastline-precision-v4-2560x1920-30p-native-av1/frames.jsonl"
)  # that video's own per-frame records


def ranges(frames):
    out = []
    start = prev = frames[0]
    for f in frames[1:]:
        if f == prev + 1:
            prev = f
            continue
        out.append([start, prev])
        start = prev = f
    out.append([start, prev])
    return out


def thumbnails(records, views, out):
    out.mkdir(parents=True, exist_ok=True)
    firsts = {v["frames"][0]: v["view"] for v in views}  # the credits view's first frame is CREDIT_START
    movie = Path(json.loads((SOURCE / "run.json").read_text())["video_source"])
    command = [
        ffmpeg_binary("ffmpeg"),
        "-v",
        "error",
        "-nostdin",
        "-i",
        str(movie),
        "-an",
        "-vf",
        f"scale={THUMB[0]}:{THUMB[1]}:flags=area",
        "-frames:v",
        str(CREDIT_START + 1),
        "-pix_fmt",
        "gray",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    size = THUMB[0] * THUMB[1]
    palette = np.array([WATER, LAND], np.float32)
    for i in range(CREDIT_START + 1):
        raw = process.stdout.read(size)
        if len(raw) != size:
            raise ValueError("Original animation ended early")
        if i not in firsts:
            continue
        original = np.frombuffer(raw, np.uint8).reshape(THUMB[1], THUMB[0]).astype(np.float32) / 255
        view = next(v for v in views if v["view"] == firsts[i])
        with Image.open(view.get("native_mask_path") or records[i]["native_mask_path"]) as m:
            mask = np.asarray(m.convert("L").resize(THUMB, Image.Resampling.BOX), np.float32) / 255
        tile = np.zeros((THUMB[1], THUMB[0] * 2 + 4, 3), np.uint8)
        tile[:] = (8, 15, 23)
        tile[:, : THUMB[0]] = np.round(original[..., None] * 255).astype(np.uint8)
        tile[:, THUMB[0] + 4 :] = np.round(palette[0] + mask[..., None] * (palette[1] - palette[0])).astype(np.uint8)
        Image.fromarray(tile).save(out / f"{firsts[i]:05d}.png", optimize=True)
    process.stdout.close()
    process.wait()


def cloud_text(r):
    cloud = None if r["satellite"].get("parent_cloud_metrics") else r["satellite"].get("cloud_fraction")
    return "n/a" if cloud is None else f"{cloud:.1%}"


def frame_table(records, geo, credit_view):
    """One row per frame of https://youtu.be/nw02Enk8o1I, using the same values it shows."""
    views = geo["views"]
    rows = []
    for i, r in enumerate(records):
        credits = i >= CREDIT_START
        v = credit_view if credits else views[geo["frame_view"][i]]
        p = r["place"]
        rows.append(
            dict(
                frame=i + 1,
                time=f"{i // 1800:d}:{i % 1800 / 30:05.2f}",
                seconds=round(i / 30, 3),
                view=v["view"],
                place="" if credits else v["place"],
                country=v["country"],
                territory=v.get("territory") or "",
                continent=v["continent"],
                water="yes" if v["water"] else "no",
                disputed=(v.get("disputed_area") or {}).get("name") or "",
                latitude=round(r["latitude"], 6),
                longitude=round(r["longitude"], 6),
                width_km=round(r["width_km"], 3),
                rotation_deg=round(r["rotation_clockwise"], 2),
                mirrored="yes" if r.get("mirror_horizontal") else "no",
                miou="" if credits else round(r.get("native_miou", r.get("iou", 0)), 4),
                nearest_place=p["name"],
                nearest_place_country=p["country"],
                nearest_place_km=round(p["distance_km"]),
                captured=capture_dates(r),
                imagery=source_label(r),
                cloud=cloud_text(r),
                note="ending credits: fixed view (not a matched frame; score n/a)" if credits else "",
            )
        )
    return rows


def credits_view(records, geo, index):
    """The single real view held through the ending credits (frames 6,563-6,955)."""
    r = records[CREDIT_START]
    if r.get("render_source_frame") != CREDIT_PHOTO:
        raise ValueError("Unexpected ending-credits record")
    corners = footprint(r).tolist()
    a = assign(corners)
    return dict(
        a,
        view=index,
        corners_lonlat=corners,
        frames=list(range(CREDIT_START, len(records))),
        place=None,
        credits=True,
        width_km=r["width_km"],
        rotation_clockwise=r["rotation_clockwise"],
        mirror_horizontal=bool(r.get("mirror_horizontal")),
        nearest_settlement=r["place"],
        miou_mean=None,
        photo_dates=sorted({s["datetime"][:10] for s in r["satellite"]["scenes"]}),
        imagery=source_label(r),
        native_mask_path=r["native_mask_path"],
    )


def frames_csv(rows):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator=chr(10))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def export(output=OUTPUT, site=SITE, youtube=YOUTUBE_ID, with_thumbnails=True):
    output = Path(output)
    site = Path(site)
    data = site / "data"
    data.mkdir(parents=True, exist_ok=True)
    records = read_records(VIDEO_RECORDS)
    geo = json.loads((output / "geostats.json").read_text(encoding="utf-8"))
    views = geo["views"]
    stats = geo["stats"]
    for i in range(CREDIT_START):  # the statistics were computed from the same views the video shows
        if not np.allclose(footprint(records[i]), views[geo["frame_view"][i]]["corners_lonlat"]):
            raise ValueError(f"Frame {i} differs from the video")
    credit = credits_view(records, geo, len(views))
    countries = [c["country"] for c in stats["by_country"]]
    if credit["country"] not in countries:
        countries.append(credit["country"])
    cidx = {c: k for k, c in enumerate(countries)}
    continents = [c for c in CONTINENT_ORDER]
    kidx = {c: k for k, c in enumerate(continents)}
    rows = []
    for v in views + [credit]:
        p = v["nearest_settlement"]
        rows.append(
            [
                v["view"],
                [round(x, 5) for c in v["corners_lonlat"] for x in c],
                ranges(v["frames"]),
                cidx[v["country"]],
                kidx[v["continent"]],
                v.get("territory"),
                v["place"],
                round(v["width_km"], 3),
                round(v["rotation_clockwise"], 2),
                int(v["mirror_horizontal"]),
                None if v["miou_mean"] is None else round(v["miou_mean"], 4),
                [p["name"], p["country"], round(p["distance_km"])],
                v["photo_dates"][0] if v["photo_dates"] else None,
                v["imagery"],
                int(v["water"]),
                v["method"],
                (v.get("disputed_area") or {}).get("name"),
                (v.get("disputed_area") or {}).get("note"),
                v.get("sea"),
                [round(x, 5) for x in v["centre_lonlat"]],
                round(v["view_area_km2"], 1),
                [[c["country"], c["share"]] for c in v["countries"][:4]],
                int(bool(v.get("credits"))),
            ]
        )
    columns = [
        "id",
        "corners",
        "frame_ranges",
        "country",
        "continent",
        "territory",
        "place",
        "width_km",
        "rotation",
        "mirrored",
        "miou",
        "nearest",
        "photo_date",
        "imagery",
        "water",
        "method",
        "disputed",
        "disputed_note",
        "sea",
        "centre",
        "area_km2",
        "country_shares",
        "credits",
    ]
    units = json.loads(
        (ROOT / "assets/coastline/natural-earth/ne_10m_admin_0_map_units.geojson").read_text(encoding="utf-8")
    )["features"]
    ja = {}
    for f in units:
        q = f["properties"]
        if q.get("NAME_JA") and (q["ADMIN"] not in ja or q["GEOUNIT"] == q["ADMIN"]):
            ja[q["ADMIN"]] = q["NAME_JA"]
    ja[OPEN_OCEAN] = "外洋"
    countries_ja = [ja.get(c, c) for c in countries]
    continents_ja = {
        "Asia": "アジア",
        "North America": "北アメリカ",
        "Europe": "ヨーロッパ",
        "Oceania": "オセアニア",
        "Africa": "アフリカ",
        "South America": "南アメリカ",
        "Antarctica": "南極",
        OPEN_OCEAN: "外洋",
    }
    (data / "views.json").write_text(
        json.dumps(
            dict(
                columns=columns,
                countries=countries,
                countries_ja=countries_ja,
                continents=continents,
                continents_ja=[continents_ja[c] for c in continents],
                rows=rows,
            ),
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    (data / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    frames = frame_table(records, geo, credit)
    (data / "frames.json").write_text(
        json.dumps(
            dict(columns=["view", "miou"], rows=[[f["view"], f["miou"] if f["miou"] != "" else None] for f in frames]),
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    text = frames_csv(frames)
    (data / "frames.csv").write_text(text, encoding="utf-8-sig")
    (output / "frames.csv").write_text(text, encoding="utf-8-sig")
    meta = dict(
        build=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        youtube_id=youtube,
        video_records=VIDEO_RECORDS.relative_to(ROOT).as_posix(),
        fps=30,
        total_frames=len(records),
        matched_frames=CREDIT_START,
        continent_colors=CONTINENT_COLORS,
        thumbnails=with_thumbnails,
        thumbnail_size=list(THUMB),
    )
    (data / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    if with_thumbnails:
        thumbnails(records, views + [credit], data / "thumbs")
    return dict(views=len(rows), countries=len(countries), bytes=(data / "views.json").stat().st_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-thumbnails", action="store_true")
    parser.add_argument("--youtube", default=YOUTUBE_ID)
    parser.add_argument("--site", type=Path, default=SITE)
    args = parser.parse_args()
    print(export(site=args.site, youtube=args.youtube, with_thumbnails=not args.no_thumbnails))
