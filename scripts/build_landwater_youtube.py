"""YouTube package for the land/water + OpenStreetMap edition: title, description
with statistics and all credits, and a thumbnail cut from the finished video."""

import argparse, hashlib, json, subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from badapple_media import ffmpeg_binary

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, default=Path("output/landwater-osm"))
p.add_argument("--frame", type=int, default=5292)
p.add_argument("--site", default="[MAP WEBSITE URL]")
args = p.parse_args()
root = args.root
out = root / "youtube"
out.mkdir(exist_ok=True)
summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
geo = json.loads((root / "geostats.json").read_text(encoding="utf-8"))
s = geo["stats"]
movie = root / "badapple-landwater-osm.mp4"
top = ", ".join(f"{c['country']} {c['frames']:,}" for c in s["by_country"][:10])
continents = ", ".join(f"{c['continent']} {c['frames']:,}" for c in s["by_continent"])

correction_date = "2026-11-12"
title = "⚠️FLASHING⚠️Bad Apple!! but every frame is real land & water on Earth (OpenStreetMap)"
title_ja = "⚠️点滅注意⚠️Bad Apple!! 全フレームを地球上の実在する陸と水で再現（OpenStreetMap）"
description = f"""PHOTOSENSITIVE SEIZURE WARNING
The main picture is the matched land/water silhouette, so it changes between light and dark like the original animation. Individual sensitivity varies.

Bad Apple!!, drawn with the real land and water of Earth.

This is the land & water edition you asked for in the comments: the matched land/water view is now the main picture, with the satellite photo, an OpenStreetMap map of the area and the world map beside it. Every frame shows one real place, cropped, scaled and rotated (mirrored where marked) to match the original silhouette. The end of the video shows the statistics: {s['matched_frames']:,} matched frames, {s['views']:,} distinct views, {s['places']:,} places, {s['countries']} countries and territories, {s['continents']} continents.

Every place on a map (country, continent, frames and drawn extent of every view), plus a list of every frame:
{args.site}

By continent (frames): {continents}.
Top countries and territories (frames): {top}.

Correction ({correction_date}): the statistics at the end show "AREA SHOWN 24.2 million km²". That figure added up the area of every view, so places shown in several views were counted more than once. Counting each place once, the views cover {s['area_union_km2'] / 1e6:.1f} million km², about {s['area_union_share_of_earth'] * 100:.1f}% of Earth's surface (oceans included). The map website shows the corrected figure. Thank you to the viewer who pointed this out.

How it is drawn
- Main picture: G. E. Kindermann's global 10 m land/water mask, sampled at the display resolution directly from the original 0.3 arc-second data, through exactly the same view (position, width, rotation, reflection) that was matched.
- OpenStreetMap panel: north-up map of the surroundings, five times the view width; gold = this view, cyan = earlier nearby views.
- Satellite photo: the same view in Copernicus Sentinel-2 (Landsat 8/9 where needed), tone-limited per frame.
- Country = the country or territory with the largest land overlap of each view (Natural Earth, de facto boundaries; no position on sovereignty implied). Open-water views use the nearest country within 200 nautical miles. Russia is split at the Urals.

Display: 2560x1920, 30 fps, all 6,955 frames once each, AV1. Audio: the original song.

Land/water mask
Georg Erich Kindermann (2025), Global Land / Water Mask Map with 0.3 arc sec. (10 m) Resolution, v1.
https://doi.org/10.5281/zenodo.14899246 — CC BY 4.0: https://creativecommons.org/licenses/by/4.0/
Changes: cropping, resampling, rotation, and horizontal reflection where marked.
Underlying mask sources: ESA WorldCover, OpenStreetMap, Copernicus DEM. © ESA WorldCover project 2020/2021.

Map
Map data © OpenStreetMap contributors, available under the Open Database License: https://www.openstreetmap.org/copyright
Vector tiles: Protomaps (https://protomaps.com), basemap style BSD-3-Clause; fonts Noto Sans (SIL OFL). Rendered with MapLibre GL JS.
Boundaries, places and world map: Made with Natural Earth (public domain), https://www.naturalearthdata.com/

Satellite imagery
Contains modified Copernicus Sentinel data 2019/2020/2021. Sentinel-2 L2A via Earth Search / AWS Open Data: https://registry.opendata.aws/sentinel-2-l2a-cogs/
Landsat 8/9 Collection 2 Level-2: data available from the U.S. Geological Survey (public domain), via Microsoft Planetary Computer.

Music and original animation
Song: "Bad Apple!! feat. nomico" — Alstroemeria Records. Arrangement: Masayoshi Minoshima. Lyrics: Haruka. Vocals: nomico.
Original composition: ZUN / Touhou Project. Original shadow-art PV: Anira (あにら), https://www.nicovideo.jp/watch/sm8628149
Music and original animation belong to their respective creators.

AI assistance (Claude by Anthropic) was used for parts of the code. All pictures are real map and satellite data; no AI-generated imagery.

Photosensitivity information: https://www.epilepsy.org.uk/info/seizure-triggers/photosensitive-epilepsy

#BadApple #Touhou #OpenStreetMap
"""

source = out / f"source-frame-{args.frame:05d}.png"
subprocess.run(
    [
        ffmpeg_binary("ffmpeg"),
        "-y",
        "-v",
        "error",
        "-nostdin",
        "-ss",
        f"{args.frame / 30:.6f}",
        "-i",
        str(movie),
        "-frames:v",
        "1",
        str(source),
    ],
    check=True,
)
frame = Image.open(source).convert("RGB")
main = frame.crop((40, 140, 1960, 1580))
original = frame.crop((2000, 184, 2520, 574))
W, H = 3840, 2160
thumb = Image.new("RGB", (W, H), (8, 15, 23))
d = ImageDraw.Draw(thumb)
fonts = Path("C:/Windows/Fonts")
F = lambda n, k="seguisb.ttf": ImageFont.truetype(str(fonts / k), n)
panel = main.resize((2520, 1890), Image.Resampling.LANCZOS)
thumb.paste(panel, (90, 180))
d.rectangle((89, 179, 2610, 2070), outline=(60, 84, 98), width=3)
orig = original.resize((1080, 810), Image.Resampling.LANCZOS)
thumb.paste(orig, (2680, 180))
d.rectangle((2679, 179, 3760, 990), outline=(60, 84, 98), width=3)
d.text((2680, 1120), "BAD APPLE!!", font=F(150), fill="white", anchor="ls")
d.text((2680, 1290), "REAL LAND", font=F(150), fill="#73e6d7", anchor="ls")
d.text((2680, 1460), "& WATER", font=F(150), fill="#73e6d7", anchor="ls")
d.text(
    (2680, 1600),
    f"{s['countries']} countries · {s['places']:,} places",
    font=F(64, "segoeui.ttf"),
    fill="#c9d6df",
    anchor="ls",
)
d.text((2680, 1700), "every frame is a real place on Earth", font=F(58, "segoeui.ttf"), fill="#9eafc0", anchor="ls")
d.text((2680, 2050), "⚠ flashing", font=F(60), fill="#ffcf6c", anchor="ls")
thumb.save(out / "thumbnail-master.png")
thumb.save(out / "thumbnail-3840x2160.jpg", quality=90, optimize=True)
thumb.resize((1280, 720), Image.Resampling.LANCZOS).save(out / "thumbnail-1280x720.jpg", quality=90, optimize=True)
(out / "youtube-title.txt").write_text(title + "\n", encoding="utf-8")
(out / "youtube-description.txt").write_text(description, encoding="utf-8")
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
(out / "UPLOAD.md").write_text(
    f"""# YouTube upload package

Title ({len(title)} characters):
{title}

Japanese localized title ({len(title_ja)} characters):
{title_ja}

Description: youtube-description.txt ({len(description)} characters). {'Replace `' + args.site + '` with the map website URL once it is published.' if args.site.startswith('[') else 'Map website: ' + args.site}

Thumbnail: thumbnail-3840x2160.jpg ({(out / 'thumbnail-3840x2160.jpg').stat().st_size:,} bytes) or thumbnail-1280x720.jpg ({(out / 'thumbnail-1280x720.jpg').stat().st_size:,} bytes).
Cut from frame {args.frame + 1} of the finished video (main land/water panel and original silhouette, exact crops, resized).

Video: {movie.name}, SHA-256 {summary['movie_sha256']}
""",
    encoding="utf-8",
)
print(
    json.dumps(
        dict(
            title_chars=len(title),
            description_chars=len(description),
            thumbnail=sha(out / "thumbnail-3840x2160.jpg")[:16],
        ),
        indent=1,
    )
)
