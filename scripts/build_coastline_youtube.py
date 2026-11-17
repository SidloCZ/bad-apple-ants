"""Deterministic publishing layout using pixels extracted from the final video."""

import argparse, hashlib, json, math, subprocess
from fractions import Fraction
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from badapple_coastline.pipeline import read_records
from badapple_media import ffmpeg_binary

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, default=Path("output/coastline-fullres-cuda-v2"))
p.add_argument("--frame", type=int, default=5292)
args = p.parse_args()
root = args.root
out = root / "youtube"
out.mkdir(exist_ok=True)
frame_index = args.frame
records = read_records(root / "frames.jsonl")
record = records[frame_index]
settings = json.loads((root / "run.json").read_text())
dictionary_meta = json.loads((Path(settings["dictionary"]) / "dictionary.json").read_text())
dictionary_count = dictionary_meta["count"]
extra_method = (
    " Candidates were retained using both area and directional coastline features, with contour-distance scoring and neighboring-frame candidate propagation."
    if "precision_objective" in settings
    else ""
)
mirrored_count = sum(bool(r.get("mirror_horizontal")) for r in records)
if settings.get("allow_mirror"):
    extra_method += f" Normal and horizontal mirror orientations were permitted ({dictionary_meta.get('count_including_mirrors',dictionary_count*2):,} sampled configurations); {mirrored_count:,} final frames are reflected and marked on screen."
if "structure_objective" in settings:
    extra_method += " Matching excludes the original video's fixed two-pixel top/bottom letterbox, retaining all original content pixels. Content-border agreement helps preserve contained shapes; previous views were remeasured using the same metric."
changes = (
    "cropping, resampling, rotation, and horizontal reflection where marked"
    if settings.get("allow_mirror")
    else "cropping, resampling, and rotation"
)
capture_years = "/".join(
    sorted(
        {s["datetime"][:4] for r in records for s in r["satellite"]["scenes"] if s.get("collection") != "landsat-c2-l2"}
    )
)
verification = json.loads((root / "badapple-coastline.verification.json").read_text())
map_note = (
    " The world map marks the current photo in gold and retains earlier footprints in cyan. Small boxes are enlarged for visibility; true photograph corners are preserved in the metadata."
    if verification.get("world_map")
    else ""
)
landsat_count = sum(bool(r.get("landsat_replacement")) for r in records)
annual_count = sum(bool(r["satellite"].get("annual_composite")) for r in records)
source = out / f"source-frame-{frame_index:05d}.png"
frame_time = float(Fraction(frame_index) / Fraction(settings.get("fps", "30000/1001")))
subprocess.run(
    [
        ffmpeg_binary("ffmpeg"),
        "-y",
        "-v",
        "error",
        "-nostdin",
        "-ss",
        f"{frame_time:.9f}",
        "-i",
        str(root / "badapple-coastline.mp4"),
        "-frames:v",
        "1",
        "-fps_mode",
        "passthrough",
        str(source),
    ],
    check=True,
)
with Image.open(source) as image:
    image = image.convert("RGB")
    if verification.get("display_content_box"):
        image = image.crop(tuple(verification["display_content_box"])).resize((1280, 960), Image.Resampling.LANCZOS)
    photo = image.crop((20, 70, 980, 790))
    original = image.crop((1000, 92, 1260, 287))

S = 3
navy = "#071019"
mint = "#75edcf"
white = "#f7fafc"
muted = "#9eafc0"
amber = "#ffd66a"
canvas = Image.new("RGB", (1280 * S, 720 * S), navy)
canvas.paste(photo.resize((960 * S, 720 * S), Image.Resampling.LANCZOS), (320 * S, 0))
d = ImageDraw.Draw(canvas)


def font(size, kind="regular"):
    paths = {"regular": "arial.ttf", "bold": "arialbd.ttf", "display": "impact.ttf"}
    return ImageFont.truetype("C:/Windows/Fonts/" + paths[kind], round(size * S))


def box(bounds, fill, radius=0, outline=None, width=1):
    bounds = tuple(round(v * S) for v in bounds)
    if radius:
        d.rounded_rectangle(bounds, radius=radius * S, fill=fill, outline=outline, width=round(width * S))
    else:
        d.rectangle(bounds, fill=fill, outline=outline, width=round(width * S))


def text(x, y, value, size, kind="regular", fill=white, anchor="lt"):
    d.text((round(x * S), round(y * S)), value, font=font(size, kind), fill=fill, anchor=anchor)


# The photograph is kept at its original aspect ratio; graphic overlays do not
# reshape or regenerate any geographic feature.
box((0, 0, 320, 684), navy)
box((318, 0, 322, 684), mint)
box((24, 24, 292, 61), "#2d291c", radius=8)
d.polygon([(38 * S, 51 * S), (47 * S, 34 * S), (56 * S, 51 * S)], outline=amber, width=2 * S)
d.line([(47 * S, 39 * S), (47 * S, 44 * S)], fill=amber, width=2 * S)
d.ellipse((46 * S, 47 * S, 48 * S, 49 * S), fill=amber)
text(67, 34, "FLASH WARNING", 16, "bold", amber)

text(24, 96, "BAD", 112, "display")
text(24, 211, "APPLE!!", 76, "display")
text(27, 311, "REAL EARTH", 29, "bold", mint)
text(28, 353, "6,955 satellite frames", 16, "regular", muted)
box((28, 395, 292, 397), "#31434e")
text(30, 421, "ORIGINAL FRAME", 16, "bold", muted)
canvas.paste(original.resize((260 * S, 195 * S), Image.Resampling.LANCZOS), (30 * S, 450 * S))
d.rectangle((29 * S, 449 * S, 291 * S, 646 * S), outline="#d3e1e6", width=S)

box((970, 24, 1250, 62), navy, radius=8)
text(
    1110,
    43,
    "REAL VIEW (MIRRORED)" if record.get("mirror_horizontal") else "REAL SATELLITE IMAGE",
    16,
    "bold",
    mint,
    anchor="mm",
)
box((343, 636, 502, 672), navy, radius=7)
country = record["place"]["country"].upper()
country_size = min(17, 235 / max(1, len(country)))
text(422, 654, country, country_size, "bold", white, anchor="mm")
box((514, 639, 824, 670), navy, radius=7)
coordinates = f"{abs(record['latitude']):.5f} {'N' if record['latitude']>=0 else 'S'} / {abs(record['longitude']):.5f} {'E' if record['longitude']>=0 else 'W'}"
text(669, 654, coordinates, 13, "regular", muted, anchor="mm")

# A separate callout connects the two actual crops from the same video frame.
p0 = (294, 540)
p1 = (464, 562)
p2 = (644, 505)
p3 = (825, 421)
curve = []
for step in range(81):
    t = step / 80
    u = 1 - t
    x = u**3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t**3 * p3[0]
    y = u**3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t**3 * p3[1]
    curve.append((round(x * S), round(y * S)))
d.line(curve, fill=navy, width=10 * S, joint="curve")
d.line(curve, fill=mint, width=5 * S, joint="curve")
angle = math.atan2(p3[1] - p2[1], p3[0] - p2[0])
head = [
    p3,
    (p3[0] - 27 * math.cos(angle) + 12 * math.sin(angle), p3[1] - 27 * math.sin(angle) - 12 * math.cos(angle)),
    (p3[0] - 27 * math.cos(angle) - 12 * math.sin(angle), p3[1] - 27 * math.sin(angle) + 12 * math.cos(angle)),
]
d.polygon([(round(x * S), round(y * S)) for x, y in head], fill=mint)

box((0, 684, 1280, 720), navy)
sentinel_years = sorted(
    {s["datetime"][:4] for s in record["satellite"]["scenes"] if s.get("collection") != "landsat-c2-l2"}
)
image_credit = "Contains modified Copernicus Sentinel data " + "/".join(sentinel_years)
if record.get("landsat_replacement"):
    image_credit = (
        image_credit + " | " if record["satellite"].get("mixed_sensors") else ""
    ) + "Landsat: U.S. Geological Survey"
text(
    18,
    690,
    image_credit + " | Mask: G. E. Kindermann (2025), CC BY 4.0 | doi:10.5281/zenodo.14899246",
    9,
    "regular",
    muted,
)
text(
    18,
    704,
    "© OpenStreetMap contributors · openstreetmap.org/copyright | © ESA WorldCover project 2020/2021 | Full credits in description",
    9,
    "regular",
    muted,
)

canvas.save(out / "thumbnail-master.png")
for size, name in [((3840, 2160), "thumbnail-3840x2160.jpg"), ((1280, 720), "thumbnail-1280x720.jpg")]:
    image = canvas if canvas.size == size else canvas.resize(size, Image.Resampling.LANCZOS)
    for quality in [94, 91, 88, 85, 82, 78]:
        image.save(out / name, quality=quality, optimize=True, subsampling=0)
        if (out / name).stat().st_size < 2_000_000:
            break
    assert (out / name).stat().st_size < 2_000_000

title = "[Photosensitive Seizure Warning] Bad Apple!! Recreated with Real Satellite Images"
description = f"""PHOTOSENSITIVE SEIZURE WARNING
This video contains rapid image changes, flashing, and high-contrast patterns that may trigger seizures in people with photosensitive epilepsy. Please avoid watching if you are sensitive to flashing images.

Bad Apple!!, recreated using the shapes of real coastlines, islands, lakes, and rivers.

For each of the 6,955 frames, I searched a global land/water mask for a similar silhouette, then retrieved satellite imagery of that location. Each frame shows one real geographic view, cropped, scaled, and rotated to match the original animation. Some views combine imagery from multiple dates at the same location to reduce clouds.

The large image is the satellite view. On the right are the original silhouette and the matched land/water mask. Coordinates, view width, rotation, and acquisition dates (or annual composite periods) are shown on screen. Place names identify the nearest mapped settlement, with its distance. mIoU averages the land and water overlap scores.{map_note}

The imagery uses Copernicus Sentinel-2; USGS Landsat 8/9 contributes to {landsat_count:,} frames with unavailable views or missing source pixels. {annual_count:,} photographs use the official ESA WorldCover Sentinel-2 annual median composites. These combine observations from the same location across a year; they are not single-date photographs.

Built with Rust + Rayon and CUDA (cuBLAS/cuFFT). I considered {dictionary_count:,} sampled geographic views from the original 10m mask, refined position, rotation and scale, and measured the final matches at the source video's original 1444 x 1080 resolution.{extra_method} This finite search does not guarantee a perfect match for every frame.

Satellite imagery
Contains modified Copernicus Sentinel data {capture_years}.
Sentinel-2 L2A via Earth Search / AWS Open Data:
https://registry.opendata.aws/sentinel-2-l2a-cogs/
Copernicus data notice:
https://cds.climate.copernicus.eu/licences/ec-sentinel

ESA WorldCover Sentinel-2 annual RGB composites (CC BY 4.0):
© ESA WorldCover project 2020/2021 / Contains modified Copernicus Sentinel data (2020/2021) processed by ESA WorldCover consortium.
https://registry.opendata.aws/esa-worldcover-vito-composites/
https://esa-worldcover.org/en/data-access
Some individual Sentinel-2 scenes and Landsat data are accessed via Microsoft Planetary Computer. Per-frame product/scene IDs, source dates/periods and processing are recorded with the video.

Landsat 8/9 Collection 2 Level-2:
Data available from the U.S. Geological Survey.
Accessed via Microsoft Planetary Computer:
https://planetarycomputer.microsoft.com/dataset/landsat-c2-l2
Landsat data policy:
https://www.usgs.gov/faqs/are-landsat-data-cloud-still-considered-be-within-public-domain

Land/water mask and map data
Georg Erich Kindermann (2025), Global Land / Water Mask Map with 0.3 arc sec. (10 m) Resolution, v1.
https://doi.org/10.5281/zenodo.14899246
CC BY 4.0: https://creativecommons.org/licenses/by/4.0/
Changes: {changes}. Imagery also uses cloud-screened mosaics of the same location.
The mask includes lakes and rivers as well as sea coasts.

Underlying mask sources: ESA WorldCover, OpenStreetMap, and Copernicus DEM.
© ESA WorldCover project 2020/2021.
https://esa-worldcover.org/en/data-access
© OpenStreetMap contributors (ODbL).
https://www.openstreetmap.org/copyright
Place names: Natural Earth (public domain).
https://www.naturalearthdata.com/about/terms-of-use/

Music and original animation
Song: "Bad Apple!! feat. nomico" — Alstroemeria Records.
Arrangement: Masayoshi Minoshima. Lyrics: Haruka. Vocals: nomico.
Original composition: ZUN / Touhou Project.
https://alst.net/
Original shadow-art PV: Anira (あにら).
https://www.nicovideo.jp/watch/sm8628149
Music and original animation belong to their respective creators.

Photosensitivity information:
https://www.epilepsy.org.uk/info/seizure-triggers/photosensitive-epilepsy

#BadApple #Touhou #SatelliteImagery
"""
if settings.get("display_treatment"):
    description = description.replace(
        "This video contains rapid image changes, flashing, and high-contrast patterns that may trigger seizures in people with photosensitive epilepsy. Please avoid watching if you are sensitive to flashing images.",
        "Flash-reduced edition. High-contrast patterns and motion remain; individual sensitivity varies. A warning is retained because screening cannot guarantee prevention of seizures.",
    )
    if not annual_count:
        start = description.index("ESA WorldCover Sentinel-2 annual RGB composites (CC BY 4.0):")
        stop = description.index("Some individual Sentinel-2 scenes", start)
        description = description[:start] + description[stop:]
        description = description.replace(
            f" {annual_count:,} photographs use the official ESA WorldCover Sentinel-2 annual median composites. These combine observations from the same location across a year; they are not single-date photographs.",
            "",
        )
    size = settings["output_size"]
    audio_note = (
        "Original audio is copied without tempo or pitch changes."
        if settings.get("audio_mode") == "original"
        else (
            "Audio: AAC 192 kbps, no tempo/pitch filter, trimmed to the video length."
            if settings.get("audio_mode") == "trim"
            else "Audio is shortened by 0.1% for synchronization."
        )
    )
    treatment = (
        "ONLY satellite photos have exposure/luminance normalization; no temporal image blending."
        if settings.get("schema", "").startswith("coastline-luminance-normalized-30p-")
        else "ONLY satellite photos have flash/color treatment."
    )
    if settings.get("local_map"):
        treatment += " LOCAL uses the native 10 m mask: north up, gold=current footprint, cyan=past views."
    note = f"Display: {size[0]}x{size[1]}, 30.000p, NDF, 6,955 frames once each. {treatment} The original comparison is decoded from its source and resized only; it may flash. {audio_note} Credits hold one real view; scores are n/a. Screening covers satellite pixels only; whole-video compliance is not claimed. Independent reference-view checks, not medical/broadcast certification.\nhttps://www.w3.org/TR/WCAG22/#dfn-general-flash-and-red-flash-thresholds\n\n"
    description = description.replace("Satellite imagery\n", note + "Satellite imagery\n", 1)
assert len(title) <= 100 and len(description) <= 5000 and len(description.encode("utf-8")) < 5000
(out / "youtube-title.txt").write_text(title + "\n", encoding="utf-8")
(out / "youtube-description.txt").write_text(description, encoding="utf-8")
meta = dict(
    source_video=str((root / "badapple-coastline.mp4").resolve()),
    source_video_sha256=json.loads((root / "summary.json").read_text())["movie_sha256"],
    match_run_id=settings["match_run_id"],
    dictionary_sha256=settings["dictionary_sha256"],
    mirrored_frames=mirrored_count,
    frame_index_zero_based=frame_index,
    on_screen_frame_number=frame_index + 1,
    timestamp_seconds=frame_index / float(Fraction(settings.get("fps", "30000/1001"))),
    source_frame_mirror_horizontal=bool(record.get("mirror_horizontal")),
    source_frame=str(source.resolve()),
    source_frame_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    crops={"satellite": [20, 70, 980, 790], "original_silhouette": [1000, 92, 1260, 287]},
    source_content_box=verification.get("display_content_box"),
    crop_reference_canvas=[1280, 960],
    modifications="Exact crops from the encoded video; proportional resizing, typography, border, and a comparison arrow. No generated or reshaped geography.",
    title=title,
    title_characters=len(title),
    description_characters=len(description),
    description_utf8_bytes=len(description.encode("utf-8")),
    exports={
        name: dict(bytes=(out / name).stat().st_size, sha256=hashlib.sha256((out / name).read_bytes()).hexdigest())
        for name in ["thumbnail-3840x2160.jpg", "thumbnail-1280x720.jpg"]
    },
    thumbnail_guidance="https://support.google.com/youtube/answer/72431?hl=en",
    photosensitivity_guidance="https://www.epilepsy.org.uk/info/seizure-triggers/photosensitive-epilepsy",
    music_credit_reference="https://alst.net/ardw0001/",
)
(out / "thumbnail-source.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
readme = f"""# YouTube upload package

Title ({len(title)} characters):
{title}

Use youtube-description.txt for the description ({len(description)} characters; includes source credits and the photosensitivity warning).

Recommended thumbnail: thumbnail-3840x2160.jpg, {meta['exports']['thumbnail-3840x2160.jpg']['bytes']:,} bytes.
Smaller export: thumbnail-1280x720.jpg, {meta['exports']['thumbnail-1280x720.jpg']['bytes']:,} bytes.
Both are 16:9 RGB JPEGs under 2 MB. The 3840x2160 dimensions follow the current official YouTube guidance:
https://support.google.com/youtube/answer/72431?hl=en

The source is the finished video's frame {frame_index+1} (zero-based index {frame_index}), at {frame_index*1001/30000:.4f} seconds.
The satellite image and original silhouette are exact crops from that same frame. The geography is not regenerated or reshaped.
The higher-resolution export upscales the original video imagery and renders the typography at 3840x2160.
Source coordinates: {coordinates}. Nearest mapped settlement country: {record['place']['country']}.

thumbnail-master.png is the lossless layout master. {source.name} is the unmodified decoded source frame.
thumbnail-source.json records source hashes, crop rectangles, and export hashes.
"""
(out / "UPLOAD.md").write_text(readme, encoding="utf-8")
print(
    json.dumps(
        {
            k: meta[k]
            for k in ["title", "title_characters", "description_characters", "description_utf8_bytes", "exports"]
        },
        indent=2,
    ),
    flush=True,
)
