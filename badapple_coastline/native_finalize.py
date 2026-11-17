"""Verify full-resolution source geometry, compose and decode the new edition."""

from __future__ import annotations
import argparse, hashlib, json, subprocess, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
from .cuda_backend import CudaAtlas
from .cuda_search import OUTPUT, FRAMES, KEYS
from .data import digest, MASK_CREDIT
from .pipeline import read_records, write_json
from .quality import file_quality
from .render import render
from .atlas import corners, view_transform
from badapple_media import Frames, ffmpeg_binary, read_exact


def credits(output, records):
    settings = json.loads((output / "run.json").read_text())
    sentinel_years = sorted(
        {s["datetime"][:4] for r in records for s in r["satellite"]["scenes"] if s.get("collection") != "landsat-c2-l2"}
    )
    landsat = sum(bool(r.get("landsat_replacement")) for r in records)
    annual = sum(bool(r["satellite"].get("annual_composite")) for r in records)
    mixed = sum(bool(r["satellite"].get("mixed_sensors")) for r in records)
    text = MASK_CREDIT + "\nhttps://doi.org/10.5281/zenodo.14899246\nhttps://creativecommons.org/licenses/by/4.0/\n"
    text += "Underlying mask sources acknowledged: ESA WorldCover, OpenStreetMap contributors, Copernicus DEM.\n"
    text += "© OpenStreetMap contributors, https://www.openstreetmap.org/copyright (ODbL source data).\n"
    text += "© ESA WorldCover project 2020/2021 / Contains modified Copernicus Sentinel data (2020/2021) processed by ESA WorldCover consortium.\nhttps://esa-worldcover.org/en/data-access\n"
    text += (
        "Contains modified Copernicus Sentinel data "
        + ", ".join(sentinel_years)
        + "\nhttps://cds.climate.copernicus.eu/licences/ec-sentinel\n"
    )
    text += "Sentinel-2 L2A visual/SCL: Earth Search / AWS Open Data, original COG window reads; some retained early frames via Microsoft Planetary Computer.\nhttps://registry.opendata.aws/sentinel-2-l2a-cogs/\nhttps://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a\n"
    text += "Landsat 8/9 Collection 2 Level-2: Data available from the U.S. Geological Survey. Public domain.\nhttps://www.usgs.gov/faqs/are-landsat-data-cloud-still-considered-be-within-public-domain\n"
    text += "Landsat via Microsoft Planetary Computer STAC and raw georeferenced crop API. uint16 DN*0.0000275-0.2, then a fixed sRGB transfer. QA cloud/snow screening.\n"
    text += f"Landsat contributes to {landsat} frames, including {mixed} frames in which real Landsat pixels fill only missing Sentinel-2 source pixels at the same coordinates.\n"
    if annual:
        text += f"{annual} frames use official ESA WorldCover Sentinel-2 RGBNIR annual median composites (CC BY 4.0), via public Terrascope true-color WMTS. These are annual products rather than single-date captures.\n"
        text += "https://registry.opendata.aws/esa-worldcover-vito-composites/\nhttps://docs.terrascope.be/Developers/WebServices/OGC/MapProxy.html\n"
    text += "All pictures retain one geographic field of view. Rigid cropping, scaling, rotation, reprojection and same-location temporal mosaics; no invented geography or synthetic pixel fill. Retained baseline photos are geographically recropped to the original video aspect.\n"
    text += "Exact source scene/product identifiers, acquisition dates or annual periods, source URLs, geographic footprints, image hashes and processing are recorded in frames.jsonl. Parent-image cloud estimates are not displayed as exact cropped-view statistics. Annual residual cloud fractions are unavailable and displayed as n/a.\n"
    text += "Original mask sampled directly from its lossless 0.3 arcsecond data; source resolution1444x1080, without spatial downscaling. "
    if "structure_objective" in settings:
        text += "Matching excludes fixed top/bottom2pixel source-video letterbox rows, using all1444x1076 true-content pixels at their original coordinates. Area, contours and content-frame border agreement are measured; descriptor/FFT padding extends the nearest content row only. Original frames and all native geography pixels remain intact. Baseline and new views use this same content metric; historical full-frame scores are not directly comparable. "
    else:
        text += "Final targets and mask scores use all original1444x1080 pixels. "
    text += "Finite sampled global search, not a continuous global optimum. White=land, black=water, including lakes and rivers.\n"
    text += f"Horizontal reflections: {sum(bool(r.get('mirror_horizontal')) for r in records)} frames. Reflected views are marked MIRRORED VIEW; orientation is recorded per frame. Photographs and native masks use the same reflected geographic view.\n"
    text += "Place names: Natural Earth (public domain), https://www.naturalearthdata.com/about/terms-of-use/ . Labels identify the nearest mapped settlement and distance, not an asserted exact locality.\n"
    text += "World map: Natural Earth 1:110m land (public domain). Gold=current photograph footprint; cyan=all earlier footprints, retained to the end. Rotated views use actual photograph corners; small boxes are enlarged on the locator for visibility. True unexpanded coordinates are retained in badapple-coastline.footprints.json.\n"
    text += "Video/music/animation: original Bad Apple material; respective creators retain their rights. Song: Bad Apple!! feat. nomico, Alstroemeria Records; arrangement Masayoshi Minoshima, lyrics Haruka, vocals nomico, original composition ZUN. Original shadow PV: Anira (あにら), https://www.nicovideo.jp/watch/sm8628149 .\n"
    (output / "CREDITS.txt").write_text(text, encoding="utf-8")


def finalize(output=OUTPUT, encoder="h264_nvenc"):
    output = Path(output)
    settings = json.loads((output / "run.json").read_text())
    records = read_records(output / "frames.jsonl")
    if [r["frame"] for r in records] != list(range(settings["frames"])):
        raise ValueError("Missing, duplicate or reordered frames")
    if digest(Path(settings["video_source"])) != settings["video_source_sha256"]:
        raise ValueError("Video source changed")
    if digest(Path(settings["dictionary"]) / "dictionary.json") != settings["dictionary_sha256"]:
        raise ValueError("Dictionary changed")
    if (
        settings.get("additional_dictionary")
        and digest(Path(settings["additional_dictionary"]) / "dictionary.json")
        != settings["additional_dictionary_sha256"]
    ):
        raise ValueError("Additional dictionary changed")
    old = read_records(Path(settings["baseline"]) / "frames.jsonl")
    metadata = json.loads((FRAMES / "frames.json").read_text())
    frames = Frames(FRAMES, metadata)
    raw = frames.thumbnails()
    masks = output / "native-masks"
    masks.mkdir(exist_ok=True)
    started = time.perf_counter()
    rows = []
    with CudaAtlas() as cuda:
        for i, r in enumerate(records):
            if r["match_run_id"] != settings["match_run_id"]:
                raise ValueError("Imagery belongs to another run")
            geo = np.array([[r[k] for k in KEYS], [old[i][k] for k in KEYS]], np.float64)
            if settings.get("allow_mirror"):
                geo = np.column_stack(
                    [geo, [int(r.get("mirror_horizontal", False)), int(old[i].get("mirror_horizontal", False))]]
                )
            binary = np.uint8(raw[i] >= 128)
            precise = "precision_objective" in settings
            scores = (
                cuda.score_structure(geo, binary, border_y=settings["content_border_y"])
                if "structure_objective" in settings
                else cuda.score_shape(geo, binary) if precise else cuda.score(geo, binary)
            )
            if (scores["missing"] > 0).any():
                raise ValueError("Missing native source pixels")
            if abs(scores["objective"][0] - r["native_objective"]) > 1e-12:
                raise ValueError("Selected geometry score changed")
            if abs(scores["objective"][1] - r["baseline_fullres_metrics"]["native_objective"]) > 1e-12:
                raise ValueError("Baseline score changed")
            if scores["objective"][0] < scores["objective"][1] - 1e-12:
                raise ValueError("Full-resolution objective regressed")
            if precise:
                if abs(scores["precision_objective"][0] - r["precision_objective"]) > 1e-12:
                    raise ValueError("Precision score changed")
                if abs(scores["precision_objective"][1] - r["baseline_fullres_metrics"]["precision_objective"]) > 1e-12:
                    raise ValueError("Baseline precision score changed")
                if (
                    scores["miou"][0] < scores["miou"][1] - 1e-12
                    or scores["precision_objective"][0] < scores["precision_objective"][1] - 1e-12
                ):
                    raise ValueError("Precision acceptance floor regressed")
            if "structure_objective" in settings:
                np.testing.assert_array_equal(scores["structure_counts"][0], r["structure_counts"])
                np.testing.assert_array_equal(
                    scores["structure_counts"][1], r["baseline_fullres_metrics"]["structure_counts"]
                )
                if scores["structure_objective"][0] < scores["structure_objective"][1] - 1e-12:
                    raise ValueError("Content structure floor regressed")
                if abs(scores["structure_objective"][0] - r["structure_objective"]) > 1e-12:
                    raise ValueError("Content structure score changed")
            exact = cuda.sample(geo[:1], (1444, 1080))[0] * 255
            signature = hashlib.sha256(
                json.dumps([geo[0].tolist(), [1444, 1080], settings["dictionary_sha256"]]).encode()
            ).hexdigest()[:16]
            # Keep the searched primary mask intact when imagery selected another view.
            path = masks / f"{i:05d}-{signature}.png"
            if path.is_file():
                with Image.open(path) as image:
                    existing = np.asarray(image.convert("L"))
                if not np.array_equal(existing, exact):
                    Image.fromarray(exact).save(path)
            else:
                Image.fromarray(exact).save(path)
            r.update(
                native_mask_path=str(path.resolve()),
                native_mask_sha256=digest(path),
                native_pixels_verified=True,
                native_objective_gain=float(scores["objective"][0] - scores["objective"][1]),
            )
            satellite = r["satellite"]
            photo = Path(satellite["path"])
            if satellite.get("placeholder") or satellite["coverage"] != 1 or digest(photo) != satellite["image_sha256"]:
                raise ValueError("Missing/changed real photo")
            if not file_quality(photo)["acceptable"]:
                raise ValueError("Photo whiteout")
            if tuple(satellite["size"]) != (1444, 1080):
                raise ValueError("Photograph aspect differs from native mask")
            expected = np.asarray(
                corners(
                    view_transform(*geo[0, :4], (1444, 1080), mirror_horizontal=bool(r.get("mirror_horizontal"))),
                    (1444, 1080),
                )
            )
            if not np.allclose(np.asarray(satellite["footprint_lonlat"]), expected, rtol=0, atol=1e-10):
                raise ValueError("Photograph footprint differs from the measured native view")
            rows.append(
                dict(
                    frame=i,
                    miou=r["native_miou"],
                    baseline_miou=r["baseline_fullres_metrics"]["native_miou"],
                    objective=r["native_objective"],
                    baseline_objective=r["baseline_fullres_metrics"]["native_objective"],
                    constant=bool(scores["counts"][0, 4] == 0),
                )
            )
            if precise:
                rows[-1].update(
                    precision_objective=r["precision_objective"],
                    baseline_precision_objective=r["baseline_fullres_metrics"]["precision_objective"],
                    soft_contour_f1=r["soft_contour_f1"],
                    baseline_soft_contour_f1=r["baseline_fullres_metrics"]["soft_contour_f1"],
                )
            if "structure_objective" in settings:
                rows[-1].update(
                    structure_objective=r["structure_objective"],
                    baseline_structure_objective=r["baseline_fullres_metrics"]["structure_objective"],
                )
            if (i + 1) % 100 == 0:
                print(
                    f"full-resolution final verification: {i+1}/{len(records)}; {time.perf_counter()-started:.1f}s",
                    flush=True,
                )
    encoded = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    (output / "frames.jsonl").write_text(encoded, encoding="utf-8")
    (output / "frames.partial.jsonl").write_text(encoded, encoding="utf-8")
    credits(output, records)
    extra = {}
    if "precision_objective" in settings:
        extra = dict(
            mean_precision_objective=float(np.mean([r["precision_objective"] for r in rows])),
            baseline_mean_precision_objective=float(np.mean([r["baseline_precision_objective"] for r in rows])),
            mean_soft_contour_f1=float(np.mean([r["soft_contour_f1"] for r in rows])),
            baseline_mean_soft_contour_f1=float(np.mean([r["baseline_soft_contour_f1"] for r in rows])),
            regressed_miou_frames=0,
            regressed_precision_objective_frames=0,
        )
    if "structure_objective" in settings:
        extra.update(
            mean_structure_objective=float(np.mean([r["structure_objective"] for r in rows])),
            baseline_mean_structure_objective=float(np.mean([r["baseline_structure_objective"] for r in rows])),
            regressed_structure_objective_frames=0,
            content_bounds=settings["content_bounds"],
            evaluated_pixels=settings["evaluated_pixels"],
            excluded_presentation_pixels=settings["excluded_presentation_pixels"],
            historical_baseline_full_frame_mean_miou=settings["historical_baseline_full_frame_mean_miou"],
            mean_frame_edge_error=float(np.mean([r["frame_edge_error"] for r in records])),
            baseline_mean_frame_edge_error=float(
                np.mean([r["baseline_fullres_metrics"]["frame_edge_error"] for r in records])
            ),
            metric_note="Both baseline and new views freshly measured on original-pixel content ROI, excluding fixed source-video letterbox rows",
        )
    write_json(
        output / "native-verification.json",
        dict(
            frames=len(rows),
            evaluation_size=[1444, 1080],
            native_pixel_differences=0,
            **extra,
            regressed_objective_frames=0,
            mean_miou=float(np.mean([r["miou"] for r in rows])),
            baseline_mean_miou=float(np.mean([r["baseline_miou"] for r in rows])),
            mean_objective=float(np.mean([r["objective"] for r in rows])),
            baseline_mean_objective=float(np.mean([r["baseline_objective"] for r in rows])),
            improved_frames=sum(r["objective"] > r["baseline_objective"] + 1e-7 for r in rows),
            elapsed_seconds=time.perf_counter() - started,
            nonconstant_frames=sum(not r["constant"] for r in rows),
            baseline_nonconstant_mean_miou=float(np.mean([r["baseline_miou"] for r in rows if not r["constant"]])),
            nonconstant_mean_miou=float(np.mean([r["miou"] for r in rows if not r["constant"]])),
        ),
    )
    write_json(
        output / "imagery-status.json",
        dict(
            frames=len(records),
            complete=len(records),
            black_placeholder_frames=0,
            landsat_frames=sum(bool(r.get("landsat_replacement")) for r in records),
            mixed_sensor_frames=sum(bool(r["satellite"].get("mixed_sensors")) for r in records),
            annual_composite_frames=sum(bool(r["satellite"].get("annual_composite")) for r in records),
            alternative_frames=sum(r["imagery_candidate_rank"] > 0 for r in records),
            actual_mean_miou=float(np.mean([r["native_miou"] for r in records])),
            regressed_objective_frames=0,
            native_pixels_verified=len(records),
        ),
    )
    atlas = SimpleNamespace(metadata={"global_coverage": True})
    render(
        Path(settings["video_source"]),
        frames,
        records,
        atlas,
        output / "badapple-coastline.mp4",
        encoder,
        settings["image_quality_policy"],
        world_map=True,
    )
    audit(output)


def audit(output):
    output = Path(output)
    movie = output / "badapple-coastline.mp4"
    records = read_records(output / "frames.jsonl")
    p = subprocess.Popen(
        [
            ffmpeg_binary("ffmpeg"),
            "-v",
            "error",
            "-xerror",
            "-nostdin",
            "-i",
            str(movie),
            "-an",
            "-vf",
            "crop=960:720:20:70,scale=96:72:flags=area",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
    )
    rows = []
    while True:
        raw = read_exact(p.stdout, 96 * 72 * 3)
        if not raw:
            break
        image = np.frombuffer(raw, np.uint8).reshape(72, 96, 3)
        light = image.mean(axis=2)
        rows.append(
            dict(
                frame=len(rows),
                white=float((image.min(axis=2) >= 225).mean()),
                bright=float((light >= 210).mean()),
                mean=float(light.mean()),
                std=float(light.std()),
            )
        )
    if p.wait() != 0 or len(rows) != len(records):
        raise ValueError("Encoded video frame/decode error")
    white = [
        r["frame"] for r in rows if r["white"] >= 0.36 or r["bright"] >= 0.61 or (r["mean"] >= 170 and r["std"] < 3)
    ]
    if white:
        raise ValueError(f"Encoded whiteout frames: {white}")
    subprocess.run(
        [ffmpeg_binary("ffmpeg"), "-v", "error", "-xerror", "-nostdin", "-i", str(movie), "-f", "null", "-"], check=True
    )
    p = subprocess.Popen(
        [
            ffmpeg_binary("ffmpeg"),
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(movie),
            "-an",
            "-vf",
            "crop=1240:48:20:910",
            "-pix_fmt",
            "gray",
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
    )
    last = None
    total = peak = 0.0
    count = 0
    while True:
        raw = read_exact(p.stdout, 1240 * 48)
        if not raw:
            break
        values = np.frombuffer(raw, np.uint8).astype(np.int16)
        if last is not None:
            error = float(np.abs(values - last).mean())
            total += error
            peak = max(peak, error)
        last = values
        count += 1
    if p.wait() != 0 or count != len(records):
        raise ValueError("Footer decode failed")
    summary = json.loads((output / "native-verification.json").read_text())
    verification = json.loads(movie.with_suffix(".verification.json").read_text())
    if verification.get("world_map"):
        from .footprint_map import audit_movie

        summary["world_map"] = audit_movie(movie, records)
    summary.update(
        movie_sha256=digest(movie),
        frames=len(records),
        decode_errors=0,
        decoded_whiteout_frames=white,
        mirrored_frames=sum(bool(r.get("mirror_horizontal")) for r in records),
        black_placeholder_frames=sum(bool(r["satellite"].get("placeholder")) for r in records),
        landsat_frames=sum(bool(r.get("landsat_replacement")) for r in records),
        full_resolution_masks_verified=True,
        footer_mean_consecutive_pixel_error=total / (count - 1),
        footer_max_consecutive_pixel_error=peak,
        all_photos_have_full_real_coverage=True,
        annual_composite_frames=sum(bool(r["satellite"].get("annual_composite")) for r in records),
        mixed_sensor_frames=sum(bool(r["satellite"].get("mixed_sensors")) for r in records),
        alternative_frames=sum(r["imagery_candidate_rank"] > 0 for r in records),
    )
    write_json(output / "summary.json", summary)
    write_json(output / "decoded-image-audit.json", rows)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=OUTPUT)
    p.add_argument("--audit-only", action="store_true")
    a = p.parse_args()
    audit(a.output) if a.audit_only else finalize(a.output)
