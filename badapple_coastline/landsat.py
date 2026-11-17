"""Use real USGS Landsat Collection 2 images for unavailable Sentinel-2 views."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import time
import urllib.request
import math
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from .atlas import corners, view_transform
from .data import digest, request_json
from .pipeline import read_records, write_json
from .quality import POLICY, file_quality, image_quality
from .satellite import cached_window, nearest_place

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "landsat-c2-l2"
CREDIT = "Data available from the U.S. Geological Survey."
LICENSE = "https://www.usgs.gov/faqs/are-landsat-data-cloud-still-considered-be-within-public-domain"
BANDS = ("red", "green", "blue")


def request_crop(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return response.read()
        except HTTPError as exc:
            if attempt == 2 or (exc.code != 429 and exc.code < 500):
                raise
            time.sleep(min(30, max(2**attempt, float(exc.headers.get("Retry-After", "1")))))
        except (URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(2**attempt)


def read_bands(
    item, transform, size, assets, resampling, cache, collection=COLLECTION, dtype="uint16", expected_count=None
):
    """PC's official raw GeoTIFF crop avoids many transcontinental COG ranges."""
    from rasterio.io import MemoryFile
    from rasterio.warp import reproject

    shared = os.environ.get("COASTLINE_NATIVE_PATCH_CACHE")
    if shared:
        return read_native_crop(
            item, transform, size, assets, resampling, Path(shared), collection, dtype, expected_count
        )
    points = np.array(corners(transform, size))
    bounds = [*points.min(axis=0), *points.max(axis=0)]
    latitude = (transform @ (size[0] / 2, size[1] / 2))[1]
    coslat = np.cos(np.deg2rad(latitude))
    pixel_m = 111320 * np.hypot(transform.a * coslat, transform.d)
    width = max(1, min(1800, int(np.ceil((bounds[2] - bounds[0]) * 111320 * coslat / pixel_m))))
    height = max(1, min(1800, int(np.ceil((bounds[3] - bounds[1]) * 111320 / pixel_m))))
    params = [
        ("collection", collection),
        ("item", item["id"]),
        ("dst_crs", "EPSG:4326"),
        ("coord_crs", "EPSG:4326"),
        ("asset_as_band", "false" if expected_count and expected_count != len(assets) else "true"),
        ("unscale", "false"),
        ("return_mask", "false"),
        ("resampling", resampling.name),
        ("reproject", resampling.name),
    ]
    params += [("assets", band) for band in assets]
    url = (
        "https://planetarycomputer.microsoft.com/api/data/v1/item/bbox/"
        + ",".join(map(str, bounds))
        + f"/{width}x{height}.tif?"
        + urlencode(params)
    )
    key = hashlib.sha256(url.encode()).hexdigest()[:24]
    directory = cache / "crops"
    directory.mkdir(exist_ok=True)
    path = directory / f"{key}.tif"
    if path.exists():
        raw = path.read_bytes()
    else:
        raw = request_crop(url)
    count = expected_count or len(assets)
    dest = np.zeros((count, size[1], size[0]), dtype)
    with MemoryFile(raw) as memory, memory.open() as source:
        if source.count != count or any(t != dtype for t in source.dtypes) or not source.crs:
            raise ValueError(f"Crop API must return {count} georeferenced, unscaled {dtype} bands")
        data = source.read()
        reproject(
            source=data,
            destination=dest,
            src_transform=source.transform,
            src_crs=source.crs,
            dst_transform=transform,
            dst_crs="EPSG:4326",
            src_nodata=0,
            dst_nodata=0,
            resampling=resampling,
            num_threads=1,
        )
    if not path.exists():
        partial = path.with_suffix(".partial.tif")
        partial.write_bytes(raw)
        partial.replace(path)
    return dest


def read_native_crop(item, transform, size, assets, resampling, cache, collection, dtype, expected_count=None):
    """Cache a small snapped native-resolution bounding box shared by nearby views."""
    from rasterio.io import MemoryFile
    from rasterio.warp import reproject

    cache.mkdir(parents=True, exist_ok=True)
    count = expected_count or len(assets)
    points = np.asarray(corners(transform, size))
    low = points.min(axis=0)
    high = points.max(axis=0)
    lat = (transform @ (size[0] / 2, size[1] / 2))[1]
    requested = 111320 * np.hypot(transform.a * np.cos(np.deg2rad(lat)), transform.d)
    sensor = 30 if collection == COLLECTION else 20 if assets == ("SCL",) else 10
    resolution = sensor * 2 ** max(0, math.floor(math.log2(max(sensor, requested) / sensor)))
    coords = []

    def collect(values):
        if len(values) >= 2 and isinstance(values[0], (float, int)):
            coords.append(values[:2])
        else:
            for value in values:
                collect(value)

    if item.get("geometry", {}).get("coordinates"):
        collect(item["geometry"]["coordinates"])
        footprint = np.asarray(coords)
        low = np.maximum(low, footprint.min(axis=0))
        high = np.minimum(high, footprint.max(axis=0))
        if (low >= high).any():
            return np.zeros((count, size[1], size[0]), dtype)
    bandlat = math.floor(lat / 5) * 5 + 2.5
    dy = resolution * 512 / 111320
    dx = dy / max(0.02, np.cos(np.deg2rad(bandlat)))
    left = math.floor((low[0] + 180) / dx) * dx - 180
    right = math.ceil((high[0] + 180) / dx) * dx - 180
    bottom = max(-90.0, math.floor((low[1] + 90) / dy) * dy - 90)
    top = min(90.0, math.ceil((high[1] + 90) / dy) * dy - 90)
    if left < -180 or right > 180:
        raise ValueError("Antimeridian crop unsupported")
    width = max(1, math.ceil((right - left) * 111320 * np.cos(np.deg2rad((bottom + top) / 2)) / resolution))
    height = max(1, math.ceil((top - bottom) * 111320 / resolution))
    params = [
        ("collection", collection),
        ("item", item["id"]),
        ("dst_crs", "EPSG:4326"),
        ("coord_crs", "EPSG:4326"),
        ("asset_as_band", "false" if count != len(assets) else "true"),
        ("unscale", "false"),
        ("return_mask", "false"),
        ("resampling", resampling.name),
        ("reproject", resampling.name),
    ] + [("assets", a) for a in assets]
    url = (
        "https://planetarycomputer.microsoft.com/api/data/v1/item/bbox/"
        + ",".join(map(str, [left, bottom, right, top]))
        + f"/{width}x{height}.tif?"
        + urlencode(params)
    )
    key = hashlib.sha256(url.encode()).hexdigest()[:24]
    path = cache / f"{key}.tif"
    if path.is_file():
        raw = path.read_bytes()
        os.utime(path, None)
    else:
        raw = request_crop(url)
    dest = np.zeros((count, size[1], size[0]), dtype)
    with MemoryFile(raw) as memory, memory.open() as source:
        if source.count != count or any(t != dtype for t in source.dtypes) or not source.crs:
            raise ValueError("Native crop band/type/georeference mismatch")
        values = source.read()
        profile = source.profile.copy()
        reproject(
            source=values,
            destination=dest,
            src_transform=source.transform,
            src_crs=source.crs,
            dst_transform=transform,
            dst_crs="EPSG:4326",
            src_nodata=0,
            dst_nodata=0,
            resampling=resampling,
            num_threads=1,
        )
    if not path.is_file():
        profile.update(compress="deflate", predictor=2, zlevel=3, tiled=True, blockxsize=256, blockysize=256)
        with MemoryFile() as packed:
            with packed.open(**profile) as sink:
                sink.write(values)
            compressed = packed.read()
        temporary = path.with_name(path.name + f".{os.getpid()}.partial")
        temporary.write_bytes(compressed)
        temporary.replace(path)
    return dest



def natural_color(dn):
    # USGS C2 L2 scale/offset, followed by the same sRGB transfer for all scenes.
    linear = np.clip(np.asarray(dn, dtype=np.float32) * 0.0000275 - 0.2, 0, 1)
    srgb = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * np.power(linear, 1 / 2.4) - 0.055)
    return np.uint8(np.clip(np.rint(srgb * 255), 0, 255))


def qa_masks(qa):
    qa = np.asarray(qa, dtype=np.uint16)
    valid = (qa != 0) & ((qa & 1) == 0)
    clear = valid & ((qa & 0b111110) == 0)
    return valid, clear


def key_for(record, size=(960, 720)):
    geometry = [record[k] for k in ("longitude", "latitude", "width_km", "rotation_clockwise")]
    return hashlib.sha256(
        json.dumps(
            [STAC, COLLECTION, geometry, size, "c2-srgb-v1"] + (["mirror"] if record.get("mirror_horizontal") else [])
        ).encode()
    ).hexdigest()[:24]


def catalog(record, transform, size, cache, interval):
    points = np.array(corners(transform, size))
    bounds = [*points.min(axis=0), *points.max(axis=0)]
    snapped = [float(np.floor(v * 4) / 4) for v in bounds[:2]] + [float(np.ceil(v * 4) / 4) for v in bounds[2:]]
    payload = dict(
        collections=[COLLECTION],
        bbox=snapped,
        datetime=interval,
        limit=200,
        query={"platform": {"in": ["landsat-8", "landsat-9"]}, "eo:cloud_cover": {"lt": 60}},
        fields={
            "include": [
                "id",
                "collection",
                "geometry",
                "properties.datetime",
                "properties.platform",
                "properties.eo:cloud_cover",
                "assets.red",
                "assets.green",
                "assets.blue",
                "assets.qa_pixel",
                "links",
            ]
        },
    )
    identity = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]
    directory = cache / "catalogs"
    directory.mkdir(exist_ok=True)
    path = directory / f"{identity}.json"
    shared = os.environ.get("COASTLINE_LANDSAT_CATALOG_CACHE")
    shared_path = Path(shared) / path.name if shared else None
    for existing in [
        path,
        *([shared_path] if shared_path else []),
        *cache.parent.glob("process-*/catalogs/" + path.name),
    ]:
        try:
            if existing.is_file():
                return json.loads(existing.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    result = request_json(STAC + "/search", payload)
    features = result.get("features", [])
    # Retrieve enough years to include summer, even in densely overlapping polar tracks.
    for _ in range(3):
        link = next((l for l in result.get("links", []) if l.get("rel") == "next"), None)
        if link is None:
            break
        if not link["href"].startswith(STAC + "/"):
            raise ValueError("Unexpected Landsat STAC pagination host")
        result = request_json(link["href"], link.get("body") if link.get("method", "GET") == "POST" else None)
        features += result.get("features", [])
    compact = [
        dict(
            id=i["id"],
            collection=i["collection"],
            geometry=i["geometry"],
            properties={
                k: i["properties"][k] for k in ("datetime", "platform", "eo:cloud_cover") if k in i["properties"]
            },
            assets={k: {"href": i["assets"][k]["href"]} for k in (*BANDS, "qa_pixel")},
        )
        for i in features
    ]
    partial = path.with_suffix(".partial.json")
    partial.write_text(json.dumps(compact), encoding="utf-8")
    partial.replace(path)
    if shared_path:
        shared_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = shared_path.with_name(shared_path.name + f".{os.getpid()}.partial")
        temporary.write_text(json.dumps(compact), encoding="utf-8")
        temporary.replace(shared_path)
    return compact


def fetch_window(
    record, cache, size=(960, 720), max_scenes=24, interval="2019-01-01T00:00:00Z/2021-12-31T23:59:59Z", background=None
):
    from rasterio.enums import Resampling
    from rasterio.features import rasterize

    cache.mkdir(parents=True, exist_ok=True)
    key = key_for(record, size)
    if background:
        key = hashlib.sha256(
            (key + background["image_sha256"] + background["validity_mask_sha256"]).encode()
        ).hexdigest()[:24]
    path = cache / f"{key}.png"
    metadata_path = cache / f"{key}.json"
    cached = cached_window(path, metadata_path)
    if cached and file_quality(path)["acceptable"]:
        return cached
    for shared in [cache.parent / "probe", *cache.parent.glob("process-*")]:
        if shared == cache:
            continue
        other = cached_window(shared / path.name, shared / metadata_path.name)
        if other and file_quality(other["path"])["acceptable"]:
            return other
    transform = view_transform(
        record["longitude"],
        record["latitude"],
        record["width_km"],
        record["rotation_clockwise"],
        size,
        mirror_horizontal=bool(record.get("mirror_horizontal")),
    )
    small = view_transform(
        record["longitude"],
        record["latitude"],
        record["width_km"],
        record["rotation_clockwise"],
        (32, 24),
        mirror_horizontal=bool(record.get("mirror_horizontal")),
    )
    items = catalog(record, transform, size, cache, interval)

    def preference(item):
        month = int(item["properties"]["datetime"][5:7])
        best, next_month = (8, 7) if record["latitude"] >= 0 else (2, 1)
        season = 0 if abs(record["latitude"]) < 40 or month == best else 1 if month == next_month else 2
        return (
            season,
            abs(int(item["properties"]["datetime"][:4]) - 2020),
            item["properties"].get("eo:cloud_cover", 100),
        )

    items = sorted(items, key=preference)
    footprints = [
        rasterize([(i["geometry"], 1)], out_shape=(24, 32), transform=small, dtype="uint8").astype(bool) for i in items
    ]
    seed_valid = np.zeros((size[1], size[0]), bool)
    seed_rgb = np.zeros((3, size[1], size[0]), np.uint8)
    seed_cloud = np.zeros_like(seed_valid)
    seed_snow = np.zeros_like(seed_valid)
    if background:
        if digest(Path(background["path"])) != background["image_sha256"]:
            raise ValueError("Partial source photo changed")
        with Image.open(background["path"]) as source:
            seed_rgb = np.asarray(source.convert("RGB")).transpose(2, 0, 1).copy()
        for name in ["validity", "cloud", "snow"]:
            seed_path = Path(background[name + "_mask_path"])
            if digest(seed_path) != background[name + "_mask_sha256"]:
                raise ValueError("Partial source pixel mask changed")
            with Image.open(seed_path) as source:
                values = np.asarray(source.convert("L")) > 0
            if name == "validity":
                seed_valid = values
            elif name == "cloud":
                seed_cloud = values
            else:
                seed_snow = values
    remaining = list(range(len(items)))
    needed = (
        np.asarray(Image.fromarray(np.uint8(~seed_valid) * 255).resize((32, 24), Image.Resampling.BOX)) > 0
        if background
        else np.ones((24, 32), bool)
    )
    # Very small gaps may round out of a coarse image. Their exact final coverage
    # is still required, and all scenes remain available after the greedy order.
    order = []
    while remaining and needed.any():
        best = max(remaining, key=lambda j: int((footprints[j] & needed).sum()))
        if not (footprints[best] & needed).any():
            break
        order.append(best)
        remaining.remove(best)
        needed &= ~footprints[best]
    if needed.mean() > 0.005:
        write_json(
            cache / f"{key}.failure.json",
            dict(
                interval=interval,
                reason="STAC footprints leave source pixels unavailable",
                geometric_uncovered_fraction=float(needed.mean()),
                geometry=[record[k] for k in ("longitude", "latitude", "width_km", "rotation_clockwise")],
            ),
        )
        raise RuntimeError("Landsat catalog cannot cover this entire real view")
    max_scenes = max(max_scenes, min(64, len(order) + 4))
    ordered = [items[i] for i in order + remaining]
    covered = seed_valid.copy()
    rgb = seed_rgb.copy()
    fallback = np.zeros_like(rgb)
    fallback_source = np.zeros(covered.shape, np.uint16)
    fallback_snow = np.zeros(covered.shape, bool)
    scenes = []
    errors = []
    for scene_index, item in enumerate(ordered[:max_scenes]):
        try:
            if scene_index % 4 == 0:
                print(f"Landsat frame {record.get('frame','?')}: scene {scene_index+1}, {item['id']}", flush=True)
            qa = read_bands(item, transform, size, ("qa_pixel",), Resampling.nearest, cache)[0]
            valid, clear = qa_masks(qa)
            usable = clear & ~covered
            if not usable.any() and not (valid & (fallback_source == 0) & ~seed_valid).any():
                continue
            dn = read_bands(item, transform, size, BANDS, Resampling.bilinear, cache)
            valid &= (dn > 0).all(axis=0)
            usable &= valid
            image = natural_color(dn)
            if not background and not image_quality(image.transpose(1, 2, 0))["acceptable"]:
                continue
            rgb[:, usable] = image[:, usable]
            covered |= usable
            use_fallback = valid & (fallback_source == 0) & ~seed_valid
            fallback[:, use_fallback] = image[:, use_fallback]
            fallback_source[use_fallback] = len(scenes) + 1
            fallback_snow[use_fallback] = (qa[use_fallback] & (1 << 5)) != 0
            scenes.append(
                dict(
                    id=item["id"],
                    collection=COLLECTION,
                    datetime=item["properties"]["datetime"],
                    platform=item["properties"].get("platform"),
                    eo_cloud_cover=item["properties"].get("eo:cloud_cover"),
                    asset_hrefs={k: item["assets"][k]["href"] for k in (*BANDS, "qa_pixel")},
                    stac_url=STAC + "/collections/" + COLLECTION + "/items/" + item["id"],
                    contributed_clear_pixels=int(usable.sum()),
                )
            )
            if covered.all() or (covered.mean() >= 0.995 and (covered | (fallback_source > 0)).all()):
                break
        except Exception as exc:
            errors.append(dict(id=item["id"], error_type=type(exc).__name__))
    obscured = ~covered & (fallback_source > 0)
    snow = obscured & fallback_snow
    cloud = obscured & ~fallback_snow
    clear_coverage = float(covered.mean()) - float(seed_cloud.mean()) - float(seed_snow.mean())
    rgb[:, obscured] = fallback[:, obscured]
    covered |= obscured
    quality = image_quality(rgb.transpose(1, 2, 0))
    if not covered.all() or not quality["acceptable"]:
        write_json(
            cache / f"{key}.failure.json",
            dict(
                interval=interval, coverage=float(covered.mean()), visible_quality=quality, scenes=scenes, errors=errors
            ),
        )
        raise RuntimeError("No complete, usable Landsat view")
    for i, s in enumerate(scenes):
        s.update(
            contributed_cloud_pixels=int((cloud & (fallback_source == i + 1)).sum()),
            contributed_snow_pixels=int((snow & (fallback_source == i + 1)).sum()),
        )
    Image.fromarray(rgb.transpose(1, 2, 0)).save(path)
    metadata = dict(
        status="complete",
        path=str(path.resolve()),
        key=key,
        size=list(size),
        interval=interval,
        coverage=1.0,
        clear_coverage=clear_coverage,
        cloud_fraction=float(cloud.mean()),
        snow_fraction=float(snow.mean()),
        scenes=(background["scenes"] if background else []) + scenes,
        credit=CREDIT,
        provider="USGS Landsat / Microsoft Planetary Computer",
        stac_api=STAC,
        license_url=LICENSE,
        image_sha256=digest(path),
        quality_policy=POLICY,
        visible_quality=quality,
        source_resolution_m=30,
        accessed_utc=datetime.now(timezone.utc).isoformat(),
        data_api="https://planetarycomputer.microsoft.com/api/data/v1",
        footprint_lonlat=corners(transform, size),
        processing="Landsat 8/9 C2 L2 RGB bands; raw unscaled uint16 GeoTIFF crops via official PC data API, preserving georeferencing; DN*0.0000275-0.2; fixed sRGB transfer; QA cloud/snow screening; same-location temporal mosaic; rigid crop/rotation/resampling",
    )
    write_json(metadata_path, metadata)
    if background:
        metadata.update(
            mixed_sensors=True,
            sentinel_background_sha256=background["image_sha256"],
            sentinel_pixel_fraction=float(seed_valid.mean()),
            cloud_fraction=float((cloud | seed_cloud).mean()),
            snow_fraction=float((snow | seed_snow).mean()),
            processing=metadata["processing"]
            + "; only missing source pixels filled with georeferenced real Landsat pixels over the same Sentinel-2 view",
        )
        write_json(metadata_path, metadata)
    return metadata


def worker(record, cache, max_scenes, interval, candidates=None):
    private = Path(cache) / f"process-{os.getpid()}"
    fields = ("longitude", "latitude", "width_km", "rotation_clockwise")
    failure = "RuntimeError"
    for rank, candidate in enumerate([record] if candidates is None else candidates):
        if candidate.get("native_iou", 0) < record.get("baseline_native_iou", 0) - 1e-7:
            continue
        view = dict(record, **candidate)
        try:
            image = fetch_window(view, private, max_scenes=max_scenes, interval=interval)
        except Exception as exc:
            failure = type(exc).__name__
            continue
        result = dict(view, satellite=image, landsat_replacement=True)
        result.pop("alternatives", None)
        if any(candidate[k] != record[k] for k in fields):
            result.update(
                landsat_alternative=True,
                landsat_alternative_rank=rank + 1,
                landsat_original_geometry={k: record[k] for k in fields},
                place=nearest_place(Path("assets/coastline/places-10m.geojson"), view["longitude"], view["latitude"]),
            )
            if "native_mask_path" not in candidate:
                result.pop("native_mask_path", None)
                result.pop("native_mask_sha256", None)
            result.pop("native_pixels_verified", None)
            result["native_iou_gain"] = result["native_iou"] - result["baseline_native_iou"]
        return result
    return dict(record, landsat_attempt_failed=True, landsat_failure_type=failure)


def apply_image_result(record, result):
    if not result.get("landsat_replacement"):
        return dict(record, landsat_attempt_failed=True, landsat_failure_type=result["landsat_failure_type"])
    if result.get("landsat_alternative"):
        if result["frame"] != record["frame"]:
            raise ValueError("Alternative matching scores belong to a different frame")
        return dict(result)
    # Shared photographs do not share the query's timestamp or matching scores.
    return dict(record, satellite=result["satellite"], landsat_replacement=True)


def replace(
    source,
    output,
    cache,
    workers=8,
    max_scenes=24,
    interval="2019-01-01T00:00:00Z/2021-12-31T23:59:59Z",
    alternatives_only=False,
):
    if source.resolve() == output.resolve():
        raise ValueError("Preserve the source movie in a separate directory")
    output.mkdir(parents=True, exist_ok=True)
    records = read_records(source / "frames.jsonl")
    requested = [r for r in records if r["satellite"].get("placeholder")]
    previous = (
        {r["frame"]: r for r in read_records(output / "landsat.partial.jsonl")}
        if (output / "landsat.partial.jsonl").exists()
        else {}
    )
    previous = {
        i: r
        for i, r in previous.items()
        if r.get("landsat_replacement")
        and Path(r["satellite"]["path"]).is_file()
        and digest(Path(r["satellite"]["path"])) == r["satellite"]["image_sha256"]
    }
    pending = [r for r in requested if not previous.get(r["frame"], {}).get("landsat_replacement")]
    os.environ.update(
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", GDAL_NUM_THREADS="1", GDAL_CACHEMAX="128", RAYON_NUM_THREADS="1"
    )
    groups = {}
    for record in pending:
        groups.setdefault(record["frame"] if alternatives_only else key_for(record), []).append(record)
    alternatives = (
        {r["frame"]: r.get("alternatives", []) for r in read_records(source / "matches.jsonl")}
        if alternatives_only
        else {}
    )
    with (
        ProcessPoolExecutor(max_workers=workers) as pool,
        (output / "landsat.partial.jsonl").open("a", encoding="utf-8") as sink,
    ):
        futures = {
            pool.submit(
                worker,
                group[0],
                str(cache.resolve()),
                max_scenes,
                interval,
                alternatives.get(group[0]["frame"], []) if alternatives_only else None,
            ): group
            for group in groups.values()
        }
        for index, future in enumerate(as_completed(futures)):
            result = future.result()
            for record in futures[future]:
                updated = apply_image_result(record, result)
                previous[record["frame"]] = updated
                sink.write(json.dumps(updated, ensure_ascii=False) + "\n")
            sink.flush()
            print(
                f"Landsat {index+1}/{len(groups)} views: frame {result['frame']}, {'OK' if result.get('landsat_replacement') else 'unavailable'}",
                flush=True,
            )
    merged = [previous.get(r["frame"], r) for r in records]
    (output / "frames.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in merged), encoding="utf-8"
    )
    for filename in ["run.json", "matches.jsonl", "CREDITS.txt", "imagery-status.json", "repair-report.json"]:
        shutil.copyfile(source / filename, output / filename)
    settings = json.loads((output / "run.json").read_text(encoding="utf-8"))
    settings.update(
        landsat_replacements_from=str(source.resolve()),
        landsat_stac=STAC,
        landsat_last_requested_interval=interval,
        landsat_max_scenes=max_scenes,
    )
    write_json(output / "run.json", settings)
    report = dict(
        requested=len(requested),
        replaced=sum(bool(r.get("landsat_replacement")) for r in merged),
        remaining_black=sum(bool(r["satellite"].get("placeholder")) for r in merged),
        alternative_location_frames=[r["frame"] for r in merged if r.get("landsat_alternative")],
    )
    write_json(output / "landsat-report.json", report)
    status = json.loads((output / "imagery-status.json").read_text(encoding="utf-8"))
    status.update(black_placeholder_frames=report["remaining_black"], landsat_replacement_frames=report["replaced"])
    write_json(output / "imagery-status.json", status)
    credits = (output / "CREDITS.txt").read_text(encoding="utf-8")
    credits = re.sub(
        r"Unretrievable views: \d+ black placeholders",
        f"Unretrievable views: {report['remaining_black']} black placeholders",
        credits,
    )
    (output / "CREDITS.txt").write_text(credits, encoding="utf-8")
    with (output / "CREDITS.txt").open("a", encoding="utf-8") as sink:
        sink.write(
            "\nLandsat replacements: "
            + CREDIT
            + "\n"
            + LICENSE
            + "\nCollection 2 Level-2, via Microsoft Planetary Computer. Natural RGB, physical reflectance scale/offset and fixed sRGB transfer. Exact scenes/dates/assets: frames.jsonl.\n"
        )
        sink.write(
            f"Landsat replaced {report['replaced']} unavailable Sentinel-2 frames; remaining black placeholders: {report['remaining_black']}.\n"
        )
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("output/coastline-wide-rust-fixed"))
    parser.add_argument("--output", type=Path, default=Path("output/coastline-wide-rust-landsat"))
    parser.add_argument("--cache", type=Path, default=Path("cache/coastline-landsat"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-scenes", type=int, default=24)
    parser.add_argument("--interval", default="2019-01-01T00:00:00Z/2021-12-31T23:59:59Z")
    parser.add_argument(
        "--alternatives-only",
        action="store_true",
        help="Retry failed frames at saved candidate locations, retaining the original native IoU floor",
    )
    replace(**vars(parser.parse_args()))
