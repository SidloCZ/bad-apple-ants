"""Real full-aspect imagery with native-source crops and isolated GDAL workers."""

from __future__ import annotations
import argparse, hashlib, json, os, shutil, time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
import numpy as np
from PIL import Image
from .atlas import corners, view_transform
from .cuda_search import OUTPUT, KEYS
from .data import digest
from .pipeline import read_records, write_json
from .quality import image_quality, file_quality, POLICY
from .satellite import nearest_place, project_asset, catalog as earth_catalog, resolve_interval, STAC as S2_STAC

SIZE = (1444, 1080)
COLLECTION = "sentinel-2-l2a"


def residual_is_safe(covered, snow, water, critical):
    residual = ~covered
    return not np.any(residual & ((snow & water) | (~snow & critical)))


def coast_regions(record):
    from scipy.ndimage import maximum_filter

    path = Path(record.get("native_mask_path", "__missing__"))
    if not path.is_file() or digest(path) != record.get("native_mask_sha256"):
        return None
    with Image.open(path) as image:
        land = np.asarray(image.convert("L")) >= 128
    if land.shape != (SIZE[1], SIZE[0]):
        return None
    edge = np.zeros_like(land)
    diff = land[:, 1:] != land[:, :-1]
    edge[:, 1:] |= diff
    edge[:, :-1] |= diff
    diff = land[1:] != land[:-1]
    edge[1:] |= diff
    edge[:-1] |= diff
    radius = max(4, min(32, int(np.ceil(60 / (record["width_km"] * 1000 / SIZE[0])))))
    water = ~land
    critical = water | maximum_filter(edge, size=2 * radius + 1)
    return water, critical



def sentinel(record, cache, max_scenes=12, preferred_dates=()):
    from rasterio.enums import Resampling
    from rasterio.features import rasterize

    if not -56 < record["latitude"] < 84:
        raise RuntimeError("Outside Sentinel-2 land coverage")
    geo = [record[k] for k in KEYS]
    interval = resolve_interval(record)
    mirrored = bool(record.get("mirror_horizontal"))
    key = hashlib.sha256(
        json.dumps([geo, SIZE, interval, "EarthSearch-native-s2-v1"] + (["mirror"] if mirrored else [])).encode()
    ).hexdigest()[:24]
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{key}.png"
    metadata = cache / f"{key}.json"
    directories = [cache]
    if record.get("constant_content"):
        directories.extend(p for p in sorted(cache.parent.glob("process-*")) if p != cache)
    for directory in directories:
        saved = directory / f"{key}.png"
        saved_metadata = directory / f"{key}.json"
        if not saved.is_file() or not saved_metadata.is_file():
            continue
        try:
            result = json.loads(saved_metadata.read_text())
            if digest(saved) == result["image_sha256"] and file_quality(saved)["acceptable"]:
                return result
        except (OSError, ValueError, KeyError):
            continue
    transform = view_transform(*geo, SIZE, mirror_horizontal=mirrored)
    points = np.array(corners(transform, SIZE))
    bounds = [*points.min(axis=0), *points.max(axis=0)]
    if bounds[0] < -180 or bounds[2] > 180:
        raise ValueError("Antimeridian imagery unsupported")
    items = earth_catalog(
        bounds, interval, 30, cache, viewport=view_transform(*geo, (32, 24), mirror_horizontal=mirrored)
    )["features"]
    for item in items:
        if "SCL" not in item["assets"] and "scl" in item["assets"]:
            item["assets"]["SCL"] = item["assets"]["scl"]

    def preference(item):
        month = int(item["properties"]["datetime"][5:7])
        lat = record["latitude"]
        season = 0 if abs(lat) < 40 or month in ([7, 8] if lat >= 0 else [1, 2]) else 1
        return (
            (item["properties"]["datetime"][:10] not in preferred_dates),
            season,
            item["properties"].get("eo:cloud_cover", 100),
        )

    items.sort(key=preference)
    # Preserve the true source aspect in footprint checks as well.
    coarse = view_transform(*geo, (361, 270), mirror_horizontal=mirrored)
    footprints = [
        rasterize([(i["geometry"], 1)], out_shape=(270, 361), transform=coarse, dtype="uint8").astype(bool)
        for i in items
    ]
    remaining = list(range(len(items)))
    needed = np.ones((270, 361), bool)
    order = []
    while remaining and needed.any():
        index = max(remaining, key=lambda j: int((footprints[j] & needed).sum()))
        if not (footprints[index] & needed).any():
            break
        order.append(index)
        remaining.remove(index)
        needed &= ~footprints[index]
    if needed.mean() > 0.05:
        raise RuntimeError("Sentinel-2 footprints cannot supply the minimum real source coverage")
    max_scenes = max(max_scenes, min(64, len(order) + 4))
    ordered = [items[i] for i in order + remaining]
    rgb = np.zeros((3, SIZE[1], SIZE[0]), np.uint8)
    covered = np.zeros((SIZE[1], SIZE[0]), bool)
    fallback = np.zeros_like(rgb)
    sources = np.zeros(covered.shape, np.uint16)
    fallback_snow = np.zeros(covered.shape, bool)
    regions = coast_regions(record)
    scenes = []
    errors = []
    for item in ordered[:max_scenes]:
        try:
            scl = project_asset(item["assets"]["SCL"], transform, SIZE, [1], Resampling.nearest, force_native=True)[0]
            valid = (scl >= 2) & (scl <= 11)
            clear = valid & ~np.isin(scl, [8, 9, 10, 11])
            usable = clear & ~covered
            available = valid & (sources == 0)
            if not usable.any() and not available.any():
                continue
            pixels = project_asset(
                item["assets"]["visual"], transform, SIZE, [1, 2, 3], Resampling.bilinear, force_native=True
            )
            if not image_quality(pixels.transpose(1, 2, 0))["acceptable"]:
                continue
            rgb[:, usable] = pixels[:, usable]
            covered |= usable
            fallback[:, available] = pixels[:, available]
            sources[available] = len(scenes) + 1
            fallback_snow[available] = scl[available] == 11
            scenes.append(
                dict(
                    id=item["id"],
                    collection=COLLECTION,
                    datetime=item["properties"]["datetime"],
                    eo_cloud_cover=item["properties"].get("eo:cloud_cover"),
                    asset_hrefs={k: item["assets"][k]["href"] for k in ["visual", "SCL"]},
                    stac_url=S2_STAC + "/collections/" + COLLECTION + "/items/" + item["id"],
                    contributed_clear_pixels=int(usable.sum()),
                )
            )
            if covered.all() or (covered.mean() >= 0.995 and (covered | (sources > 0)).all()):
                break
            # Snow on actual land still reveals the coastline. Avoid downloading
            # extra dates solely to remove it, or clouds far inside land, when
            # every real source pixel is present and all water/shore remains clear.
            if (
                regions is not None
                and (covered | (sources > 0)).all()
                and residual_is_safe(covered, fallback_snow, *regions)
            ):
                merged = rgb.copy()
                missing = ~covered
                merged[:, missing] = fallback[:, missing]
                if image_quality(merged.transpose(1, 2, 0))["acceptable"]:
                    break
        except Exception as exc:
            errors.append(dict(id=item["id"], error_type=type(exc).__name__))
    obscured = ~covered & (sources > 0)
    snow = obscured & fallback_snow
    cloud = obscured & ~fallback_snow
    rgb[:, obscured] = fallback[:, obscured]
    quality = image_quality(rgb.transpose(1, 2, 0))
    coverage = float((covered | obscured).mean())
    if coverage < 0.95 or not quality["acceptable"]:
        write_json(
            cache / f"{key}.failure.json",
            dict(coverage=float((covered | obscured).mean()), errors=errors, quality=quality),
        )
        raise RuntimeError("Incomplete or whiteout Sentinel-2 view")
    for i, s in enumerate(scenes):
        s.update(
            contributed_cloud_pixels=int((cloud & (sources == i + 1)).sum()),
            contributed_snow_pixels=int((snow & (sources == i + 1)).sum()),
        )
    Image.fromarray(rgb.transpose(1, 2, 0)).save(path)
    result = dict(
        path=str(path.resolve()),
        size=list(SIZE),
        status="complete" if coverage == 1 else "partial-real-source",
        coverage=coverage,
        clear_coverage=float(covered.mean()),
        cloud_fraction=float(cloud.mean()),
        snow_fraction=float(snow.mean()),
        scenes=scenes,
        interval=interval,
        image_sha256=digest(path),
        quality_policy=POLICY,
        visible_quality=quality,
        provider="Copernicus Sentinel-2 / Earth Search / AWS Open Data",
        stac_api=S2_STAC,
        credit="Contains modified Copernicus Sentinel data 2020/2021",
        license_url="https://cds.climate.copernicus.eu/licences/ec-sentinel",
        source_resolution_m=10,
        footprint_lonlat=corners(transform, SIZE),
        accessed_utc=datetime.now(timezone.utc).isoformat(),
        processing="Original Sentinel-2 L2A visual and SCL COG assets read directly from AWS; native-window/appropriate overview sampling finer than the output grid; no downsampled whole-scene cache; rigid crop/rotation/resampling; SCL cloud/snow-screened same-location temporal mosaic; remaining clouds quantified",
    )
    if coverage < 1:
        for name, values in [("validity", covered | obscured), ("cloud", cloud), ("snow", snow)]:
            mask_path = cache / f"{key}.{name}.png"
            Image.fromarray(np.uint8(values) * 255).save(mask_path)
            result[name + "_mask_path"] = str(mask_path.resolve())
            result[name + "_mask_sha256"] = digest(mask_path)
    write_json(metadata, result)
    return result


def baseline_crop(record, parent, cache):
    geo = [record[k] for k in KEYS]
    original = [parent[k] for k in KEYS]
    satellite = parent["satellite"]
    source = Path(satellite["path"])
    if digest(source) != satellite["image_sha256"]:
        raise ValueError("Baseline real photograph changed")
    with Image.open(source) as image:
        mirrored = bool(record.get("mirror_horizontal"))
        old_mirrored = bool(parent.get("mirror_horizontal"))
        image = image.convert("RGB")
        old_transform = view_transform(*original, image.size, mirror_horizontal=old_mirrored)
        new_transform = view_transform(*geo, SIZE, mirror_horizontal=mirrored)
        if geo == original and image.size == SIZE and mirrored == old_mirrored:
            if not file_quality(source)["acceptable"]:
                raise ValueError("Existing photograph is whiteout")
            return dict(satellite)
        mapping = (~old_transform) @ new_transform
        points = np.array([mapping @ p for p in [(0, 0), (SIZE[0], 0), SIZE, (0, SIZE[1])]])
        # Inverting geographic affines amplifies sub-nanodegree roundoff.
        # Pixel centers remain inside for this <1/100000-pixel edge allowance.
        edge_tolerance = 1e-5
        if (
            (points[:, 0] < -edge_tolerance).any()
            or (points[:, 0] > image.width + edge_tolerance).any()
            or (points[:, 1] < -edge_tolerance).any()
            or (points[:, 1] > image.height + edge_tolerance).any()
        ):
            raise ValueError("Baseline crop would require pixels outside its real image")
        key = hashlib.sha256(
            json.dumps([satellite["image_sha256"], geo, SIZE, mirrored, old_mirrored, "rigid-existing-v2"]).encode()
        ).hexdigest()[:24]
        path = cache / f"{key}.png"
        cache.mkdir(parents=True, exist_ok=True)
        if not path.is_file():
            image.transform(SIZE, Image.Transform.AFFINE, tuple(mapping)[:6], resample=Image.Resampling.BICUBIC).save(
                path
            )
    quality = file_quality(path)
    if not quality["acceptable"]:
        raise ValueError("Baseline crop is whiteout")
    return dict(
        satellite,
        path=str(path.resolve()),
        size=list(SIZE),
        image_sha256=digest(path),
        visible_quality=quality,
        footprint_lonlat=corners(new_transform, SIZE),
        baseline_photo_sha256=satellite["image_sha256"],
        baseline_photo_path=str(source.resolve()),
        processing=satellite.get("processing", "")
        + "; rigid geographic subwindow reprojected from this actual prior photograph to original source aspect; no fill",
        parent_cloud_metrics=True,
    )


@lru_cache(maxsize=1)
def baseline_records(path):
    return read_records(Path(path) / "frames.jsonl")


def worker(record, root, baseline_path):
    os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", GDAL_NUM_THREADS="1", GDAL_CACHEMAX="128")
    root = Path(root)
    cache = root / f"process-{os.getpid()}"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["COASTLINE_NATIVE_PATCH_CACHE"] = str(root / "native-crops")
    shared_catalog = root / "earthsearch-catalogs"
    os.environ["COASTLINE_CATALOG_READ_CACHE"] = os.pathsep.join(
        map(
            str,
            [
                shared_catalog,
                (Path(baseline_path) / "imagery/earthsearch-catalogs").resolve(),
                Path("cache/coastline-satellite/catalogs").resolve(),
            ],
        )
    )
    os.environ["COASTLINE_CATALOG_WRITE_CACHE"] = str(shared_catalog)
    os.environ["COASTLINE_LANDSAT_CATALOG_CACHE"] = str(root / "landsat-catalogs")
    baseline = baseline_records(baseline_path)[record["frame"]]
    options = [{k: v for k, v in record.items() if k != "alternatives"}] + record.get("alternatives", [])
    errors = []
    for rank, candidate in enumerate(options):
        # Each candidate already has independent full-resolution scores for this frame.
        if candidate["native_objective"] < record["baseline_fullres_metrics"]["native_objective"] - 1e-9:
            continue
        if "precision_objective" in record:
            base = record["baseline_fullres_metrics"]
            if (
                candidate["native_miou"] < base["native_miou"] - 1e-12
                or candidate["precision_objective"] < base["precision_objective"] - 1e-12
            ):
                continue
            if (
                "structure_objective" in record
                and candidate["structure_objective"] < base["structure_objective"] - 1e-12
            ):
                continue
        view = dict(
            {
                k: v
                for k, v in record.items()
                if k not in ("alternatives", "baseline_photo_frame", "baseline_crop_frame")
            },
            **candidate,
        )
        view.pop("alternatives", None)
        if any(view[k] != record[k] for k in KEYS) or bool(view.get("mirror_horizontal")) != bool(
            record.get("mirror_horizontal")
        ):
            view.pop("native_mask_path", None)
            view.pop("native_mask_sha256", None)
        try:
            satellite = None
            if "baseline_photo_frame" in candidate or "baseline_crop_frame" in candidate:
                try:
                    photo_parent = baseline_records(baseline_path)[
                        int(
                            candidate.get("baseline_photo_frame", candidate.get("baseline_crop_frame", record["frame"]))
                        )
                    ]
                    satellite = baseline_crop(view, photo_parent, cache)
                    landsat = bool(photo_parent.get("landsat_replacement"))
                except Exception as exc:
                    errors.append(dict(rank=rank, source="baseline reuse", error_type=type(exc).__name__))
            if satellite is None:
                view.pop("baseline_photo_frame", None)
                view.pop("baseline_crop_frame", None)
                distance = np.hypot(
                    (view["longitude"] - baseline["longitude"]) * 111.32 * np.cos(np.deg2rad(view["latitude"])),
                    (view["latitude"] - baseline["latitude"]) * 111.32,
                )
                preferred = (
                    {s["datetime"][:10] for s in baseline["satellite"]["scenes"]}
                    if distance < max(view["width_km"], baseline["width_km"])
                    else set()
                )
                try:
                    satellite = sentinel(view, cache, preferred_dates=preferred)
                    landsat = False
                    if satellite["coverage"] < 1:
                        from .landsat import fetch_window

                        satellite = fetch_window(
                            view, cache / "landsat", size=SIZE, max_scenes=12, background=satellite
                        )
                        landsat = True
                except Exception as exc:
                    errors.append(dict(rank=rank, source="Sentinel-2", error_type=type(exc).__name__))
                    from .landsat import fetch_window

                    satellite = fetch_window(view, cache / "landsat", size=SIZE, max_scenes=12)
                    landsat = True
            view.update(
                satellite=satellite,
                landsat_replacement=landsat,
                imagery_candidate_rank=rank,
                native_objective_gain=view["native_objective"] - record["baseline_fullres_metrics"]["native_objective"],
                place=nearest_place(Path("assets/coastline/places-10m.geojson"), view["longitude"], view["latitude"]),
                imagery_errors=errors,
            )
            if "precision_objective" in view:
                view["precision_objective_gain"] = (
                    view["precision_objective"] - record["baseline_fullres_metrics"]["precision_objective"]
                )
            if "structure_objective" in view:
                view["structure_objective_gain"] = (
                    view["structure_objective"] - record["baseline_fullres_metrics"]["structure_objective"]
                )
            return view
        except Exception as exc:
            errors.append(dict(rank=rank, source="real imagery", error_type=type(exc).__name__))
    raise RuntimeError(f'No valid real photo including baseline for frame {record["frame"]}: {errors}')


def retrieve(output=OUTPUT, workers=16, follow=False):
    output = Path(output)
    root = (output / "imagery").resolve()
    root.mkdir(parents=True, exist_ok=True)
    settings = json.loads((output / "run.json").read_text())
    partial = output / "frames.partial.jsonl"
    # Exact request-keyed metadata can be reused across editions. Preserve
    # source catalogs and any newer destination entries; copy no raw imagery.
    for name in ["earthsearch-catalogs", "landsat-catalogs"]:
        source = Path(settings["baseline"]) / "imagery" / name
        destination = root / name
        destination.mkdir(exist_ok=True)
        for path in source.glob("*.json"):
            target = destination / path.name
            if not target.is_file():
                shutil.copy2(path, target)
    known = {r["frame"]: r for r in read_records(partial)} if partial.is_file() else {}
    # Warm identical new constant views serially before parallel workers read
    # their shared PNG/metadata. Each picture still comes from real sensors.
    if settings.get("structure_objective") and (output / "matches.jsonl").is_file():
        warmed = set()
        with partial.open("a", encoding="utf-8") as sink:
            for record in read_records(output / "matches.jsonl"):
                if not record.get("constant_content") or record["frame"] in known:
                    continue
                key = tuple(record[k] for k in KEYS) + (bool(record.get("mirror_horizontal")),)
                if key in warmed:
                    continue
                result = worker(record, str(root), settings["baseline"])
                warmed.add(key)
                known[result["frame"]] = result
                sink.write(json.dumps(result, ensure_ascii=False) + "\n")
                sink.flush()
                print(f'constant real-view cache warmed: frame{record["frame"]+1}', flush=True)
    # Keep each completed record durable; final ordering is restored after all workers finish.
    with ProcessPoolExecutor(max_workers=workers) as pool, partial.open("a", encoding="utf-8") as sink:
        pending = {}
        submitted = set(known)
        started = time.perf_counter()
        last_trim = 0
        while True:
            if time.perf_counter() - last_trim > 60:
                trim_cache(root)
                last_trim = time.perf_counter()
            complete = (output / "matches.jsonl").is_file()
            matches = read_records(output / ("matches.jsonl" if complete else "matches.partial.jsonl"))
            available = [r for r in matches if r["frame"] not in submitted]
            stopping = (output / ".imagery-stop").is_file()
            for record in available[: 0 if stopping else max(0, workers * 3 - len(pending))]:
                future = pool.submit(worker, record, str(root), settings["baseline"])
                pending[future] = record["frame"]
                submitted.add(record["frame"])
            if pending:
                finished, _ = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
                for future in finished:
                    frame = pending.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        write_json(output / f"imagery-failed-{frame:05d}.json", dict(frame=frame, error=str(exc)))
                        raise
                    known[frame] = result
                    sink.write(json.dumps(result, ensure_ascii=False) + "\n")
                    sink.flush()
                if finished:
                    print(
                        f'full-aspect real imagery: {len(known)}/{settings["frames"]}; {time.perf_counter()-started:.1f}s',
                        flush=True,
                    )
            elif stopping or (complete and len(known) == len(matches)):
                break
            elif not follow:
                break
            else:
                time.sleep(5)
    if len(known) == settings["frames"]:
        ordered = [known[i] for i in range(settings["frames"])]
        (output / "frames.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ordered), encoding="utf-8"
        )
        write_json(
            output / "imagery-status.json",
            dict(
                frames=len(known),
                complete=len(known),
                black_placeholder_frames=0,
                landsat_frames=sum(r.get("landsat_replacement", False) for r in ordered),
                alternative_frames=sum(r["imagery_candidate_rank"] > 0 for r in ordered),
            ),
        )
        credits = (
            (Path(settings["baseline"]) / "CREDITS.txt").read_text(encoding="utf-8").split("\nWhiteout repair:")[0]
        )
        credits += "\nNew full-resolution CUDA edition: original 1444x1080 silhouettes; original 0.3 arcsecond mask; exact final binary measurements.\n"
        credits += "New imagery retrieved from Copernicus Sentinel-2 L2A and USGS Landsat Collection 2 via Microsoft Planetary Computer STAC and raw GeoTIFF crop API. Original scene/date/asset provenance is recorded in frames.jsonl. Contains modified Copernicus Sentinel data 2019-2021.\n"
        credits += "Landsat: Data available from the U.S. Geological Survey. Collection 2 Level-2, DN*0.0000275-0.2 followed by a fixed sRGB transfer. https://www.usgs.gov/faqs/are-landsat-data-cloud-still-considered-be-within-public-domain\n"
        credits += f"Landsat frames in this edition: {sum(r.get('landsat_replacement',False) for r in ordered)}. Black placeholders: 0.\n"
        credits += "Existing baseline photos, where retained, are recropped by a rigid geographic transform to the original source aspect ratio. No synthetic geographic shapes or fill.\n"
        (output / "CREDITS.txt").write_text(credits, encoding="utf-8")


def trim_cache(root, budget=24 * 1024**3):
    """Only evict regenerable raw crop caches (native-crops, native-patches) under this imagery directory."""
    root = Path(root).resolve()
    if root.name != "imagery":
        raise ValueError(f"Unexpected raw cache root {root}: expected an <output>/imagery directory")
    entries = []
    for name in ["native-crops", "native-patches"]:
        directory = (root / name).resolve()
        if not directory.is_relative_to(root):
            raise ValueError("Raw cache outside output")
        for path in directory.glob("*.tif"):
            try:
                stat = path.stat()
                entries.append((stat.st_mtime, stat.st_size, path))
            except FileNotFoundError:
                pass
    total = sum(n for _, n, _ in entries)
    removed = 0
    freed = 0
    for _, length, path in sorted(entries):
        if total <= budget:
            break
        try:
            path.unlink()
            total -= length
            removed += 1
            freed += length
        except (PermissionError, FileNotFoundError):
            pass
    if removed:
        write_json(
            root / "cache-maintenance.json",
            dict(
                removed_raw_crops=removed,
                bytes_freed=freed,
                remaining_bytes=total,
                policy="LRU/FIFO eviction of newly generated raw GeoTIFF caches only; output photos, masks, metadata, and original data retained",
            ),
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=OUTPUT)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--follow", action="store_true")
    a = p.parse_args()
    retrieve(a.output, a.workers, a.follow)
