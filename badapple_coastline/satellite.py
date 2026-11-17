"""Retrieve cloud-screened Sentinel-2 COG windows via the public STAC catalog."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from threading import Lock
from urllib.error import URLError
from urllib.parse import urlsplit, parse_qs

import numpy as np
from PIL import Image

from .atlas import corners, view_transform
from .data import digest, request_json
from .quality import POLICY, file_quality, image_quality

STAC = "https://earth-search.aws.element84.com/v1"
PLACES = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/ca96624a56bd078437bca8184e78163e5039ad19/geojson/ne_10m_populated_places.geojson"
_catalog_locks, _catalog_guard = {}, Lock()
_scene_counts, _scene_locks, _scene_guard, _scene_disabled = {}, {}, Lock(), set()
_sas_tokens, _sas_guard = {}, Lock()
SEARCH_FIELDS = {
    "include": [
        "id",
        "collection",
        "geometry",
        "properties.datetime",
        "properties.eo:cloud_cover",
        "assets.visual.href",
        "assets.scl.href",
        "assets.SCL.href",
    ]
}
_BAND_ASSETS = [
    "aot",
    "blue",
    "coastal",
    "green",
    "nir",
    "nir08",
    "nir09",
    "red",
    "rededge1",
    "rededge2",
    "rededge3",
    "swir16",
    "swir22",
    "wvp",
    "visual",
    "scl",
]
# This server includes all default assets unless the unwanted nested fields are
# explicitly excluded; including visual.href alone still returns every band.
SEARCH_FIELDS["exclude"] = (
    [f"assets.{name}" for name in _BAND_ASSETS if name not in ["visual", "scl"]]
    + [f"assets.{name}-jp2" for name in _BAND_ASSETS]
    + ["assets.thumbnail", "assets.tileinfo_metadata", "assets.granule_metadata", "links", "bbox"]
)


def compact_item(item):
    result = {key: item[key] for key in ["id", "collection", "geometry"] if key in item}
    result["properties"] = {
        key: value for key, value in item.get("properties", {}).items() if key in ["datetime", "eo:cloud_cover"]
    }
    result["assets"] = {
        key: {"href": asset["href"]}
        for key, asset in item.get("assets", {}).items()
        if key in ["visual", "scl", "SCL"] and "href" in asset
    }
    return result


def fetch_places(destination: Path):
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = request_json(PLACES)
        places = {
            "type": "FeatureCollection",
            "source_url": PLACES,
            "license_url": "https://www.naturalearthdata.com/about/terms-of-use/",
            "features": [
                {
                    "geometry": f["geometry"],
                    "properties": {k: f["properties"].get(k, "") for k in ["NAMEASCII", "ADM0NAME"]},
                }
                for f in data["features"]
            ],
        }
        destination.write_text(json.dumps(places), encoding="utf-8")


@lru_cache(maxsize=4)
def place_data(path: str, modified_ns: int):
    cities = json.loads(Path(path).read_text(encoding="utf-8"))["features"]
    coords = np.array([c["geometry"]["coordinates"][:2] for c in cities])
    return cities, coords


def nearest_place(path: Path, lon: float, lat: float) -> dict:
    cities, coords = place_data(str(path.resolve()), path.stat().st_mtime_ns)
    p1, p2 = np.deg2rad(lat), np.deg2rad(coords[:, 1])
    dlat, dlon = p2 - p1, np.deg2rad(coords[:, 0] - lon)
    distances = (
        6371.0088
        * 2
        * np.arcsin(np.sqrt(np.clip(np.sin(dlat / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2, 0, 1)))
    )
    i = int(np.argmin(distances))
    props = cities[i]["properties"]
    return {
        "name": props["NAMEASCII"],
        "country": props["ADM0NAME"],
        "distance_km": float(distances[i]),
        "type": "nearest Natural Earth settlement, not exact place name",
    }


def signed_href(href: str) -> str:
    if ".blob.core.windows.net/" not in href:
        return href
    parsed = urlsplit(href)
    if set(parse_qs(parsed.query)) & {"st", "se", "sp"}:
        return href
    account = parsed.hostname.split(".")[0]
    container = parsed.path.strip("/").split("/")[0]
    identity = (account, container)
    # Match Microsoft's SDK: reuse one container token until close to expiry.
    # Tokens stay in memory; source metadata always retains unsigned asset URLs.
    with _sas_guard:
        cached = _sas_tokens.get(identity)
        if (
            cached is None
            or (
                datetime.fromisoformat(cached["msft:expiry"].replace("Z", "+00:00")) - datetime.now(timezone.utc)
            ).total_seconds()
            < 60
        ):
            cached = request_json(f"https://planetarycomputer.microsoft.com/api/sas/v1/token/{account}/{container}")
            _sas_tokens[identity] = cached
    return href + ("&" if parsed.query else "?") + cached["token"].lstrip("?")


def cached_asset(href, side, resampling, directory=Path("cache/coastline-satellite/scene-cache")):
    """Materialize frequently reused scenes; retain native CRS, bounds and bands."""
    import rasterio
    from affine import Affine

    key = hashlib.sha256(f"{href}:{side}:{resampling.name}".encode()).hexdigest()[:24]
    path = directory / f"{key}.tif"
    if path.is_file():
        return str(path)
    for shared in os.environ.get("COASTLINE_SCENE_READ_CACHE", "").split(os.pathsep):
        if shared and (Path(shared) / f"{key}.tif").is_file():
            return str(Path(shared) / f"{key}.tif")
    if os.environ.get("COASTLINE_SCENE_CACHE"):
        directory = Path(os.environ["COASTLINE_SCENE_CACHE"])
        path = directory / f"{key}.tif"
        if path.is_file():
            return str(path)
    with _scene_guard:
        if (href, side, resampling.name) in _scene_disabled:
            return signed_href(href)
        _scene_counts[href] = _scene_counts.get(href, 0) + 1
        hits = _scene_counts[href]
        lock = _scene_locks.setdefault(key, Lock())
    if hits < 4:
        return signed_href(href)
    directory.mkdir(parents=True, exist_ok=True)
    with lock:
        if path.is_file():
            return str(path)
        with rasterio.open(signed_href(href)) as source:
            width = min(side, source.width)
            height = max(1, round(source.height * width / source.width))
            data = source.read(out_shape=(source.count, height, width), resampling=resampling)
            transform = source.transform @ Affine.scale(source.width / width, source.height / height)
            profile = {
                "driver": "GTiff",
                "width": width,
                "height": height,
                "count": source.count,
                "dtype": data.dtype,
                "crs": source.crs,
                "transform": transform,
                "nodata": source.nodata,
                "compress": "deflate",
                "predictor": 2,
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
            }
            metadata = {
                "source_href": href,
                "native_size": [source.width, source.height],
                "cached_size": [width, height],
                "resampling": resampling.name,
                "crs": str(source.crs),
                "bounds": list(source.bounds),
            }
        partial = path.with_suffix(".partial.tif")
        with rasterio.open(partial, "w", **profile) as sink:
            sink.write(data)
        partial.replace(path)
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return str(path)


def project_asset(asset: dict, transform, size: tuple[int, int], bands, resampling, dtype="uint8", force_native=False):
    import rasterio
    from affine import Affine
    from rasterio.warp import reproject, transform_bounds
    from rasterio.windows import Window, from_bounds

    dest = np.zeros((len(bands), size[1], size[0]), dtype=dtype)
    lat = (transform @ (size[0] / 2, size[1] / 2))[1]
    pixel_m = 111320 * np.hypot(transform.a * np.cos(np.deg2rad(lat)), transform.d)
    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        GDAL_HTTP_MAX_RETRY=int(os.environ.get("COASTLINE_HTTP_MAX_RETRY", "3")),
        GDAL_HTTP_TIMEOUT=int(os.environ.get("COASTLINE_HTTP_TIMEOUT", "60")),
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
        CPL_VSIL_CURL_CACHE_SIZE="268435456",
    ):
        side = 4096 if pixel_m < 55 else 2048
        try:
            path = signed_href(asset["href"]) if force_native else cached_asset(asset["href"], side, resampling)
        except (rasterio.errors.RasterioIOError, TimeoutError, URLError):
            with _scene_guard:
                _scene_disabled.add((asset["href"], side, resampling.name))
            path = signed_href(asset["href"])
        with rasterio.open(path) as source:
            points = np.array(corners(transform, size))
            bbox = (points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max())
            bounds = transform_bounds("EPSG:4326", source.crs, *bbox, densify_pts=21)
            window = from_bounds(*bounds, transform=source.transform)
            try:
                window = window.intersection(Window(0, 0, source.width, source.height))
            except rasterio.errors.WindowError:
                return dest
            # Read COG overviews before reprojection; never download a full 10 m tile
            # merely to render a small window hundreds of metres per output pixel.
            ow = max(1, min(round(window.width), int(np.ceil(window.width * abs(source.transform.a) / pixel_m * 1.5))))
            oh = max(
                1, min(round(window.height), int(np.ceil(window.height * abs(source.transform.e) / pixel_m * 1.5)))
            )
            data = source.read(bands, window=window, out_shape=(len(bands), oh, ow), resampling=resampling)
            local_transform = source.window_transform(window) @ Affine.scale(window.width / ow, window.height / oh)
            reproject(
                source=data,
                destination=dest,
                src_transform=local_transform,
                src_crs=source.crs,
                dst_transform=transform,
                dst_crs="EPSG:4326",
                src_nodata=0,
                dst_nodata=0,
                resampling=resampling,
                num_threads=1,
            )
    return dest


def window_key(record: dict, size, interval, max_cloud):
    key = {k: record[k] for k in ["longitude", "latitude", "width_km", "rotation_clockwise"]}
    key.update(size=size, interval=interval, max_cloud=max_cloud, provider=STAC, schema=2)
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:24]


def cached_window(path: Path, metadata_path: Path):
    try:
        if path.is_file() and metadata_path.is_file():
            cached = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(cached, dict) and cached.get("coverage") == 1 and cached.get("image_sha256") == digest(path):
                return dict(cached, path=str(path.resolve()))
    except (OSError, ValueError):
        pass
    return None


def resolve_interval(record, interval="auto"):
    if interval != "auto":
        return interval
    if record["latitude"] >= 25:
        return "2020-05-01T00:00:00Z/2020-10-01T23:59:59Z"
    if record["latitude"] <= -25:
        return "2020-11-01T00:00:00Z/2021-04-01T23:59:59Z"
    return "2020-01-01T00:00:00Z/2021-12-31T23:59:59Z"


def catalog(bbox, interval, max_cloud, directory, viewport=None, max_pages=16):
    # Nearby frames can reuse catalog responses without changing the actual view.
    snapped = [float(np.floor(v * 4) / 4) for v in bbox[:2]] + [float(np.ceil(v * 4) / 4) for v in bbox[2:]]
    payload = {
        "collections": ["sentinel-2-l2a"],
        "bbox": snapped,
        "datetime": interval,
        "query": {"eo:cloud_cover": {"lt": max_cloud}},
        "limit": 200,
    }
    key = hashlib.sha256(json.dumps([STAC, payload], sort_keys=True).encode()).hexdigest()[:24]
    directory = directory / "catalogs"
    directory.mkdir(exist_ok=True)
    path = directory / f"{key}.json"
    with _catalog_guard:
        lock = _catalog_locks.setdefault(key, Lock())
    with lock:
        if path.exists():
            result = json.loads(path.read_text(encoding="utf-8"))
        else:
            shared = next(
                (
                    Path(folder) / path.name
                    for folder in os.environ.get("COASTLINE_CATALOG_READ_CACHE", "").split(os.pathsep)
                    if folder and (Path(folder) / path.name).is_file()
                ),
                None,
            )
            result = (
                json.loads(shared.read_text(encoding="utf-8"))
                if shared
                else request_json(STAC + "/search", dict(payload, fields=SEARCH_FIELDS))
            )
        result["features"] = [compact_item(item) for item in result.get("features", [])]
        # Large views need more than the first 200 acquisitions. Follow the
        # catalog's own pagination until the actual rotated view is covered.
        pages = 0
        while viewport is not None and pages < max_pages:
            from rasterio.features import rasterize

            features = result.get("features", [])
            geometries = [(item["geometry"], 1) for item in features if item.get("geometry")]
            if any(not item.get("geometry") for item in features):
                break
            if (
                geometries
                and rasterize(geometries, out_shape=(24, 32), transform=viewport, dtype="uint8").mean() >= 0.99
            ):
                break
            following = next((link for link in result.get("links", []) if link.get("rel") == "next"), None)
            if following is None:
                break
            href = following["href"]
            if not href.startswith(STAC + "/"):
                raise ValueError("Unexpected STAC pagination host")
            body = following.get("body", {}) if following.get("method", "GET") == "POST" else None
            if following.get("merge"):
                body = dict(payload, **(body or {}))
            if body is not None:
                body = dict(body, fields=SEARCH_FIELDS)
            page = request_json(href, body)
            known = {item["id"] for item in features}
            additional = [compact_item(item) for item in page.get("features", []) if item["id"] not in known]
            result["features"] = features + additional
            result["links"] = page.get("links", [])
            pages += 1
            if not additional:
                break
        partial = path.with_suffix(".partial.json")
        partial.write_text(json.dumps(result), encoding="utf-8")
        partial.replace(path)
        shared_write = os.environ.get("COASTLINE_CATALOG_WRITE_CACHE")
        if shared_write:
            directory = Path(shared_write)
            directory.mkdir(parents=True, exist_ok=True)
            shared = directory / path.name
            temporary = shared.with_name(shared.name + f".{os.getpid()}.partial")
            temporary.write_text(json.dumps(result), encoding="utf-8")
            temporary.replace(shared)
        return result


def fetch_window(
    record: dict,
    destination: Path,
    size=(960, 720),
    interval="auto",
    max_cloud: float = 25.0,
    max_scenes: int = 12,
    quality_policy=None,
    exclude_scene_ids=(),
) -> dict:
    from rasterio.enums import Resampling

    if min(size) < 1 or size[0] * 3 != size[1] * 4 or max_scenes < 1 or not 0 <= max_cloud <= 100:
        raise ValueError("Invalid satellite size, scene count or cloud threshold")
    interval = resolve_interval(record, interval)
    destination.mkdir(parents=True, exist_ok=True)
    key = window_key(record, size, interval, max_cloud)
    path, meta_path = destination / f"{key}.png", destination / f"{key}.json"
    cached = cached_window(path, meta_path)

    def accepted_cache(cached):
        if cached is None:
            return None
        if quality_policy:
            try:
                quality = file_quality(cached["path"])
            except OSError:
                return None
            if not quality["acceptable"]:
                return None
            return dict(cached, quality_policy=POLICY, visible_quality=quality)
        return cached

    cached = accepted_cache(cached)
    if cached is not None:
        return cached
    for shared in os.environ.get("COASTLINE_SHARED_WINDOW_CACHE", "").split(os.pathsep):
        if not shared:
            continue
        shared_path = Path(shared) / f"{key}.png"
        shared_metadata = Path(shared) / f"{key}.json"
        cached = accepted_cache(cached_window(shared_path, shared_metadata))
        if cached is not None:
            return cached
    transform = view_transform(
        record["longitude"], record["latitude"], record["width_km"], record["rotation_clockwise"], size
    )
    points = np.array(corners(transform, size))
    bbox = [float(points[:, 0].min()), float(points[:, 1].min()), float(points[:, 0].max()), float(points[:, 1].max())]
    if bbox[0] < -180 or bbox[2] > 180:
        raise ValueError("Satellite windows crossing the antimeridian are not supported")
    coarse_transform = view_transform(
        record["longitude"], record["latitude"], record["width_km"], record["rotation_clockwise"], (32, 24)
    )
    result = catalog(bbox, interval, max_cloud, destination, viewport=coarse_transform)

    def preference(item):
        month = int(item["properties"]["datetime"][5:7])
        season = 0
        if quality_policy and abs(record["latitude"]) >= 40:
            good = [7, 8] if record["latitude"] >= 0 else [1, 2]
            season = 0 if month in good else 1 if month in ([6, 9] if record["latitude"] >= 0 else [12, 3]) else 2
        return season, item["properties"].get("eo:cloud_cover", 100)

    items = sorted([item for item in result["features"] if item["id"] not in exclude_scene_ids], key=preference)
    for item in items:
        if "scl" in item["assets"] and "SCL" not in item["assets"]:
            item["assets"]["SCL"] = item["assets"]["scl"]
    # Cloud sorting alone favors thin orbital-edge fragments. Cover the actual
    # viewport with STAC footprints first, then try further clear acquisitions.
    from rasterio.features import rasterize

    masks = [
        (
            rasterize([(item["geometry"], 1)], out_shape=(24, 32), transform=coarse_transform, dtype="uint8").astype(
                bool
            )
            if item.get("geometry")
            else np.ones((24, 32), bool)
        )
        for item in items
    ]
    remaining = list(range(len(items)))
    needed = np.ones((24, 32), bool)
    order = []
    while remaining and needed.any():
        best = max(remaining, key=lambda j: int(np.sum(masks[j] & needed)))
        if not (masks[best] & needed).any():
            break
        order.append(best)
        remaining.remove(best)
        needed &= ~masks[best]
    if needed.mean() > 0.01:
        failure = {
            "status": "incomplete",
            "reason": "Available STAC footprints do not cover the viewport",
            "coverage": float(1 - needed.mean()),
            "coverage_type": "coarse geometric upper bound, not retrieved pixels",
            "scenes": [],
            "errors": [],
            "bbox": bbox,
            "key": key,
        }
        (destination / f"{key}.failure.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        raise RuntimeError(f"Sentinel-2 window {key}: catalog footprint coverage incomplete; no synthetic fill")
    ordered = [items[j] for j in order + remaining]
    rgb = np.zeros((3, size[1], size[0]), dtype="uint8")
    covered = np.zeros((size[1], size[0]), dtype=bool)
    fallback_rgb = np.zeros_like(rgb)
    fallback_source = np.zeros(covered.shape, dtype="uint16")
    fallback_snow = np.zeros(covered.shape, dtype=bool)
    scenes, errors = [], []
    for item in ordered[:max_scenes]:
        try:
            scl = project_asset(item["assets"]["SCL"], transform, size, [1], Resampling.nearest)[0]
            # Retain real cast shadows; reject no-data, saturated, clouds and cirrus.
            valid = (scl >= 2) & (scl <= 11)
            usable = valid & ~np.isin(scl, [8, 9, 10, 11] if quality_policy else [8, 9, 10]) & ~covered
            fallback_valid = valid & (fallback_source == 0)
            if not usable.any() and not fallback_valid.any():
                continue
            image = project_asset(item["assets"]["visual"], transform, size, [1, 2, 3], Resampling.bilinear)
            if quality_policy and not image_quality(image.transpose(1, 2, 0))["acceptable"]:
                continue
            rgb[:, usable] = image[:, usable]
            covered |= usable
            fallback_rgb[:, fallback_valid] = image[:, fallback_valid]
            fallback_source[fallback_valid] = len(scenes) + 1
            fallback_snow[fallback_valid] = scl[fallback_valid] == 11
            scenes.append(
                {
                    "id": item["id"],
                    "collection": item["collection"],
                    "datetime": item["properties"]["datetime"],
                    "eo_cloud_cover": item["properties"].get("eo:cloud_cover"),
                    "visual_href": item["assets"]["visual"]["href"],
                    "scl_href": item["assets"]["SCL"]["href"],
                    "stac_url": STAC + "/collections/sentinel-2-l2a/items/" + item["id"],
                    "contributed_clear_pixels": int(usable.sum()),
                }
            )
            # Avoid reading many more acquisitions to replace a handful of cloud
            # pixels. Every remaining pixel must already have real RGB coverage.
            if covered.all() or (covered.mean() >= 0.995 and (covered | (fallback_source != 0)).all()):
                break
        except Exception as exc:
            # Do not persist signed URLs or SAS tokens from GDAL error messages.
            errors.append(
                {"id": item["id"], "error_type": type(exc).__name__, "http_status": getattr(exc, "code", None)}
            )
    clear_coverage = float(covered.mean())
    obscured = ~covered & (fallback_source != 0)
    snow_pixels = obscured & fallback_snow
    cloud_pixels = obscured & ~fallback_snow
    rgb[:, obscured] = fallback_rgb[:, obscured]
    for i, scene in enumerate(scenes):
        scene["contributed_cloud_pixels"] = int(np.sum(cloud_pixels & (fallback_source == i + 1)))
        scene["contributed_snow_pixels"] = int(np.sum(snow_pixels & (fallback_source == i + 1)))
    covered |= obscured
    coverage = float(covered.mean())
    if not covered.all():
        failure = {
            "status": "incomplete",
            "coverage": coverage,
            "scenes": scenes,
            "errors": errors,
            "bbox": bbox,
            "key": key,
        }
        (destination / f"{key}.failure.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        raise RuntimeError(
            f"Sentinel-2 window {key}: clear coverage {coverage:.1%}; no synthetic fill. See failure JSON"
        )
    quality = image_quality(rgb.transpose(1, 2, 0)) if quality_policy else None
    if quality is not None and not quality["acceptable"]:
        raise RuntimeError(f"Sentinel-2 window {key}: whiteout rejected")
    years = sorted({s["datetime"][:4] for s in scenes})
    credit = "Contains modified Copernicus Sentinel data " + (
        years[0] if len(years) == 1 else f"{years[0]}-{years[-1]}"
    )
    Image.fromarray(rgb.transpose(1, 2, 0)).save(path)
    metadata = {
        "status": "complete",
        "path": str(path.resolve()),
        "key": key,
        "size": size,
        "bbox": bbox,
        "footprint_lonlat": points.tolist(),
        "coverage": coverage,
        "clear_coverage": clear_coverage,
        "cloud_fraction": float(cloud_pixels.mean()),
        "snow_fraction": float(snow_pixels.mean()),
        "scenes": scenes,
        "credit": credit,
        "interval": interval,
        "image_sha256": digest(path),
        "provider": "Earth Search / AWS Open Data",
        "stac_api": STAC,
        "accessed_utc": datetime.now(timezone.utc).isoformat(),
        "processing": "Sentinel-2 L2A visual; prefer SCL-clear pixels; remaining cloud pixels retained from actual imagery and quantified; same-location temporal mosaic; rigid crop/rotation/resampling; frequently reused scenes cached at 2048/4096 pixels per tile",
        "license_url": "https://cds.climate.copernicus.eu/licences/ec-sentinel",
    }
    if quality_policy:
        metadata.update(quality_policy=POLICY, visible_quality=quality)
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"satellite: {key}, {len(scenes)} scenes", flush=True)
    return metadata
