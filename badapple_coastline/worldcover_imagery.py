"""ESA's actual Sentinel-2 annual RGB composite, via its documented public WMTS."""

from __future__ import annotations
import hashlib, io, json, math, os, urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image
from .atlas import corners, view_transform
from .data import digest
from .pipeline import write_json
from .quality import image_quality, file_quality, POLICY

BASE = "https://mapproxy.terrascope.be/mapproxy/wmts"
DOC = "https://docs.terrascope.be/Developers/WebServices/OGC/MapProxy.html"
LICENSE = "https://esa-worldcover.org/en/data-access"
RADIUS = 6378137.0
HALF = math.pi * RADIUS


def fetch(record, cache, size=(1444, 1080), year=2020, display_width=960):
    from affine import Affine
    from rasterio.warp import reproject
    from rasterio.enums import Resampling

    geo = [record[k] for k in ["longitude", "latitude", "width_km", "rotation_clockwise"]]
    transform = view_transform(*geo, size)
    points = np.array(corners(transform, size))
    if points[:, 0].min() < -180 or points[:, 0].max() > 180 or points[:, 1].min() < -60 or points[:, 1].max() > 83:
        raise ValueError("Outside composite optical coverage")
    layer = f"esa-worldcover-s2rgbnir-10m-{year}-v{1 if year==2020 else 2}_tcc"
    key = hashlib.sha256(json.dumps([geo, size, layer, "public-WMTS-v1"]).encode()).hexdigest()[:24]
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    photo = cache / f"{key}.png"
    meta = cache / f"{key}.json"
    if photo.is_file() and meta.is_file():
        result = json.loads(meta.read_text())
        if digest(photo) == result["image_sha256"] and file_quality(photo)["acceptable"]:
            return result
    # Match the delivered 960px photograph pane, while retaining original query/mask geometry.
    requested = max(10.0, record["width_km"] * 1000 / display_width)
    zoom = max(
        0, min(19, math.ceil(math.log2(156543.03392804097 * np.cos(np.deg2rad(record["latitude"])) / requested)))
    )
    n = 2**zoom
    resolution = 2 * HALF / (n * 256)
    xx = RADIUS * np.deg2rad(points[:, 0])
    yy = RADIUS * np.log(np.tan(math.pi / 4 + np.deg2rad(points[:, 1]) / 2))
    x0 = max(0, math.floor((xx.min() + HALF) / (resolution * 256)))
    x1 = min(n - 1, math.floor((xx.max() + HALF) / (resolution * 256)))
    y0 = max(0, math.floor((HALF - yy.max()) / (resolution * 256)))
    y1 = min(n - 1, math.floor((HALF - yy.min()) / (resolution * 256)))
    if (x1 - x0 + 1) * (y1 - y0 + 1) > 256:
        raise ValueError("Composite tile budget exceeded")
    tile_cache = cache.parent / "worldcover-tiles" / str(year)
    tile_cache.mkdir(parents=True, exist_ok=True)
    requests = [(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]

    def retrieve(tile):
        x, y = tile
        path = tile_cache / f"{zoom:02d}-{x}-{y}.png"
        url = f"{BASE}/{layer}/webmercator/{zoom:02d}/{x}/{y}.png"
        if path.is_file():
            raw = path.read_bytes()
        else:
            req = urllib.request.Request(
                url, headers={"User-Agent": "BadAppleCoastline/0.1 (ESA WorldCover attribution retained)"}
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                raw = response.read()
        with Image.open(io.BytesIO(raw)) as image:
            if image.size != (256, 256):
                raise ValueError("Unexpected WMTS tile shape")
            pixels = np.asarray(image.convert("RGBA"))
        if not path.is_file():
            temporary = path.with_name(path.name + f".{os.getpid()}.partial")
            temporary.write_bytes(raw)
            temporary.replace(path)
        return x, y, pixels, url

    source = np.zeros((4, (y1 - y0 + 1) * 256, (x1 - x0 + 1) * 256), np.uint8)
    urls = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for x, y, pixels, url in pool.map(retrieve, requests):
            source[:, (y - y0) * 256 : (y - y0 + 1) * 256, (x - x0) * 256 : (x - x0 + 1) * 256] = pixels.transpose(
                2, 0, 1
            )
            urls.append(url)
    source_transform = Affine(
        resolution, 0, -HALF + x0 * 256 * resolution, 0, -resolution, HALF - y0 * 256 * resolution
    )
    target = np.zeros((4, size[1], size[0]), np.uint8)
    reproject(
        source=source[:3],
        destination=target[:3],
        src_transform=source_transform,
        src_crs="EPSG:3857",
        dst_transform=transform,
        dst_crs="EPSG:4326",
        resampling=Resampling.bilinear,
        num_threads=1,
    )
    reproject(
        source=source[3],
        destination=target[3],
        src_transform=source_transform,
        src_crs="EPSG:3857",
        dst_transform=transform,
        dst_crs="EPSG:4326",
        resampling=Resampling.nearest,
        num_threads=1,
    )
    if (target[3] < 250).any():
        write_json(
            cache / f"{key}.failure.json",
            dict(
                geometry=geo,
                source_alpha_min=int(source[3].min()),
                coverage=float((target[3] >= 250).mean()),
                alpha_values=np.unique(target[3]).tolist(),
            ),
        )
        raise RuntimeError("Real annual composite has missing source pixels")
    quality = image_quality(target[:3].transpose(1, 2, 0))
    if not quality["acceptable"]:
        raise RuntimeError("Annual composite whiteout")
    native = Path(record.get("native_mask_path", "__missing__"))
    if native.is_file():
        with Image.open(native) as image:
            water = np.asarray(image.resize(size, Image.Resampling.NEAREST)) < 128
        if water.mean() > 0.01 and (target[:3].min(axis=0)[water] >= 215).mean() > 0.25:
            raise RuntimeError("Annual composite has bright ice/clouds over the matched water silhouette")
    Image.fromarray(target[:3].transpose(1, 2, 0)).save(photo)
    scene = dict(
        id=layer,
        collection="esa-worldcover-s2rgbnir",
        datetime=f"{year}-01-01T00:00:00Z",
        start_datetime=f"{year}-01-01T00:00:00Z",
        end_datetime=f"{year}-12-31T23:59:59Z",
        product="Sentinel-2 L2A annual cloud-screened median RGB composite",
        wmts_layer=layer,
        tile_urls=urls,
        source_resolution_m=10,
        rendered_ground_pixel_m=resolution * np.cos(np.deg2rad(record["latitude"])),
        documentation=DOC,
        license_url=LICENSE,
        doi="https://doi.org/10.71570/2e2s-y214" if year == 2020 else "https://doi.org/10.71570/79k4-x482",
    )
    result = dict(
        status="complete",
        path=str(photo.resolve()),
        size=list(size),
        coverage=1.0,
        cloud_metrics="annual product cloud screening; residual cloud fraction unavailable",
        scenes=[scene],
        annual_composite=True,
        credit=f"© ESA WorldCover project {year} / Contains modified Copernicus Sentinel data ({year}) processed by ESA WorldCover consortium",
        provider="ESA WorldCover / VITO Terrascope public WMTS",
        stac_api="https://services.terrascope.be/stac",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        image_sha256=digest(photo),
        quality_policy=POLICY,
        visible_quality=quality,
        footprint_lonlat=corners(transform, size),
        source_resolution_m=10,
        accessed_utc=datetime.now(timezone.utc).isoformat(),
        processing="Official Sentinel-2 L2A WorldCover annual median true-color raster, no map labels; RGB imagery tiles at or finer than displayed photo pixels; rigid geographic reprojection/crop/rotation; complete source alpha required; no invented or filled geographic pixels",
    )
    write_json(meta, result)
    return result
