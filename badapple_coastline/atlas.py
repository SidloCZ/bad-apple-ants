"""A small geographic search atlas derived only from the selected 10 m mask."""

from __future__ import annotations

import json
import hashlib
import os
import threading
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.ndimage import map_coordinates

from .data import DOI, digest

KM_PER_DEGREE = 111.32


def view_transform(
    lon: float, lat: float, width_km: float, angle: float, size: tuple[int, int], mirror_horizontal=False
):
    """Pixel-edge transform; clockwise rotation and optional explicit reflection."""
    from affine import Affine

    w, h = size
    a = np.deg2rad(angle)
    c, s = np.cos(a), np.sin(a)
    dx = width_km / w
    lon_scale = KM_PER_DEGREE * np.cos(np.deg2rad(lat))
    transform = Affine(
        c * dx / lon_scale,
        -s * dx / lon_scale,
        lon + (-c * w / 2 + s * h / 2) * dx / lon_scale,
        -s * dx / KM_PER_DEGREE,
        -c * dx / KM_PER_DEGREE,
        lat + (s * w / 2 + c * h / 2) * dx / KM_PER_DEGREE,
    )
    return transform @ Affine(-1, 0, w, 0, 1, 0) if mirror_horizontal else transform


def corners(transform, size):
    w, h = size
    return [transform @ p for p in [(0, 0), (w, 0), (w, h), (0, h)]]


def prepare_atlas(source: Path, destination: Path, resolution: float = 0.01, allow_partial: bool = False) -> dict:
    import rasterio
    from rasterio.enums import Resampling

    if resolution <= 0 or not np.isclose(30 / resolution, round(30 / resolution)):
        raise ValueError("Resolution must divide 30 degrees exactly")
    vrt = ET.parse(source / "landWater2020.vrt").getroot()
    sources = vrt.findall(".//SimpleSource")
    expected = [s.findtext("SourceFilename") for s in sources]
    if any(not n or Path(n).name != n or "/" in n or "\\" in n for n in expected):
        raise ValueError("VRT must reference tile filenames within the source directory")
    missing = [n for n in expected if not (source / n).is_file()]
    if missing and not allow_partial:
        raise FileNotFoundError(
            f"Missing {len(missing)} / {len(expected)} tiles. Fetch all, or explicitly use --allow-partial"
        )
    present = [n for n in expected if n not in missing]
    if not present:
        raise ValueError("No downloaded mask tiles")
    destination.mkdir(parents=True, exist_ok=True)
    building = destination / "land.building.npy"
    shape = (round(180 / resolution), round(360 / resolution))
    atlas = np.lib.format.open_memmap(building, mode="w+", dtype="uint8", shape=shape)
    atlas[:] = 255  # Missing coverage must never become synthetic sea.
    provenance = []
    for name in present:
        path = source / name
        with rasterio.open(path) as ds:
            size = round(30 / resolution)
            values = ds.read(1, out_shape=(size, size), resampling=Resampling.nearest)
            if not np.isin(values, [0, 1]).all():
                raise ValueError(f"Unexpected land/water codes: {np.unique(values)}")
            x = round((ds.bounds.left + 180) / resolution)
            y = round((90 - ds.bounds.top) / resolution)
            atlas[y : y + size, x : x + size] = values
            provenance.append({"file": name, "sha256": digest(path), "bounds": list(ds.bounds)})
        print(f"atlas: {len(provenance)}/{len(present)} {name}", flush=True)
    atlas.flush()
    del atlas
    building.replace(destination / "land.npy")
    manifest = {
        "schema": 1,
        "doi": DOI,
        "resolution_degrees": resolution,
        "source_directory": str(source.resolve()),
        "shape": shape,
        "land_value": 1,
        "water_value": 0,
        "unknown_value": 255,
        "global_coverage": not missing,
        "missing_tiles": missing,
        "resampling": "nearest; 0.3 arcsec source to search atlas",
        "tiles": provenance,
    }
    (destination / "atlas.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


@dataclass
class Atlas:
    path: Path

    def __post_init__(self):
        self.metadata = json.loads((self.path / "atlas.json").read_text(encoding="utf-8"))
        self.land = np.load(self.path / "land.npy", mmap_mode="r", allow_pickle=False)
        self.resolution = self.metadata["resolution_degrees"]

    def sample(
        self, lon: float, lat: float, width_km: float, angle: float, size: tuple[int, int] = (32, 24)
    ) -> tuple[np.ndarray, np.ndarray]:
        transform = view_transform(lon, lat, width_km, angle, size)
        yy, xx = np.mgrid[: size[1], : size[0]]
        longitude = transform.a * (xx + 0.5) + transform.b * (yy + 0.5) + transform.c
        latitude = transform.d * (xx + 0.5) + transform.e * (yy + 0.5) + transform.f
        ix = (longitude + 180) / self.resolution - 0.5
        iy = (90 - latitude) / self.resolution - 0.5
        in_bounds = (ix >= 0) & (ix <= self.land.shape[1] - 1) & (iy >= 0) & (iy <= self.land.shape[0] - 1)
        values = map_coordinates(self.land, [iy, ix], order=0, mode="nearest", prefilter=False)
        valid = in_bounds & (values != 255)
        return np.where(valid, values, 0).astype(np.float32), valid

    def coastal_centers(self, count: int, seed: int = 42, bounds=None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        points = []
        # Restrict optical search to Sentinel-2's latitude range, leaving a margin.
        for tile in self.metadata["tiles"]:
            left, bottom, right, top = tile["bounds"]
            x0 = round((left + 180) / self.resolution)
            y0 = round((90 - top) / self.resolution)
            step = max(1, round(0.04 / self.resolution))
            size = round(30 / self.resolution)
            part = self.land[y0 : y0 + size : step, x0 : x0 + size : step]
            edge = np.zeros(part.shape, dtype=bool)
            edge[1:, :] |= (part[1:, :] != part[:-1, :]) & (part[1:, :] != 255) & (part[:-1, :] != 255)
            edge[:, 1:] |= (part[:, 1:] != part[:, :-1]) & (part[:, 1:] != 255) & (part[:, :-1] != 255)
            y, x = np.nonzero(edge)
            if len(y):
                lon = -180 + (x0 + x * step + 0.5) * self.resolution
                lat = 90 - (y0 + y * step + 0.5) * self.resolution
                ok = (lat >= -54) & (lat <= 80) & (lon > -179) & (lon < 179)
                if bounds is not None:
                    ok &= (lon >= bounds[0]) & (lat >= bounds[1]) & (lon <= bounds[2]) & (lat <= bounds[3])
                points.append(np.column_stack([lon[ok], lat[ok]]))
        if not points:
            raise ValueError("No valid water boundaries in atlas")
        points = np.concatenate(points)
        return points[rng.choice(len(points), min(count, len(points)), replace=False)]

    def native_mask_key(self, lon, lat, width_km, angle, size=(260, 195)):
        identity = [lon, lat, width_km, angle, size, self.metadata["tiles"]]
        return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]

    def native_mask(self, lon, lat, width_km, angle, size=(260, 195)) -> Path:
        """Read the selected window directly from the original 0.3 arcsecond tiles."""
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.vrt import WarpedVRT
        from PIL import Image

        key = self.native_mask_key(lon, lat, width_km, angle, size)
        directory = self.path / "native-masks"
        directory.mkdir(exist_ok=True)
        path = directory / f"{key}.png"
        if path.exists():
            return path
        from .native_rle import sample as rle_sample

        exact = rle_sample(self.metadata, lon, lat, width_km, angle, size)
        if exact is not None:
            partial = path.with_name(f"{path.stem}.{os.getpid()}.{threading.get_ident()}.partial.png")
            try:
                Image.fromarray(exact * 255).save(partial)
                try:
                    partial.replace(path)
                except PermissionError:
                    if not path.is_file():
                        raise
            finally:
                partial.unlink(missing_ok=True)
            return path
        transform = view_transform(lon, lat, width_km, angle, size)
        points = np.array(corners(transform, size))
        bbox = [points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max()]
        values = np.full((size[1], size[0]), 255, np.uint8)
        root = Path(self.metadata.get("source_directory", "assets/coastline"))
        for tile in self.metadata["tiles"]:
            left, bottom, right, top = tile["bounds"]
            if left >= bbox[2] or right <= bbox[0] or bottom >= bbox[3] or top <= bbox[1]:
                continue
            with rasterio.open(root / tile["file"]) as source:
                with WarpedVRT(
                    source,
                    crs="EPSG:4326",
                    transform=transform,
                    width=size[0],
                    height=size[1],
                    nodata=255,
                    resampling=Resampling.nearest,
                ) as vrt:
                    layer = vrt.read(1)
                    covered = layer != 255
                    values[covered] = layer[covered]
        if (values == 255).any():
            raise ValueError("Native mask window includes unacquired source coverage")
        partial = path.with_name(f"{path.stem}.{os.getpid()}.{threading.get_ident()}.partial.png")
        try:
            Image.fromarray(values * 255).save(partial)
            try:
                partial.replace(path)
            except PermissionError:
                if not path.is_file():
                    raise
        finally:
            partial.unlink(missing_ok=True)
        return path
