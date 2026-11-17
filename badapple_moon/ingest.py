from __future__ import annotations

import concurrent.futures as futures
import json
import time
from pathlib import Path

import numpy as np

from .catalog import load
from .geometry import fit_quadratic, ortho_forward

GDAL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
    GDAL_HTTP_MAX_RETRY="6",
    GDAL_HTTP_RETRY_DELAY="2",
    GDAL_HTTP_TIMEOUT="120",
    VSI_CACHE="FALSE",
)
MOON_GEOGRAPHIC = (
    'GEOGCS["Moon",DATUM["Moon",SPHEROID["Moon",1737400,0]],'
    'PRIMEM["Reference Meridian",0],UNIT["degree",0.0174532925199433]]'
)
MAX_BASE_METRES = 95.0
# Observations whose 1-99 % range is this narrow are mostly signal-starved exposures: once stretched,
# the seam between the detector halves, column stripes and compression blocks dominate them.
MIN_RANGE_DN = 250


def qa_images(tiles: Path) -> list[str]:
    """Write tc_exclude.json: TC observations that must never become candidates."""
    excluded = []
    with (tiles / "tc_index.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                r = json.loads(line)
                if r["stretch"][1] - r["stretch"][0] < MIN_RANGE_DN:
                    excluded.append(r["id"])
    (tiles / "tc_exclude.json").write_text(json.dumps(excluded), encoding="utf-8")
    print(f"excluded {len(excluded)} low-signal observations (1-99 % range < {MIN_RANGE_DN} DN)")
    return excluded


def stretch_u8(values: np.ndarray, valid: np.ndarray, low: float, high: float) -> np.ndarray:
    """1..255 for valid pixels, 0 reserved for nodata."""
    scaled = np.clip((values.astype(np.float32) - low) / max(high - low, 1e-6), 0, 1)
    out = (1 + np.rint(scaled * 254)).astype(np.uint8)
    out[~valid] = 0
    return out


def erode(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] &= mask[:-1, :]
    out[:-1, :] &= mask[1:, :]
    out[:, 1:] &= mask[:, :-1]
    out[:, :-1] &= mask[:, 1:]
    out[0, :] = out[-1, :] = False
    out[:, 0] = out[:, -1] = False
    return out


def _ingest_one(row: dict, max_base_m: float = MAX_BASE_METRES) -> tuple[dict, bytes]:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import transform

    url = "/vsicurl/" + row["href"]
    with rasterio.Env(**GDAL_ENV):
        with rasterio.open(url) as full:
            width, height, crs, affine = full.width, full.height, full.crs, full.transform
            overviews = full.overviews(1)
        level = 0
        for index, factor in enumerate(overviews):
            if row["gsd"] * factor <= max_base_m:
                level = index
        with rasterio.open(url, overview_level=level) as src:
            data = src.read(1)
        geographic = CRS.from_wkt(MOON_GEOGRAPHIC)
        # Quadratic tangent-plane (km) -> full-resolution pixel-edge coordinates, fitted through GDAL/PROJ.
        grid = np.linspace(0, 1, 13)
        cols, rows = np.meshgrid(grid * width, grid * height)
        px, py = affine * (cols.ravel(), rows.ravel())
        lons, lats = transform(crs, geographic, list(px), list(py))
        cx, cy = affine * (width / 2, height / 2)
        (lon0,), (lat0,) = transform(crs, geographic, [cx], [cy])
    valid = erode(data != row["nodata"])
    if valid.mean() < 0.02:
        raise ValueError("almost no valid pixels")
    low, high = np.percentile(data[valid], [1, 99])
    if high - low < 30:  # a handful of observations hold only DN 0..2 (dark/failed exposures)
        raise ValueError(f"flat image: DN {low}..{high}")
    base = stretch_u8(data, valid, float(low), float(high))
    tx, ty = ortho_forward(np.asarray(lons), np.asarray(lats), lon0, lat0)
    forward, residual = fit_quadratic(tx, ty, cols.ravel(), rows.ravel())
    inverse, inverse_residual = fit_quadratic(cols.ravel(), rows.ravel(), tx, ty)
    record = {
        "id": row["id"],
        "href": row["href"],
        "width": int(base.shape[1]),
        "height": int(base.shape[0]),
        "full_width": int(width),
        "full_height": int(height),
        "lon0": float(lon0),
        "lat0": float(lat0),
        "forward": forward.tolist(),
        "inverse": inverse.tolist(),
        "fit_residual_px": residual,
        "inverse_residual_km": inverse_residual,
        "stretch": [float(low), float(high)],
        "gsd": row["gsd"],
        "sun_elevation": row["sun_elevation"],
        "sun_azimuth": row["sun_azimuth"],
        "datetime": row["datetime"],
        "valid_fraction": float(valid.mean()),
    }
    return record, base.tobytes()


GLOBAL_SOURCES = {
    # name: (file, decimation) -- decimated with GDAL's nodata-aware averaging
    "wac": ("assets/moon/raw/wac_100m.tif", 8),
    # The 474 m TC global orthomosaic was tried and dropped: its stripes and block gaps
    # produced false matches with the silhouettes.
}


def ingest_global(output: Path) -> None:
    """Average-decimate the global equirectangular mosaic(s) into a u8 search base (0 = nodata)."""
    import rasterio
    from rasterio.enums import Resampling

    output.mkdir(parents=True, exist_ok=True)
    for name, (path, factor) in GLOBAL_SOURCES.items():
        target = output / f"{name}.u8"
        if target.exists():
            continue
        with rasterio.open(path) as src:
            shape = (src.height // factor, src.width // factor)
            nodata, res_m = src.nodata, src.transform.a
            if src.dtypes[0] == "uint8":
                data = src.read(1, out_shape=shape, resampling=Resampling.average, masked=True)
                values = np.asarray(data, dtype=np.float32)
                valid = ~np.ma.getmaskarray(data) & (values != 0)
            else:
                # The TC orthomosaic also marks gaps with -32767 besides its declared nodata, so
                # average manually and keep only blocks whose pixels are all positive.
                full = src.read(1)[: shape[0] * factor, : shape[1] * factor].astype(np.float32)
                blocks = full.reshape(shape[0], factor, shape[1], factor)
                values = blocks.mean(axis=(1, 3))
                valid = (blocks > 0).all(axis=(1, 3))
        valid = erode(valid)
        low, high = np.percentile(values[valid], [1, 99])
        base = stretch_u8(values, valid, float(low), float(high))
        base.tofile(target)
        meta = {
            "name": name,
            "file": path,
            "width": shape[1],
            "height": shape[0],
            "factor": factor,
            "native_metres": res_m,
            "stretch": [float(low), float(high)],
            "nodata": nodata,
            "valid_fraction": float(valid.mean()),
        }
        (output / f"{name}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        print(name, meta)


def ingest_tc(
    catalog: Path,
    output: Path,
    workers: int = 48,
    limit: int | None = None,
    part: int = 0,
    parts: int = 1,
    max_base_m: float = MAX_BASE_METRES,
    ids: set[str] | None = None,
) -> None:
    """Fetch one overview per Kaguya TC image (<= max_base_m per pixel) into a resumable u8 pack + JSONL index."""
    output.mkdir(parents=True, exist_ok=True)
    pack_path, index_path = output / "tc_base.u8", output / "tc_index.jsonl"
    done: set[str] = set()
    end = 0
    if index_path.exists():
        for line in index_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                done.add(record["id"])
                end = max(end, record["offset"] + record["width"] * record["height"])
    rows = [row for row in load(catalog)[part::parts] if row["id"] not in done and (ids is None or row["id"] in ids)]
    if limit is not None:
        rows = rows[:limit]
    with pack_path.open("ab") as pack:
        pack.truncate(end)
    failures = []
    started = time.time()
    with (
        pack_path.open("r+b") as pack,
        index_path.open("a", encoding="utf-8") as index,
        futures.ThreadPoolExecutor(workers) as pool,
    ):
        pack.seek(end)
        jobs = {pool.submit(_ingest_one, row, max_base_m): row["id"] for row in rows}
        for count, job in enumerate(futures.as_completed(jobs), 1):
            try:
                record, data = job.result()
            except Exception as error:  # keep going; failures are listed at the end
                failures.append({"id": jobs[job], "error": repr(error)[:300]})
                print(f"\nfailed {jobs[job]}: {repr(error)[:160]}", flush=True)
                continue
            record["offset"] = end
            pack.write(data)
            end += len(data)
            pack.flush()
            index.write(json.dumps(record, separators=(",", ":")) + "\n")
            index.flush()
            if count % 50 == 0 or count == len(rows):
                rate = count / (time.time() - started)
                print(
                    f"\r{count}/{len(rows)} images, {rate:.1f}/s, {end / 1e9:.2f} GB, {len(failures)} failed",
                    end="",
                    flush=True,
                )
    print()
    (output / "tc_failures.json").write_text(json.dumps(failures, indent=1), encoding="utf-8")
