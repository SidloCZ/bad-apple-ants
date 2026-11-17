"""Verified, restartable downloads: the Zenodo land/water mask and the Natural Earth layers."""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RECORD = "https://zenodo.org/api/records/14899246"
DOI = "https://doi.org/10.5281/zenodo.14899246"
MASK_CREDIT = (
    "Georg Erich Kindermann (2025), Global Land / Water Mask Map with 0.3 arc sec. (10 m) Resolution, v1, "
    "doi:10.5281/zenodo.14899246, CC BY 4.0; cropped/resampled/rotated"
)


def request_json(url: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"User-Agent": "BadAppleCoastline/0.1", "Content-Type": "application/json"}
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=90) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if attempt == 3 or (exc.code != 429 and exc.code < 500):
                raise
            delay = max(2**attempt, float(exc.headers.get("Retry-After", "1")))
            time.sleep(min(60, delay))
        except (urllib.error.URLError, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(2**attempt)


def digest(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_mask(directory: Path, tiles: list[str] | None = None, workers: int = 3) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    metadata = request_json(RECORD)
    license_id = metadata["metadata"].get("license", {}).get("id")
    if license_id != "cc-by-4.0":
        raise ValueError(f"Dataset license changed: {license_id!r}; review before using")
    (directory / "zenodo-14899246.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    files = [f for f in metadata["files"] if f["key"].endswith((".tif", ".vrt", ".pdf"))]
    if any(Path(f["key"]).name != f["key"] or "/" in f["key"] or "\\" in f["key"] for f in files):
        raise ValueError("Unsafe filename in dataset metadata")
    available = {f["key"] for f in files if f["key"].endswith(".tif")}
    if tiles is not None and set(tiles) - available:
        raise ValueError(f"Unknown mask tiles: {set(tiles) - available}")
    selected = sorted(
        [f for f in files if not f["key"].endswith(".tif") or tiles is None or f["key"] in tiles],
        key=lambda f: f["size"],
    )

    def download(f):
        path = directory / f["key"]
        algorithm, expected = f["checksum"].split(":")
        if path.exists() and path.stat().st_size == f["size"] and digest(path, algorithm) == expected:
            return
        partial = path.with_suffix(path.suffix + ".part")
        if partial.exists() and partial.stat().st_size == f["size"] and digest(partial, algorithm) == expected:
            partial.replace(path)
            return
        for attempt in range(4):
            try:
                url = f"https://zenodo.org/records/14899246/files/{f['key']}?download=1"
                offset = partial.stat().st_size if partial.exists() else 0
                if offset >= f["size"]:
                    offset = 0
                req = urllib.request.Request(
                    url, headers={"User-Agent": "BadAppleCoastline/0.1", "Range": f"bytes={offset}-"}
                )
                with urllib.request.urlopen(req, timeout=120) as response:
                    if offset and (
                        response.status != 206
                        or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")
                    ):
                        offset = 0
                    with partial.open("ab" if offset else "wb") as stream:
                        for chunk in iter(lambda: response.read(256 * 1024), b""):
                            stream.write(chunk)
                if partial.stat().st_size != f["size"] or digest(partial, algorithm) != expected:
                    raise RuntimeError(f"Checksum/size mismatch: {path.name}")
                partial.replace(path)
                print(f"download: {path.name} ({f['size'] / 1e6:.1f} MB), verified", flush=True)
                return
            except (urllib.error.URLError, TimeoutError):
                if attempt == 3:
                    raise
                time.sleep(2**attempt)

    with ThreadPoolExecutor(max_workers=max(1, min(workers, 4))) as pool:
        list(pool.map(download, selected))
    return {"doi": DOI, "license": license_id, "tiles": len([f for f in selected if f["key"].endswith(".tif")])}


# Natural Earth vector layers, pinned to one commit of the official GitHub mirror.
NATURAL_EARTH_COMMIT = "ca96624a56bd078437bca8184e78163e5039ad19"
NATURAL_EARTH_URL = f"https://raw.githubusercontent.com/nvkelso/natural-earth-vector/{NATURAL_EARTH_COMMIT}/geojson/"
NATURAL_EARTH = {
    # file name: (destination relative to assets/coastline, SHA-256)
    "ne_110m_land.geojson": (
        "ne_110m_land.geojson",
        "9e0729ee253ca7d7a5c4ae9395fb1902264c5377c52e224d13dd85010e2835d9",
    ),
    "ne_10m_admin_0_map_units.geojson": (
        "natural-earth/ne_10m_admin_0_map_units.geojson",
        "57da82be755f4afccd8f3b14251bb2752f5df1395f47d2d86f817470c4a48862",
    ),
    "ne_10m_geography_marine_polys.geojson": (
        "natural-earth/ne_10m_geography_marine_polys.geojson",
        "53f865e8ffa966cdd402145c82c5cd14ee7ce974cd0eb9a3f59f03a4cfd2d66c",
    ),
    "ne_10m_admin_0_disputed_areas.geojson": (
        "natural-earth/ne_10m_admin_0_disputed_areas.geojson",
        "9cafef8b7dfb6b164dc58f218f981f4ace9f716f6c03795d4c62d1ac9f3d50f5",
    ),
}


def fetch_natural_earth(directory: Path) -> dict:
    """World map, country and sea layers (public domain), verified by SHA-256."""
    fetched = {}
    for name, (relative, expected) in NATURAL_EARTH.items():
        path = directory / relative
        if not (path.exists() and digest(path) == expected):
            path.parent.mkdir(parents=True, exist_ok=True)
            request = urllib.request.Request(NATURAL_EARTH_URL + name, headers={"User-Agent": "BadAppleCoastline/0.1"})
            with urllib.request.urlopen(request, timeout=300) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != expected:
                raise RuntimeError(f"Natural Earth checksum mismatch: {name}")
            partial = path.with_suffix(path.suffix + ".part")
            partial.write_bytes(data)
            partial.replace(path)
            print(f"download: {name} ({len(data) / 1e6:.1f} MB), verified", flush=True)
        fetched[name] = str(path)
    return {"natural_earth_commit": NATURAL_EARTH_COMMIT, "files": fetched}
