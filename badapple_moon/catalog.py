from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

STAC = "https://stac.astrogeology.usgs.gov/api/collections/{}/items?limit=250"
TC_MONO = "kaguya_terrain_camera_monoscopic_uncontrolled_observations"
TC_EXTRA = {
    "stereo": "kaguya_terrain_camera_stereoscopic_uncontrolled_observations",
    "sp": "kaguya_terrain_camera_spsupport_uncontrolled_observations",
}


def _get(url: str, tries: int = 6) -> dict:
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                return json.load(response)
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(2 + 3 * attempt)
    raise RuntimeError(url)


def _projection(wkt: str) -> str:
    if "North Polar" in wkt:
        return "north_polar"
    if "South Polar" in wkt:
        return "south_polar"
    if "Equirectangular" in wkt:
        return "equirectangular"
    raise ValueError(wkt[:120])


def _row(feature: dict, keep_wkt: bool) -> dict:
    p = feature["properties"]
    band = feature["assets"]["image"]["raster:bands"][0]
    row = {
        "id": feature["id"],
        "collection": feature.get("collection"),
        "href": feature["assets"]["image"]["href"],
        "bbox": feature["bbox"],
        "lat": p["proj:centroid"]["lat"],
        "lon": p["proj:centroid"]["lon"],
        "gsd": p["gsd"],
        "shape": p["proj:shape"],
        "transform": p["proj:transform"][:6],
        "projection": _projection(p["proj:wkt2"]),
        "sun_elevation": p.get("view:sun_elevation"),
        "sun_azimuth": p.get("view:sun_azimuth"),
        "off_nadir": p.get("view:off_nadir"),
        "datetime": p.get("datetime"),
        "nodata": band.get("nodata"),
        "valid_percent": band.get("statistics", {}).get("valid_percent"),
    }
    if keep_wkt:
        row["wkt"] = p["proj:wkt2"]
    return row


def _pages(url: str):
    while url:
        page = _get(url)
        yield page["features"]
        url = next((link["href"] for link in page.get("links", []) if link["rel"] == "next"), None)


def harvest(collection: str, output: Path, tile_deg: float | None = None, workers: int = 16) -> int:
    """Write one compact JSON line per STAC item. With tile_deg the globe is split into bbox tiles
    fetched in parallel (items crossing tiles are de-duplicated by id)."""
    import concurrent.futures as futures

    output.parent.mkdir(parents=True, exist_ok=True)
    base = STAC.format(collection)
    if tile_deg is None:
        urls = [base]
    else:
        urls = []
        lat = -90.0
        while lat < 90:
            lon = -180.0
            while lon < 180:
                urls.append(f"{base}&bbox={lon},{lat},{min(lon + tile_deg, 180)},{min(lat + tile_deg, 90)}")
                lon += tile_deg
            lat += tile_deg
    seen: set[str] = set()
    tmp = output.with_suffix(".part")
    with tmp.open("w", encoding="utf-8") as stream, futures.ThreadPoolExecutor(workers) as pool:
        for done, features in enumerate(pool.map(lambda u: [f for page in _pages(u) for f in page], urls), 1):
            for feature in features:
                if feature["id"] in seen:
                    continue
                seen.add(feature["id"])
                stream.write(json.dumps(_row(feature, tile_deg is None), separators=(",", ":")) + "\n")
            print(f"\r{collection}: {done}/{len(urls)} queries, {len(seen)} items", end="", flush=True)
    tmp.replace(output)
    print()
    return len(seen)


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]
