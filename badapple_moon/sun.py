"""Local Sun elevation for the footer.

The STAC field `view:sun_elevation` of the Kaguya TC observations is one value per observation that barely
changes along an orbit (it is not the Sun's elevation at the imaged place: 88.6°S showed 6.0° although the
Sun can never rise more than about 3° there). So the footer computes the elevation at the matched place
from the sub-solar point (JPL Horizons, MOON_ME frame, east longitude) at the observation time."""

from __future__ import annotations

import json
import math
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

HORIZONS = "https://ssd.jpl.nasa.gov/api/horizons.api"
STAC_ITEM = "https://stac.astrogeology.usgs.gov/api/collections/{}/items/{}"
COLLECTIONS = (
    "kaguya_terrain_camera_monoscopic_uncontrolled_observations",
    "kaguya_terrain_camera_stereoscopic_uncontrolled_observations",
)
CACHE = Path("cache/moon/subsolar.json")


def _get(url: str, tries: int = 5):
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code == 404 or attempt == tries - 1:
                raise
        except Exception:
            if attempt == tries - 1:
                raise
        time.sleep(2 + 3 * attempt)


def observation_time(raster: str) -> str:
    """Start time of one TC observation from its STAC item (for indexes that predate `datetime`)."""
    order = COLLECTIONS if raster[3] == "S" else COLLECTIONS[::-1]
    for collection in order:
        try:
            item = json.loads(_get(STAC_ITEM.format(collection, raster)))
        except urllib.error.HTTPError:
            continue
        return item["properties"]["datetime"]
    raise KeyError(raster)


def _jd(iso: str) -> float:
    return 2440587.5 + datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp() / 86400


def fetch_subsolar(times: dict[str, str], batch: int = 80) -> dict[str, list[float]]:
    """{raster: ISO time} -> {raster: [sub-solar east longitude, latitude]} in degrees."""
    items = sorted(times.items(), key=lambda kv: kv[1])
    out = {}
    for i in range(0, len(items), batch):
        chunk = items[i : i + batch]
        query = {
            "format": "text",
            "COMMAND": "'301'",
            "CENTER": "'500@399'",
            "EPHEM_TYPE": "'OBSERVER'",
            "QUANTITIES": "'15'",
            "TLIST_TYPE": "'JD'",
            "TIME_TYPE": "'UT'",
            "ANG_FORMAT": "'DEG'",
            "CSV_FORMAT": "'YES'",
            "OBJ_DATA": "'NO'",
            "TLIST": "'" + " ".join(f"{_jd(t):.7f}" for _, t in chunk) + "'",
        }
        text = _get(HORIZONS + "?" + urllib.parse.urlencode(query)).decode()
        if "MOON_ME" not in text or "East-longitude positive" not in text:
            raise ValueError("unexpected Horizons frame or longitude convention")
        rows = text.split("$$SOE")[1].split("$$EOE")[0].strip().splitlines()
        if len(rows) != len(chunk):
            raise ValueError(f"Horizons returned {len(rows)} rows for {len(chunk)} times")
        for (raster, _), row in zip(chunk, rows):
            cells = [c.strip() for c in row.split(",")]
            out[raster] = [float(cells[3]), float(cells[4])]
        print(f"\rsub-solar points {len(out)}/{len(items)}", end="", flush=True)
    print()
    return out


def load_subsolar(rasters: set[str], records: dict[str, dict] | None = None, cache: Path = CACHE):
    """Sub-solar point of every TC raster, cached; missing ones are fetched (time from the index or STAC)."""
    known = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    missing = sorted(r for r in rasters if r not in known)
    if missing:
        records = records or {}
        import concurrent.futures as futures

        lookup = [r for r in missing if not (records.get(r) or {}).get("datetime")]
        with futures.ThreadPoolExecutor(16) as pool:
            fetched = dict(zip(lookup, pool.map(observation_time, lookup)))
        times = {r: (records.get(r) or {}).get("datetime") or fetched[r] for r in missing}
        known.update(fetch_subsolar(times))
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(known, indent=0), encoding="utf-8")
    return known


def local_sun(lat: float, lon: float, subsolar: list[float]) -> tuple[float, bool]:
    """(Sun elevation in degrees above the local horizon of a sphere, morning?) at (lat, lon) east."""
    sub_lon, sub_lat = subsolar
    p, b = math.radians(lat), math.radians(sub_lat)
    elevation = math.degrees(
        math.asin(math.sin(p) * math.sin(b) + math.cos(p) * math.cos(b) * math.cos(math.radians(lon - sub_lon)))
    )
    morning = (sub_lon - lon + 540) % 360 - 180 > 0  # Sun east of the place: it is still rising
    return elevation, morning


def camera_line(elevation: float, morning: bool) -> str:
    elevation = round(elevation, 1) or 0.0  # no "-0.0"
    return f"KAGUYA TERRAIN CAMERA  {'MORNING' if morning else 'EVENING'}  SUN {elevation:4.1f}°"
