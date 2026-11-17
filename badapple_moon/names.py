from __future__ import annotations

import hashlib
import io
import json
import struct
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DBF = Path("assets/moon/raw/names/MOON_nomenclature_center_pts.dbf")
# USGS Gazetteer of Planetary Nomenclature, regenerated daily (the video used the 2026-11-01 export).
NAMES_URL = "https://asc-planetarynames-data.s3.us-west-2.amazonaws.com/MOON_nomenclature_center_pts.zip"
R_KM = 1737.4

# Descriptor terms reworded for display; the rest stay IAU's own Latin terms (Mare, Mons, ...).
TYPE_EN = {"Satellite Feature": "Crater", "Landing site name": "Landing site", "Albedo Feature": "Albedo feature"}


def _read_dbf(path: Path) -> list[dict]:
    raw = path.read_bytes()
    count, header, size = struct.unpack("<IHH", raw[4:12])
    fields, pos = [], 32
    while raw[pos] != 0x0D:
        name = raw[pos : pos + 11].split(b"\0")[0].decode()
        fields.append((name, raw[pos + 16]))
        pos += 32
    rows = []
    for i in range(count):
        record = raw[header + i * size : header + (i + 1) * size]
        if record[:1] == b"*":
            continue
        offset, row = 1, {}
        for name, length in fields:
            row[name] = record[offset : offset + length].decode("utf-8", "replace").strip()
            offset += length
        rows.append(row)
    return rows


class Gazetteer:
    """Nearest IAU-approved lunar feature: the smallest one containing the point, else the nearest rim."""

    def __init__(self, path: Path = DBF):
        rows = [r for r in _read_dbf(path) if r.get("approval", "").startswith("Adopted")]
        self.names = [r["name"] or r["clean_name"] for r in rows]  # clean_name drops ' - . and accents
        self.types = [r["type"].split(",")[0].strip() for r in rows]
        lon = np.array([float(r["center_lon"]) for r in rows])
        self.lon = np.radians((lon + 180) % 360 - 180)
        self.lat = np.radians(np.array([float(r["center_lat"]) for r in rows]))
        self.radius = np.array([float(r["diameter"] or 0) for r in rows]) / 2

    def nearest(self, lon: float, lat: float) -> tuple[str, str, float, float]:
        phi, lam = np.radians(lat), np.radians(lon)
        cos = np.sin(phi) * np.sin(self.lat) + np.cos(phi) * np.cos(self.lat) * np.cos(self.lon - lam)
        dist = R_KM * np.arccos(np.clip(cos, -1, 1))
        inside = np.flatnonzero(dist < self.radius)
        if inside.size:
            i = inside[np.argmin(self.radius[inside])]
            edge = 0.0
        else:
            gap = dist - self.radius
            i = int(np.argmin(gap))
            edge = float(gap[i])
        kind = self.types[i]
        return self.names[i], TYPE_EN.get(kind, kind), edge, float(2 * self.radius[i])


def fetch_names(dbf: Path = DBF) -> dict:
    """Download the IAU lunar nomenclature (center points) and record its date and checksum."""
    request = urllib.request.Request(NAMES_URL, headers={"User-Agent": "bad-apple-earth-moon/1.0"})
    with urllib.request.urlopen(request, timeout=300) as response:
        data = response.read()
    dbf.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        archive.extractall(dbf.parent)
    if not dbf.is_file():
        raise RuntimeError(f"{dbf.name} missing from {NAMES_URL}")
    record = {
        "url": NAMES_URL,
        "zip_sha256": hashlib.sha256(data).hexdigest(),
        "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (dbf.parent / "source.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record
