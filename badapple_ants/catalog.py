"""Build an ant image catalog from the GBIF REST API or a GBIF DwC-A download."""

from __future__ import annotations

import heapq
import io
import json
import re
import sys
import time
import zipfile
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import httpx
from tqdm import tqdm

GBIF_SEARCH_URL = "https://api.gbif.org/v1/occurrence/search"
FORMICIDAE_KEY = 4342
INAT_DATASET_KEY = "50c9509d-22c7-4a22-a47d-8c48425ef4a7"
ANTWEB_DATASET_KEY = "13b70480-bd69-11dd-b15f-b8a03c50a862"

INAT_RE = re.compile(
    r"^(https://(?:inaturalist-open-data\.s3\.amazonaws\.com|static\.inaturalist\.org)"
    r"/photos/(\d+))/(?:original|large|medium|small)(\.(?:jpg|jpeg|png|webp))$",
    re.IGNORECASE,
)
ANTWEB_RE = re.compile(
    r"^(https://static\.antweb\.org/images/.+?_(h|p|d|v|f)(?:_(\d+))?)_(low|med|high|thumbview)"
    r"(\.(?:jpg|jpeg|png))$",
    re.IGNORECASE,
)
MCZ_RE = re.compile(
    r"^https://iiif\.mcz\.harvard\.edu/iiif/3/(\d+)/full/[^/]+/0/default\.(jpg|jpeg|png)$",
    re.IGNORECASE,
)
NMNH_RE = re.compile(
    r"^https://collections\.nmnh\.si\.edu/media/\?i=(\d+)(?:&.*)?$",
    re.IGNORECASE,
)

ANTWEB_VIEW_MAP = {
    "h": "head",
    "p": "profile",
    "d": "dorsal",
    "v": "ventral",
    "f": "frontal",
}


def normalize_license(raw: str | None, *, commercial_only: bool = False) -> str | None:
    """Return a clean short license code for permissive licenses, or None to exclude."""
    if not raw:
        return None
    s = raw.lower().replace("_", "-")
    if any(
        bad in s
        for bad in (
            "no-deriv",
            "by-nd",
            "by-nc-nd",
            "all-rights",
            "all rights",
            "restricted",
            "usage conditions",
        )
    ):
        return None
    if "copyright" in s and "cc by" not in s and "cc-by" not in s and "creativecommons" not in s:
        return None

    ver = "-4.0" if ("4.0" in s or "4-0" in s) else ("-3.0" if ("3.0" in s or "3-0" in s) else "")
    code: str | None = None
    if "zero" in s or "cc0" in s or "no rights reserved" in s:
        code = "CC0-1.0"
    elif "by-nc-sa" in s:
        code = f"CC-BY-NC-SA{ver}"
    elif "by-nc" in s:
        code = f"CC-BY-NC{ver}"
    elif "by-sa" in s:
        code = f"CC-BY-SA{ver}"
    elif (
        "creativecommons.org/licenses/by/" in s
        or "creativecommons.org/licences/by/" in s
        or "cc by" in s
        or "cc-by" in s
    ):
        code = f"CC-BY{ver}"

    if code and commercial_only and "NC" in code:
        return None
    return code


def parse_media_candidate(
    ident: str,
    *,
    title: str = "",
    description: str = "",
    media_license: str = "",
    occ_license: str | None = None,
    commercial_only: bool = False,
) -> dict | None:
    """Validate URL, label exclusion, and license; return normalized source/URL dict or None."""
    # Only fall back to occurrence license when media-level license is empty, so media-level ND/copyright rejects.
    raw_lic = media_license.strip() if media_license and media_license.strip() else (occ_license or "")
    lic = normalize_license(raw_lic, commercial_only=commercial_only)
    if not lic:
        return None

    ident = ident.strip()
    title_l = title.lower()
    desc_l = description.lower()

    if match := INAT_RE.match(ident):
        base_url, photo_id, ext = match.group(1), match.group(2), match.group(3)
        return {
            "id": f"inat_{photo_id}",
            "source": "inat",
            "view": "field",
            "thumb_url": f"{base_url}/small{ext}",
            "full_url": f"{base_url}/original{ext}",
            "fallback_full_url": f"{base_url}/large{ext}",
            "license": lic,
        }

    if match := ANTWEB_RE.match(ident):
        if "_l_" in ident.lower() or "label" in title_l or "label" in desc_l:
            return None
        prefix, view_code, _, _, ext = match.groups()
        stem = prefix.rsplit("/", 1)[-1]
        return {
            "id": f"antweb_{stem}",
            "source": "antweb",
            "view": ANTWEB_VIEW_MAP.get(view_code.lower(), "specimen"),
            "thumb_url": f"{prefix}_thumbview{ext}",
            "full_url": f"{prefix}_high{ext}",
            "fallback_full_url": f"{prefix}_med{ext}",
            "license": lic,
        }

    if match := MCZ_RE.match(ident):
        if "label" in title_l or "label" in desc_l or "lbsa" in desc_l:
            return None
        mcz_id, ext = match.group(1), match.group(2).lower()
        text = f"{title_l} {desc_l}"
        if "head" in text or "frontal" in text:
            view = "head"
        elif "dorsal" in text:
            view = "dorsal"
        elif "lateral" in text or "hala" in text:
            view = "lateral"
        else:
            view = "specimen"
        return {
            "id": f"mcz_{mcz_id}",
            "source": "mcz",
            "view": view,
            "thumb_url": f"https://iiif.mcz.harvard.edu/iiif/3/{mcz_id}/full/!256,256/0/default.{ext}",
            "full_url": ident,
            "fallback_full_url": f"https://iiif.mcz.harvard.edu/iiif/3/{mcz_id}/full/!1200,1200/0/default.{ext}",
            "license": lic,
        }

    if match := NMNH_RE.match(ident):
        nmnh_id = match.group(1)
        suf = title.strip().rsplit("_", 1)[-1].upper() if "_" in title else ""
        if "label" in desc_l or "pin" in desc_l or suf.startswith("L") or suf.startswith("O"):
            return None
        if "head" in desc_l or suf in {"H", "HEAD"}:
            view = "head"
        elif "dorsal" in desc_l or suf in {"D", "1D"}:
            view = "dorsal"
        elif "profile" in desc_l or suf in {"P", "1P"}:
            view = "profile"
        else:
            view = "specimen"
        return {
            "id": f"nmnh_{nmnh_id}",
            "source": "nmnh",
            "view": view,
            "thumb_url": f"https://collections.nmnh.si.edu/media/?i={nmnh_id}&w=320",
            "full_url": ident,
            "fallback_full_url": f"https://collections.nmnh.si.edu/media/?i={nmnh_id}&h=1200",
            "license": lic,
        }

    return None


def _get_json(client: httpx.Client, params: dict) -> dict:
    for attempt in range(4):
        try:
            resp = client.get(GBIF_SEARCH_URL, params=params, timeout=30.0)
            if resp.status_code == 429:
                time.sleep(2.0 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError:
            if attempt == 3:
                raise
            time.sleep(1.5 * (attempt + 1))
    return {}


def _taxon_fields(occ: dict) -> tuple[str, str]:
    species = (occ.get("species") or "").strip()
    genus = (occ.get("genus") or "").strip()
    if not genus and species:
        genus = species.split()[0]
    return species, genus


def _fetch_inat(client: httpx.Client, target: int, seen_thumbs: set[str]) -> list[dict]:
    items: list[dict] = []
    years = [2025, 2024, 2023, 2022, 2021, 2020]
    per_year_cap = max(1, (target + len(years) - 1) // len(years))

    for year in years:
        if len(items) >= target:
            break
        year_count = 0
        offset = 0
        while year_count < per_year_cap and len(items) < target and offset < 90000:
            data = _get_json(
                client,
                {
                    "taxonKey": FORMICIDAE_KEY,
                    "datasetKey": INAT_DATASET_KEY,
                    "mediaType": "StillImage",
                    "year": year,
                    "limit": 300,
                    "offset": offset,
                },
            )
            results = data.get("results") or []
            if not results:
                break
            for occ in results:
                occ_key = occ.get("key")
                species, genus = _taxon_fields(occ)
                country = occ.get("country") or occ.get("countryCode") or ""
                lat = occ.get("decimalLatitude")
                lon = occ.get("decimalLongitude")
                occ_lic = occ.get("license")
                added_for_occ = 0

                for m in occ.get("media") or []:
                    cand = parse_media_candidate(
                        m.get("identifier") or "",
                        title=m.get("title") or "",
                        description=m.get("description") or "",
                        media_license=m.get("license") or "",
                        occ_license=occ_lic,
                    )
                    if not cand or cand["source"] != "inat":
                        continue
                    thumb_url = cand["thumb_url"]
                    if thumb_url in seen_thumbs:
                        continue
                    seen_thumbs.add(thumb_url)
                    items.append(
                        {
                            **cand,
                            "species": species,
                            "genus": genus,
                            "country": country,
                            "lat": lat,
                            "lon": lon,
                            "creator": (m.get("creator") or occ.get("recordedBy") or "").strip(),
                            "gbif_url": f"https://www.gbif.org/occurrence/{occ_key}" if occ_key else "",
                            "reference_url": (m.get("references") or occ.get("references") or "").strip(),
                        }
                    )
                    added_for_occ += 1
                    year_count += 1
                    if added_for_occ >= 2 or year_count >= per_year_cap or len(items) >= target:
                        break
                if year_count >= per_year_cap or len(items) >= target:
                    break
            if data.get("endOfRecords"):
                break
            offset += 300
        print(f"catalog inat year={year}: total={len(items)}/{target}", flush=True)
    return items


def _fetch_antweb(client: httpx.Client, target: int, seen_thumbs: set[str]) -> list[dict]:
    items: list[dict] = []
    # Spread across spaced offset windows in GBIF's AntWeb occurrences for genus diversity.
    windows = [0, 8000, 20000, 35000, 50000, 68000, 85000]
    per_window_cap = max(1, (target + len(windows) - 1) // len(windows))

    for start_offset in windows:
        if len(items) >= target:
            break
        win_count = 0
        offset = start_offset
        while win_count < per_window_cap and len(items) < target and offset < 99000:
            data = _get_json(
                client,
                {
                    "taxonKey": FORMICIDAE_KEY,
                    "datasetKey": ANTWEB_DATASET_KEY,
                    "mediaType": "StillImage",
                    "limit": 300,
                    "offset": offset,
                },
            )
            results = data.get("results") or []
            if not results:
                break
            for occ in results:
                occ_key = occ.get("key")
                species, genus = _taxon_fields(occ)
                country = occ.get("country") or occ.get("countryCode") or ""
                lat = occ.get("decimalLatitude")
                lon = occ.get("decimalLongitude")
                occ_lic = occ.get("license")
                added_for_occ = 0

                for m in occ.get("media") or []:
                    cand = parse_media_candidate(
                        m.get("identifier") or "",
                        title=m.get("title") or "",
                        description=m.get("description") or "",
                        media_license=m.get("license") or "",
                        occ_license=occ_lic,
                    )
                    if not cand or cand["source"] != "antweb":
                        continue
                    thumb_url = cand["thumb_url"]
                    if thumb_url in seen_thumbs:
                        continue
                    seen_thumbs.add(thumb_url)
                    items.append(
                        {
                            **cand,
                            "species": species,
                            "genus": genus,
                            "country": country,
                            "lat": lat,
                            "lon": lon,
                            "creator": (m.get("creator") or occ.get("recordedBy") or "").strip(),
                            "gbif_url": f"https://www.gbif.org/occurrence/{occ_key}" if occ_key else "",
                            "reference_url": (m.get("references") or "").strip(),
                        }
                    )
                    added_for_occ += 1
                    win_count += 1
                    if added_for_occ >= 2 or win_count >= per_window_cap or len(items) >= target:
                        break
                if win_count >= per_window_cap or len(items) >= target:
                    break
            if data.get("endOfRecords"):
                break
            offset += 300
        print(f"catalog antweb offset={start_offset}: total={len(items)}/{target}", flush=True)
    return items


def _write_catalog_jsonl(items: list[dict], output_path: Path) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_suffix(".jsonl.part")
    with partial.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    partial.replace(output_path)

    sources = Counter(it["source"] for it in items)
    species_set = {it["species"] for it in items if it["species"]}
    genera_set = {it["genus"] for it in items if it["genus"]}
    stats = {
        "total": len(items),
        "sources": dict(sources),
        "unique_species": len(species_set),
        "unique_genera": len(genera_set),
    }
    print(
        f"catalog complete: {stats['total']} items "
        f"({', '.join(f'{k}={v}' for k, v in sorted(sources.items()))}), "
        f"{stats['unique_species']} species, {stats['unique_genera']} genera",
        flush=True,
    )
    return stats


def build_catalog(
    output_path: Path,
    inat_target: int = 6500,
    antweb_target: int = 3500,
) -> dict:
    """Query GBIF REST API for iNaturalist and AntWeb ant images and write JSONL."""
    seen_thumbs: set[str] = set()
    headers = {"User-Agent": "badapple-ants-proto/0.1"}

    with httpx.Client(headers=headers, follow_redirects=True) as client:
        inat_items = _fetch_inat(client, inat_target, seen_thumbs) if inat_target > 0 else []
        antweb_items = _fetch_antweb(client, antweb_target, seen_thumbs) if antweb_target > 0 else []

    return _write_catalog_jsonl(inat_items + antweb_items, output_path)


def _float_or_none(val: str) -> float | None:
    if not val:
        return None
    try:
        return float(val)
    except ValueError:
        return None


def _resolve_dwca_zip(zip_path: Path | None) -> Path:
    if zip_path is not None:
        return zip_path
    zips = sorted(Path("assets").glob("*.zip"))
    if not zips:
        raise FileNotFoundError("No DwC-A .zip archive found in assets/; pass --zip explicitly")
    return zips[-1]


def build_catalog_dwca(
    output_path: Path,
    zip_path: Path | None = None,
    inat_target: int = 150000,
    antweb_target: int = 85000,
    mcz_target: int = 10000,
    nmnh_target: int = 5000,
    commercial_only: bool = False,
    fill_remainder: bool = True,
    max_per_occurrence: int = 2,
) -> dict:
    """Stream GBIF Darwin Core Archive zip and write a species-balanced JSONL catalog."""
    archive = _resolve_dwca_zip(zip_path)
    targets = {
        "inat": max(0, inat_target),
        "antweb": max(0, antweb_target),
        "mcz": max(0, mcz_target),
        "nmnh": max(0, nmnh_target),
    }
    total_goal = sum(targets.values())
    if total_goal <= 0:
        raise ValueError("Total catalog target must be positive")

    slack = (targets["mcz"] + targets["nmnh"]) if fill_remainder else 0
    heap_caps = {
        "inat": targets["inat"] + slack,
        "antweb": targets["antweb"] + slack,
        "mcz": targets["mcz"],
        "nmnh": targets["nmnh"],
    }

    print(f"catalog-dwca: reading occurrences from {archive}...", flush=True)
    occ_map: dict[str, tuple] = {}
    with zipfile.ZipFile(archive) as zf:
        with io.TextIOWrapper(zf.open("occurrence.txt"), encoding="utf-8", errors="replace") as f:
            hdr = f.readline().rstrip("\r\n").split("\t")
            i_gid = hdr.index("gbifID")
            i_sp = hdr.index("species")
            i_gen = hdr.index("genus")
            i_cc = hdr.index("countryCode")
            i_country = hdr.index("level0Name") if "level0Name" in hdr else i_cc
            i_lat = hdr.index("decimalLatitude")
            i_lon = hdr.index("decimalLongitude")
            i_lic = hdr.index("license")
            i_rec = hdr.index("recordedBy")
            i_ds = hdr.index("datasetKey")
            i_ref = hdr.index("references")
            max_idx = max(i_gid, i_sp, i_gen, i_cc, i_country, i_lat, i_lon, i_lic, i_rec, i_ds, i_ref)

            for line in tqdm(f, desc="dwca-occ", unit="occ"):
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) <= max_idx:
                    continue
                gid = parts[i_gid]
                species = sys.intern(parts[i_sp].strip()) if parts[i_sp] else ""
                genus = sys.intern(parts[i_gen].strip()) if parts[i_gen] else ""
                if not genus and species:
                    genus = sys.intern(species.split()[0])
                country = sys.intern((parts[i_country] or parts[i_cc]).strip())
                occ_lic = sys.intern(parts[i_lic].strip()) if parts[i_lic] else ""
                ds_key = sys.intern(parts[i_ds].strip()) if parts[i_ds] else ""
                occ_map[gid] = (
                    species,
                    genus,
                    country,
                    _float_or_none(parts[i_lat]),
                    _float_or_none(parts[i_lon]),
                    occ_lic,
                    parts[i_rec].strip(),
                    ds_key,
                    parts[i_ref].strip(),
                )

        heaps: dict[str, list[tuple[int, int, int, dict]]] = {k: [] for k in targets}
        sp_seen: dict[str, Counter[str]] = {k: Counter() for k in targets}
        seen_thumbs: set[str] = set()
        occ_counts: Counter[str] = Counter()
        seq = 0

        with io.TextIOWrapper(zf.open("multimedia.txt"), encoding="utf-8", errors="replace") as f:
            mhdr = f.readline().rstrip("\r\n").split("\t")
            mi_gid = mhdr.index("gbifID")
            mi_type = mhdr.index("type")
            mi_id = mhdr.index("identifier")
            mi_ref = mhdr.index("references")
            mi_title = mhdr.index("title")
            mi_desc = mhdr.index("description")
            mi_cr = mhdr.index("creator")
            mi_lic = mhdr.index("license")
            mi_rh = mhdr.index("rightsHolder") if "rightsHolder" in mhdr else mi_cr
            m_max = max(mi_gid, mi_type, mi_id, mi_ref, mi_title, mi_desc, mi_cr, mi_lic, mi_rh)

            for line in tqdm(f, desc="dwca-media", unit="row"):
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) <= m_max or parts[mi_type] != "StillImage":
                    continue
                gid = parts[mi_gid]
                if max_per_occurrence > 0 and occ_counts[gid] >= max_per_occurrence:
                    continue
                occ = occ_map.get(gid)
                if occ is None:
                    continue
                species, genus, country, lat, lon, occ_lic, rec_by, ds_key, occ_ref = occ

                cand = parse_media_candidate(
                    parts[mi_id],
                    title=parts[mi_title],
                    description=parts[mi_desc],
                    media_license=parts[mi_lic],
                    occ_license=occ_lic,
                    commercial_only=commercial_only,
                )
                if cand is None:
                    continue
                src = cand["source"]
                cap = heap_caps.get(src, 0)
                if cap <= 0:
                    continue
                thumb_url = cand["thumb_url"]
                if thumb_url in seen_thumbs:
                    continue
                seen_thumbs.add(thumb_url)
                occ_counts[gid] += 1

                sp_key = species or genus or "_unknown_"
                rank = sp_seen[src][sp_key]
                sp_seen[src][sp_key] = rank + 1
                tie = zlib.crc32(cand["id"].encode("utf-8"))
                seq += 1

                creator = (parts[mi_cr] or rec_by or parts[mi_rh] or "").strip()
                item = {
                    **cand,
                    "species": species,
                    "genus": genus,
                    "country": country,
                    "lat": lat,
                    "lon": lon,
                    "creator": creator,
                    "dataset_key": ds_key,
                    "gbif_url": f"https://www.gbif.org/occurrence/{gid}",
                    "reference_url": (parts[mi_ref] or occ_ref or "").strip(),
                }
                entry = (-rank, -tie, seq, item)
                hp = heaps[src]
                if len(hp) < cap:
                    heapq.heappush(hp, entry)
                elif entry > hp[0]:
                    heapq.heapreplace(hp, entry)

    # Free occurrence lookup table before ranking and JSON serialization.
    occ_map.clear()
    del occ_map

    # Extract items ordered best-diversity-first (-neg_rank ascending = rank 0 first) per source.
    ordered: dict[str, list[dict]] = {}
    selected: list[dict] = []
    for src in ("inat", "antweb", "mcz", "nmnh"):
        ranked = [e[3] for e in sorted(heaps[src], key=lambda e: (-e[0], -e[1], e[2]))]
        take = min(len(ranked), targets[src])
        selected.extend(ranked[:take])
        ordered[src] = ranked[take:]

    if fill_remainder and len(selected) < total_goal:
        deficit = total_goal - len(selected)
        extras = ordered["inat"] + ordered["antweb"]
        selected.extend(extras[:deficit])

    selected.sort(key=lambda it: (it["source"], it["id"]))
    return _write_catalog_jsonl(selected, output_path)
