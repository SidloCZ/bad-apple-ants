"""Build a mixed iNaturalist + AntWeb ant image catalog from the GBIF REST API."""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path

import httpx

GBIF_SEARCH_URL = "https://api.gbif.org/v1/occurrence/search"
FORMICIDAE_KEY = 4342
INAT_DATASET_KEY = "50c9509d-22c7-4a22-a47d-8c48425ef4a7"
ANTWEB_DATASET_KEY = "13b70480-bd69-11dd-b15f-b8a03c50a862"

INAT_RE = re.compile(
    r"^(https://(?:inaturalist-open-data\.s3\.amazonaws\.com|static\.inaturalist\.org)"
    r"/photos/(\d+))/original(\.(?:jpg|jpeg|png|webp))$",
    re.IGNORECASE,
)
ANTWEB_RE = re.compile(
    r"^(https://static\.antweb\.org/images/.+?_(h|p|d|v)(?:_(\d+))?)_(low|med|high|thumbview)"
    r"(\.(?:jpg|jpeg|png))$",
    re.IGNORECASE,
)
ANTWEB_VIEW_MAP = {
    "h": "head",
    "p": "profile",
    "d": "dorsal",
    "v": "ventral",
}


def normalize_license(raw: str | None) -> str | None:
    """Return a clean short license code for permissive licenses, or None to exclude."""
    if not raw:
        return None
    s = raw.lower()
    if "no-deriv" in s or "by-nd" in s or "by-nc-nd" in s or "all-rights" in s or "restricted" in s:
        return None
    if "zero" in s or "cc0" in s:
        return "CC0-1.0"
    if "by-nc-sa" in s:
        return "CC-BY-NC-SA-4.0" if "4.0" in s else "CC-BY-NC-SA"
    if "by-nc" in s:
        return "CC-BY-NC-4.0" if "4.0" in s else "CC-BY-NC"
    if "by-sa" in s:
        return "CC-BY-SA-4.0" if "4.0" in s else "CC-BY-SA"
    if "creativecommons.org/licenses/by/" in s or "cc by" in s:
        return "CC-BY-4.0" if "4.0" in s else "CC-BY"
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
                    ident = (m.get("identifier") or "").strip()
                    match = INAT_RE.match(ident)
                    if not match:
                        continue
                    lic = normalize_license(m.get("license") or occ_lic)
                    if not lic:
                        continue
                    base_url, photo_id, ext = match.group(1), match.group(2), match.group(3)
                    thumb_url = f"{base_url}/small{ext}"
                    if thumb_url in seen_thumbs:
                        continue
                    seen_thumbs.add(thumb_url)
                    items.append(
                        {
                            "id": f"inat_{photo_id}",
                            "source": "inat",
                            "view": "field",
                            "thumb_url": thumb_url,
                            "full_url": ident,
                            "fallback_full_url": f"{base_url}/large{ext}",
                            "species": species,
                            "genus": genus,
                            "country": country,
                            "lat": lat,
                            "lon": lon,
                            "creator": (m.get("creator") or occ.get("recordedBy") or "").strip(),
                            "license": lic,
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
    # Spread across spaced offset windows in GBIF's ~148k AntWeb occurrences for genus diversity.
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
                    ident = (m.get("identifier") or "").strip()
                    title = (m.get("title") or "").lower()
                    if "_l_" in ident.lower() or "label" in title:
                        continue
                    match = ANTWEB_RE.match(ident)
                    if not match:
                        continue
                    lic = normalize_license(m.get("license") or occ_lic)
                    if not lic:
                        continue
                    prefix, view_code, _, _, ext = match.groups()
                    thumb_url = f"{prefix}_thumbview{ext}"
                    if thumb_url in seen_thumbs:
                        continue
                    seen_thumbs.add(thumb_url)
                    stem = prefix.rsplit("/", 1)[-1]
                    items.append(
                        {
                            "id": f"antweb_{stem}",
                            "source": "antweb",
                            "view": ANTWEB_VIEW_MAP.get(view_code.lower(), "specimen"),
                            "thumb_url": thumb_url,
                            "full_url": f"{prefix}_high{ext}",
                            "fallback_full_url": f"{prefix}_med{ext}",
                            "species": species,
                            "genus": genus,
                            "country": country,
                            "lat": lat,
                            "lon": lon,
                            "creator": (m.get("creator") or occ.get("recordedBy") or "").strip(),
                            "license": lic,
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


def build_catalog(
    output_path: Path,
    inat_target: int = 6500,
    antweb_target: int = 3500,
) -> dict:
    """Query GBIF REST API for iNaturalist and AntWeb ant images and write JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen_thumbs: set[str] = set()
    headers = {"User-Agent": "badapple-ants-proto/0.1"}

    with httpx.Client(headers=headers, follow_redirects=True) as client:
        inat_items = _fetch_inat(client, inat_target, seen_thumbs) if inat_target > 0 else []
        antweb_items = _fetch_antweb(client, antweb_target, seen_thumbs) if antweb_target > 0 else []

    all_items = inat_items + antweb_items
    partial = output_path.with_suffix(".jsonl.part")
    with partial.open("w", encoding="utf-8") as f:
        for item in all_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    partial.replace(output_path)

    sources = Counter(it["source"] for it in all_items)
    species_set = {it["species"] for it in all_items if it["species"]}
    genera_set = {it["genus"] for it in all_items if it["genus"]}
    stats = {
        "total": len(all_items),
        "sources": dict(sources),
        "unique_species": len(species_set),
        "unique_genera": len(genera_set),
    }
    print(
        f"catalog complete: {stats['total']} items "
        f"(inat={sources.get('inat', 0)}, antweb={sources.get('antweb', 0)}), "
        f"{stats['unique_species']} species, {stats['unique_genera']} genera",
        flush=True,
    )
    return stats
