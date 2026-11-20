"""Async thumbnail and full-resolution downloader plus packed uint8 cache builder."""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
from urllib.parse import urlparse

import httpx
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm


def _ext_from_url(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
        return ".jpg" if suffix == ".jpeg" else suffix
    return ".jpg"


def _thumb_path(cache_dir: Path, item: dict) -> Path:
    ext = _ext_from_url(item["thumb_url"])
    return cache_dir / "thumbs" / item["source"] / f"{item['id']}{ext}"


def _full_path(cache_dir: Path, item: dict) -> Path:
    ext = _ext_from_url(item["full_url"])
    return cache_dir / "full" / item["source"] / f"{item['id']}{ext}"


async def _fetch_one(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    url: str,
    dest: Path,
) -> bool:
    if dest.is_file() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    async with sem:
        for attempt in range(3):
            try:
                resp = await client.get(url, timeout=20.0)
                if resp.status_code in (403, 404, 410):
                    return False
                if resp.status_code == 429 or resp.status_code >= 500:
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue
                resp.raise_for_status()
                data = resp.content
                with Image.open(io.BytesIO(data)) as im:
                    im.verify()
                partial = dest.with_suffix(dest.suffix + ".part")
                partial.write_bytes(data)
                partial.replace(dest)
                return True
            except Exception:
                if attempt == 2:
                    return False
                await asyncio.sleep(0.5 * (attempt + 1))
    return False


async def _download_thumbs_async(items: list[dict], cache_dir: Path, concurrency: int) -> int:
    sem = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency + 8, max_keepalive_connections=concurrency)
    headers = {"User-Agent": "badapple-ants-proto/0.1"}
    ok_count = 0
    async with httpx.AsyncClient(headers=headers, follow_redirects=True, limits=limits) as client:
        tasks = [
            _fetch_one(client, sem, item["thumb_url"], _thumb_path(cache_dir, item))
            for item in items
        ]
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="thumbs", unit="img"):
            if await coro:
                ok_count += 1
    return ok_count


def pack_thumbs(
    items: list[dict],
    cache_dir: Path,
    canvas_size: int = 192,
) -> dict:
    """Validate downloaded thumbnails and pack resized grayscale arrays into cache/pack/."""
    pack_dir = cache_dir / "pack"
    pack_dir.mkdir(parents=True, exist_ok=True)

    valid_items: list[dict] = []
    dims_list: list[tuple[int, int]] = []
    grays: list[np.ndarray] = []
    skipped = 0

    for item in tqdm(items, desc="pack", unit="img"):
        p = _thumb_path(cache_dir, item)
        if not p.is_file():
            skipped += 1
            continue
        try:
            with Image.open(p) as raw:
                img = ImageOps.exif_transpose(raw).convert("L")
                orig_w, orig_h = img.size
                if min(orig_w, orig_h) < 64:
                    skipped += 1
                    continue
                ratio = orig_w / float(orig_h)
                if ratio < 0.35 or ratio > 2.85:
                    skipped += 1
                    continue
                if orig_w >= orig_h:
                    new_w = canvas_size
                    new_h = max(1, int(round(orig_h * canvas_size / orig_w)))
                else:
                    new_h = canvas_size
                    new_w = max(1, int(round(orig_w * canvas_size / orig_h)))
                resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                arr = np.asarray(resized, dtype=np.uint8)
                if float(arr.std()) < 8.0:
                    skipped += 1
                    continue
                canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
                canvas[:new_h, :new_w] = arr
                grays.append(canvas)
                dims_list.append((new_w, new_h))
                entry = dict(item)
                entry["thumb_path"] = str(p)
                entry["pack_dims"] = [new_w, new_h]
                valid_items.append(entry)
        except Exception:
            skipped += 1

    if not valid_items:
        raise RuntimeError("No valid thumbnails survived packing filters")

    gray_arr = np.stack(grays, axis=0)
    dims_arr = np.asarray(dims_list, dtype=np.int32)
    np.save(pack_dir / "gray.npy", gray_arr, allow_pickle=False)
    np.save(pack_dir / "dims.npy", dims_arr, allow_pickle=False)
    (pack_dir / "items.json").write_text(
        json.dumps(valid_items, ensure_ascii=False), encoding="utf-8"
    )
    meta = {
        "count": len(valid_items),
        "skipped": skipped,
        "canvas_size": canvas_size,
    }
    (pack_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"packed {len(valid_items)} thumbnails ({skipped} skipped) into {pack_dir}", flush=True)
    return meta


def download_thumbs(
    catalog_path: Path,
    cache_dir: Path,
    canvas_size: int = 192,
    concurrency: int = 24,
) -> dict:
    """Download thumbnails from catalog.jsonl and build packed search cache."""
    items = [
        json.loads(line)
        for line in catalog_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ok = asyncio.run(_download_thumbs_async(items, cache_dir, concurrency))
    print(f"downloaded/verified {ok}/{len(items)} thumbnails", flush=True)
    return pack_thumbs(items, cache_dir, canvas_size=canvas_size)


async def _download_winners_async(
    unique_items: list[dict],
    cache_dir: Path,
    concurrency: int,
) -> dict[str, str]:
    sem = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency + 4, max_keepalive_connections=concurrency)
    headers = {"User-Agent": "badapple-ants-proto/0.1"}
    resolved: dict[str, str] = {}

    async def get_best(client: httpx.AsyncClient, item: dict) -> tuple[str, str]:
        dest = _full_path(cache_dir, item)
        if await _fetch_one(client, sem, item["full_url"], dest):
            return item["id"], str(dest)
        fb = item.get("fallback_full_url") or ""
        if fb:
            fb_dest = cache_dir / "full" / item["source"] / f"{item['id']}_fb{_ext_from_url(fb)}"
            if await _fetch_one(client, sem, fb, fb_dest):
                return item["id"], str(fb_dest)
        return item["id"], str(_thumb_path(cache_dir, item))

    async with httpx.AsyncClient(headers=headers, follow_redirects=True, limits=limits) as client:
        tasks = [get_best(client, it) for it in unique_items]
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="full-res", unit="img"):
            item_id, path_str = await coro
            resolved[item_id] = path_str
    return resolved


def download_winners(
    matches_path: Path,
    cache_dir: Path,
    concurrency: int = 12,
) -> dict[str, str]:
    """Download full-resolution photos for unique winning matches."""
    payload = json.loads(matches_path.read_text(encoding="utf-8"))
    unique_map: dict[str, dict] = {}
    for fr in payload["frames"]:
        item = fr["item"]
        unique_map[item["id"]] = item
    unique_items = list(unique_map.values())
    resolved = asyncio.run(_download_winners_async(unique_items, cache_dir, concurrency))
    manifest_path = cache_dir / "full" / "resolved.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(resolved, indent=2), encoding="utf-8")
    full_hits = sum(1 for p in resolved.values() if "thumbs" not in Path(p).parts)
    print(f"resolved {len(resolved)} winners ({full_hits} high-res)", flush=True)
    return resolved
