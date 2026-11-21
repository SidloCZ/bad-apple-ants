"""Async thumbnail and full-resolution downloader plus packed uint8 cache builder."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import httpx
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

_HOST_CONCURRENCY = {
    "iiif.mcz.harvard.edu": 6,
    "collections.nmnh.si.edu": 6,
}


def _ext_from_url(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
        return ".jpg" if suffix == ".jpeg" else suffix
    return ".jpg"


def _thumb_path(cache_dir: Path, item: dict) -> Path:
    """Return sharded thumbnail path, reusing legacy unsharded prototype files if present."""
    ext = _ext_from_url(item["thumb_url"])
    filename = f"{item['id']}{ext}"
    legacy = cache_dir / "thumbs" / item["source"] / filename
    if legacy.is_file() and legacy.stat().st_size > 0:
        return legacy
    shard = hashlib.blake2b(item["id"].encode("utf-8"), digest_size=1).hexdigest()
    return cache_dir / "thumbs" / item["source"] / shard / filename


def _full_path(cache_dir: Path, item: dict) -> Path:
    ext = _ext_from_url(item["full_url"])
    return cache_dir / "full" / item["source"] / f"{item['id']}{ext}"


async def _fetch_one(
    client: httpx.AsyncClient,
    host_sems: dict[str, asyncio.Semaphore],
    default_sem: asyncio.Semaphore,
    url: str,
    dest: Path,
) -> bool:
    if dest.is_file() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    host = urlparse(url).netloc.lower()
    sem = host_sems.get(host, default_sem)
    async with sem:
        for attempt in range(3):
            try:
                resp = await client.get(url, timeout=25.0)
                if resp.status_code in (403, 404, 410):
                    return False
                if resp.status_code == 429 or resp.status_code >= 500:
                    await asyncio.sleep(1.5 * (attempt + 1))
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
    default_sem = asyncio.Semaphore(concurrency)
    host_sems = {host: asyncio.Semaphore(cap) for host, cap in _HOST_CONCURRENCY.items()}
    limits = httpx.Limits(max_connections=concurrency + 16, max_keepalive_connections=concurrency)
    headers = {"User-Agent": "badapple-ants-proto/0.1"}

    pending: list[tuple[str, Path]] = []
    cached = 0
    for item in items:
        dest = _thumb_path(cache_dir, item)
        if dest.is_file() and dest.stat().st_size > 0:
            cached += 1
        else:
            pending.append((item["thumb_url"], dest))

    print(f"thumbs: {cached}/{len(items)} already cached, fetching {len(pending)}...", flush=True)
    if not pending:
        return cached

    ok_new = 0
    queue: asyncio.Queue[tuple[str, Path] | None] = asyncio.Queue(maxsize=concurrency * 4)

    async def producer() -> None:
        for pair in pending:
            await queue.put(pair)
        for _ in range(concurrency):
            await queue.put(None)

    async with httpx.AsyncClient(headers=headers, follow_redirects=True, limits=limits) as client:
        pbar = tqdm(total=len(pending), desc="thumbs", unit="img")

        async def worker() -> None:
            nonlocal ok_new
            while True:
                pair = await queue.get()
                if pair is None:
                    queue.task_done()
                    break
                url, dest = pair
                if await _fetch_one(client, host_sems, default_sem, url, dest):
                    ok_new += 1
                pbar.update(1)
                queue.task_done()

        try:
            await asyncio.gather(producer(), *(worker() for _ in range(concurrency)))
        finally:
            pbar.close()

    return cached + ok_new


def _process_one_thumb(args: tuple[int, dict, Path, int]) -> tuple[int, np.ndarray, int, int, str] | None:
    idx, item, path, canvas_size = args
    if not path.is_file():
        return None
    try:
        with Image.open(path) as raw:
            img = ImageOps.exif_transpose(raw).convert("L")
            orig_w, orig_h = img.size
            if min(orig_w, orig_h) < 64:
                return None
            ratio = orig_w / float(orig_h)
            if ratio < 0.35 or ratio > 2.85:
                return None
            if orig_w >= orig_h:
                new_w = canvas_size
                new_h = max(1, int(round(orig_h * canvas_size / orig_w)))
            else:
                new_h = canvas_size
                new_w = max(1, int(round(orig_w * canvas_size / orig_h)))
            resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            arr = np.asarray(resized, dtype=np.uint8)
            if float(arr.std()) < 8.0:
                return None
            canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
            canvas[:new_h, :new_w] = arr
            return idx, canvas, new_w, new_h, str(path)
    except Exception:
        return None


def pack_thumbs(
    items: list[dict],
    cache_dir: Path,
    canvas_size: int = 192,
) -> dict:
    """Validate downloaded thumbnails and pack resized grayscale arrays via low-RAM memmap."""
    pack_dir = cache_dir / "pack"
    pack_dir.mkdir(parents=True, exist_ok=True)

    total = len(items)
    if total == 0:
        raise RuntimeError("Catalog is empty")

    tmp_dat = pack_dir / "gray.tmp.dat"
    tmp_mm = np.memmap(tmp_dat, dtype=np.uint8, mode="w+", shape=(total, canvas_size, canvas_size))

    valid_items: list[dict] = []
    dims_list: list[tuple[int, int]] = []
    skipped = 0
    valid_count = 0

    tasks = ((i, it, _thumb_path(cache_dir, it), canvas_size) for i, it in enumerate(items))
    workers = min(8, os.cpu_count() or 4)
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for res in tqdm(pool.map(_process_one_thumb, tasks, chunksize=64), total=total, desc="pack", unit="img"):
                if res is None:
                    skipped += 1
                    continue
                idx, canvas, new_w, new_h, path_str = res
                tmp_mm[valid_count] = canvas
                dims_list.append((new_w, new_h))
                entry = dict(items[idx])
                entry["thumb_path"] = path_str
                entry["pack_dims"] = [new_w, new_h]
                valid_items.append(entry)
                valid_count += 1
        tmp_mm.flush()

        if valid_count == 0:
            raise RuntimeError("No valid thumbnails survived packing filters")

        part_npy = pack_dir / "gray.part.npy"
        dst_mm = np.lib.format.open_memmap(
            part_npy, mode="w+", dtype=np.uint8, shape=(valid_count, canvas_size, canvas_size)
        )
        try:
            for s in range(0, valid_count, 4096):
                e = min(valid_count, s + 4096)
                dst_mm[s:e] = tmp_mm[s:e]
            dst_mm.flush()
        finally:
            if getattr(dst_mm, "_mmap", None) is not None:
                dst_mm._mmap.close()
            del dst_mm
        part_npy.replace(pack_dir / "gray.npy")
    finally:
        if getattr(tmp_mm, "_mmap", None) is not None:
            tmp_mm._mmap.close()
        del tmp_mm
        if tmp_dat.exists():
            tmp_dat.unlink()

    dims_arr = np.asarray(dims_list, dtype=np.int32)
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
    concurrency: int = 32,
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
    default_sem = asyncio.Semaphore(concurrency)
    host_sems = {host: asyncio.Semaphore(min(6, concurrency)) for host in _HOST_CONCURRENCY}
    limits = httpx.Limits(max_connections=concurrency + 4, max_keepalive_connections=concurrency)
    headers = {"User-Agent": "badapple-ants-proto/0.1"}
    resolved: dict[str, str] = {}

    async def get_best(client: httpx.AsyncClient, item: dict) -> tuple[str, str]:
        dest = _full_path(cache_dir, item)
        if await _fetch_one(client, host_sems, default_sem, item["full_url"], dest):
            return item["id"], str(dest)
        fb = item.get("fallback_full_url") or ""
        if fb:
            fb_dest = cache_dir / "full" / item["source"] / f"{item['id']}_fb{_ext_from_url(fb)}"
            if await _fetch_one(client, host_sems, default_sem, fb, fb_dest):
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
