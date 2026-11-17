from __future__ import annotations

import concurrent.futures as futures
import os
import time
import urllib.request
from pathlib import Path

CHUNK = 16 << 20


def _size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        return int(response.headers["Content-Length"])


def _fetch(url: str, start: int, end: int, tries: int = 8) -> bytes:
    for attempt in range(tries):
        try:
            request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end - 1}"})
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
            if len(data) != end - start:
                raise IOError("short read")
            return data
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(1 + 2 * attempt)
    raise RuntimeError


def download(url: str, path: Path, workers: int = 24) -> None:
    """Parallel ranged download; a sidecar bitmap of finished chunks makes it resumable."""
    if path.exists() and not path.with_suffix(path.suffix + ".chunks").exists():
        return
    size = _size(url)
    count = (size + CHUNK - 1) // CHUNK
    done_path = path.with_suffix(path.suffix + ".chunks")
    done = bytearray(done_path.read_bytes()) if done_path.exists() else bytearray(count)
    part = path.with_suffix(path.suffix + ".part")
    if not part.exists():
        with part.open("wb") as stream:
            stream.truncate(size)
    started = time.time()
    fetched = 0
    with part.open("r+b") as stream, futures.ThreadPoolExecutor(workers) as pool:
        jobs = {pool.submit(_fetch, url, i * CHUNK, min(size, (i + 1) * CHUNK)): i for i in range(count) if not done[i]}
        for job in futures.as_completed(jobs):
            i = jobs[job]
            data = job.result()
            stream.seek(i * CHUNK)
            stream.write(data)
            done[i] = 1
            fetched += len(data)
            done_path.write_bytes(done)
            rate = fetched / max(time.time() - started, 1e-3) / 1e6
            print(f"\r{path.name}: {sum(done)}/{count} chunks, {rate:.1f} MB/s", end="", flush=True)
    print()
    os.replace(part, path)
    done_path.unlink()
