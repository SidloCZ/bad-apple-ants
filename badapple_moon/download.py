from __future__ import annotations

import concurrent.futures as futures
import hashlib
import os
import time
import urllib.request
from pathlib import Path

CHUNK = 16 << 20


def _retry(call, tries: int = 8):
    for attempt in range(tries):
        try:
            return call()
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(1 + 2 * attempt)
    raise RuntimeError("unreachable")


def _size(url: str) -> int:
    def head() -> int:
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=60) as response:
            return int(response.headers["Content-Length"])

    return _retry(head)


def _fetch(url: str, start: int, end: int) -> bytes:
    def get() -> bytes:
        request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end - 1}"})
        with urllib.request.urlopen(request, timeout=120) as response:
            data = response.read()
        if len(data) != end - start:
            raise IOError("short read")
        return data

    return _retry(get)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, path: Path, sha256: str | None = None, workers: int = 24) -> None:
    """Parallel ranged download; a sidecar bitmap of finished chunks makes it resumable.

    With ``sha256`` the finished file is verified before it replaces ``path``.
    """
    done_path = path.with_suffix(path.suffix + ".chunks")
    part = path.with_suffix(path.suffix + ".part")
    if path.exists():
        # A bitmap without a .part file is left over from an interrupted final rename.
        if done_path.exists() and not part.exists():
            done_path.unlink()
        if not done_path.exists():
            return
    path.parent.mkdir(parents=True, exist_ok=True)
    size = _size(url)
    count = (size + CHUNK - 1) // CHUNK
    if not part.exists() or part.stat().st_size != size:
        with part.open("wb") as stream:
            stream.truncate(size)
        done_path.unlink(missing_ok=True)  # the bitmap only describes the .part it was written with
    done = bytearray(done_path.read_bytes()) if done_path.exists() else bytearray(count)
    if len(done) != count:
        raise RuntimeError(f"{done_path} does not match the remote size; delete it and {part} to restart")
    started = time.time()
    fetched = 0
    with part.open("r+b") as stream, futures.ThreadPoolExecutor(workers) as pool:
        jobs = {pool.submit(_fetch, url, i * CHUNK, min(size, (i + 1) * CHUNK)): i for i in range(count) if not done[i]}
        try:
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
        except BaseException:
            pool.shutdown(cancel_futures=True)  # finished chunks stay recorded for the next run
            raise
    print()
    if sha256:
        actual = _sha256(part)
        if actual != sha256:
            part.unlink()
            done_path.unlink()
            raise RuntimeError(f"{path.name}: SHA-256 {actual} does not match the pinned {sha256}")
    os.replace(part, path)
    done_path.unlink()
