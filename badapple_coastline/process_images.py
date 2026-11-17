"""Isolated GDAL workers with private writable caches and resumable metadata."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .pipeline import read_records, write_json


def retrieve(output: Path, cache: Path, size, interval, max_scenes, workers):
    settings = json.loads((output / "run.json").read_text(encoding="utf-8"))
    records = read_records(output / "matches.jsonl")
    count = len(records)
    workers = max(1, min(workers, count, 32))
    root = output / ".image-workers" / settings["match_run_id"][:16]
    known = {}
    paths = [output / "frames.jsonl", output / "frames.partial.jsonl"]
    paths += list(root.glob("n*/*/frames.jsonl")) + list(root.glob("n*/*/frames.partial.jsonl"))
    for path in paths:
        if not path.is_file():
            continue
        for record in read_records(path):
            if record.get("match_run_id") == settings["match_run_id"]:
                previous = known.get(record["frame"])
                score = lambda r: (
                    r.get("native_iou", r.get("iou", 0)),
                    r.get("satellite", {}).get("clear_coverage", 0),
                )
                if previous is None or score(record) > score(previous):
                    known[record["frame"]] = record
    shared = {cache.resolve()}
    shared.update(Path(p) for p in os.environ.get("COASTLINE_SHARED_WINDOW_CACHE", "").split(os.pathsep) if p)
    shared.update(Path(r["satellite"]["path"]).resolve().parent for r in known.values() if "satellite" in r)
    shared_scene = {p / "scene-cache" for p in shared if (p / "scene-cache").is_dir()}
    shared_scene.update(Path(p) for p in os.environ.get("COASTLINE_SCENE_READ_CACHE", "").split(os.pathsep) if p)
    env = os.environ.copy()
    env.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        RAYON_NUM_THREADS="1",
        GDAL_NUM_THREADS="1",
        GDAL_CACHEMAX="128",
        COASTLINE_IMAGE_CHILD="1",
        COASTLINE_SHARED_WINDOW_CACHE=os.pathsep.join(map(str, sorted(shared))),
        COASTLINE_SCENE_READ_CACHE=os.pathsep.join(map(str, sorted(shared_scene))),
    )
    jobs = []
    for index in range(workers):
        first = count * index // workers
        last = count * (index + 1) // workers
        shard = root / f"n{workers}" / f"{index:02d}"
        shard.mkdir(parents=True, exist_ok=True)
        write_json(shard / "run.json", settings)
        subset = records[first:last]
        (shard / "matches.jsonl").write_text("".join(json.dumps(r) + "\n" for r in subset), encoding="utf-8")
        ids = {r["frame"] for r in subset}
        prior = {i: r for i, r in known.items() if i in ids}
        if prior:
            text = "".join(json.dumps(prior[i]) + "\n" for i in sorted(prior))
            (shard / "frames.jsonl").write_text(text, encoding="utf-8")
            (shard / "frames.partial.jsonl").write_text(text, encoding="utf-8")
        private = cache / "workers" / settings["match_run_id"][:16] / f"n{workers}" / f"{index:02d}"
        child_env = dict(env, COASTLINE_SCENE_CACHE=str((private / "scene-cache").resolve()))
        log = (shard / "worker.log").open("w", encoding="utf-8")
        command = [
            sys.executable,
            "-u",
            "-m",
            "badapple_coastline",
            "imagery",
            "--output",
            str(shard),
            "--cache",
            str(private),
            "--workers",
            "1",
            "--max-scenes",
            str(max_scenes),
            "--size",
            f"{size[0]}x{size[1]}",
            "--interval",
            interval,
        ]
        process = subprocess.Popen(command, env=child_env, stdout=log, stderr=subprocess.STDOUT)
        jobs.append(dict(index=index, process=process, log=log, shard=shard, command=command, env=child_env, retries=0))
    errors = []
    try:
        while True:
            finished = completed = 0
            for job in jobs:
                code = job["process"].poll()
                if code is not None:
                    if code != 0 and job["retries"] < 2:
                        job["retries"] += 1
                        job["process"] = subprocess.Popen(
                            job["command"], env=job["env"], stdout=job["log"], stderr=subprocess.STDOUT
                        )
                        print(f"image process {job['index']}: resuming after exit {code}", flush=True)
                    else:
                        finished += 1
                        if code != 0 and job["index"] not in errors:
                            errors.append(job["index"])
                path = job["shard"] / "frames.partial.jsonl"
                if not path.is_file():
                    path = job["shard"] / "frames.jsonl"
                if path.is_file():
                    completed += path.read_bytes().count(b"\n")
            print(f"image processes: {completed}/{count}, finished {finished}/{workers}", flush=True)
            if finished == workers:
                break
            time.sleep(10)
    finally:
        for job in jobs:
            if job["process"].poll() is None:
                job["process"].terminate()
        for job in jobs:
            job["process"].wait()
            job["log"].close()
    if errors:
        raise RuntimeError(f"Image processes {errors} failed; resume imagery to retain completed work. Logs: {root}")
    merged = sorted([r for job in jobs for r in read_records(job["shard"] / "frames.jsonl")], key=lambda r: r["frame"])
    if [r["frame"] for r in merged] != [r["frame"] for r in records]:
        raise RuntimeError("Image process output missing, duplicated or reordered")
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in merged)
    (output / "frames.jsonl").write_text(text, encoding="utf-8")
    (output / "frames.partial.jsonl").write_text(text, encoding="utf-8")
    # All photos now exist; use the ordinary serial path to verify checksums and
    # write the full sequence's credits, capture years and final quality report.
    from .pipeline import imagery

    return imagery(output, cache, size, interval, max_scenes, workers=1)
