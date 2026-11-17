"""Comparable local-refinement benchmark on the actual source video and atlas."""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .atlas import Atlas
from .improvement import geographic_key, silhouette
from .matching import refine
from .pipeline import read_records
from .rust_backend import refine_batch
from badapple_media import Frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("output/coastline"))
    parser.add_argument("--atlas", type=Path, default=Path("cache/coastline-atlas-fine"))
    parser.add_argument("--output", type=Path, default=Path("output/coastline-rust-benchmark.json"))
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if min(args.samples, args.workers, args.repeats) < 1:
        parser.error("Sample, worker and repeat counts must be positive")
    os.environ["RAYON_NUM_THREADS"] = str(args.workers)
    settings = json.loads((args.baseline / "run.json").read_text(encoding="utf-8"))
    directory = Path(settings["frames_directory"])
    frames = Frames(directory, json.loads((directory / "frames.json").read_text(encoding="utf-8")))
    thumbs = frames.thumbnails()
    records = read_records(args.baseline / "frames.jsonl")
    atlas = Atlas(args.atlas)
    indices = np.linspace(min(150, len(records) - 1), min(6282, len(records) - 1), args.samples, dtype=int)
    tasks = [(silhouette(thumbs[i], (32, 24)), geographic_key(records[i])) for i in indices]
    rotations = (-7.5, 0.0, 7.5)
    scales = (0.85, 1.0, 1.18)

    def python(task):
        q, c = task
        return refine(atlas, q, c, rotations=rotations, scales=scales)

    def rust(task):
        q, c = task
        return refine_batch(atlas, q, [c], rotations=rotations, scales=scales)[0]

    python(tasks[0])
    rust(tasks[0])
    timings = {}
    outputs = {}
    for name, function in [("python", python), ("rust", rust)]:
        elapsed = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                outputs[name] = list(pool.map(function, tasks))
            elapsed.append(time.perf_counter() - started)
        timings[name] = float(np.median(elapsed))
    gains = [r["iou"] - p["iou"] for p, r in zip(outputs["python"], outputs["rust"])]
    if min(gains) < -1e-5:
        raise RuntimeError("Rust precision differs from the Python reference beyond tolerance")
    summary = dict(
        tasks=len(tasks),
        parallel_workers=args.workers,
        rotations=rotations,
        scales=scales,
        python_seconds=timings["python"],
        rust_seconds=timings["rust"],
        speedup=timings["python"] / timings["rust"],
        min_rust_iou_gain=min(gains),
        max_rust_iou_gain=max(gains),
        note=f"Identical real atlas, queries, seeds, angle and scale grid; median of {args.repeats} runs; one warmup excluded.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
