"""One resumable retry of higher-scoring views using the finalized retrieval path."""

import argparse, json, os, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from .cuda_search import OUTPUT
from .native_imagery import worker
from .pipeline import read_records, write_json


def attempt(record, root, baseline):
    try:
        primary = dict(record, alternatives=[])
        return dict(frame=record["frame"], record=worker(primary, root, baseline), status="retrieved")
    except Exception as exc:
        return dict(frame=record["frame"], status="unavailable", error_type=type(exc).__name__)


def run(output=OUTPUT, workers=48, precomplete_limit=None, merge_only=False):
    output = Path(output)
    settings = json.loads((output / "run.json").read_text())
    metric = (
        "structure_objective"
        if "structure_objective" in settings
        else "precision_objective" if "precision_objective" in settings else "native_objective"
    )
    current = read_records(output / ("frames.partial.jsonl" if precomplete_limit else "frames.jsonl"))
    matches = read_records(output / "matches.jsonl")
    current_by_frame = {r["frame"]: r for r in current}
    previous = (
        {r["frame"]: r for r in read_records(output / "retry-primary.partial.jsonl")}
        if (output / "retry-primary.partial.jsonl").is_file()
        else {}
    )
    pending = [
        r
        for r in matches
        if r["frame"] in current_by_frame
        and (precomplete_limit is None or r["frame"] < precomplete_limit)
        and r[metric] > current_by_frame[r["frame"]][metric] + 1e-7
        and r["frame"] not in previous
    ]
    if merge_only:
        pending = []
    started = time.perf_counter()
    os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", RAYON_NUM_THREADS="1", GDAL_NUM_THREADS="1")
    with (
        ProcessPoolExecutor(max_workers=workers) as pool,
        (output / "retry-primary.partial.jsonl").open("a", encoding="utf-8") as sink,
    ):
        futures = [pool.submit(attempt, r, str((output / "imagery").resolve()), settings["baseline"]) for r in pending]
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            previous[result["frame"]] = result
            sink.write(json.dumps(result, ensure_ascii=False) + "\n")
            sink.flush()
            print(
                f'primary real-photo retry: {i+1}/{len(pending)}, {result["status"]}; {time.perf_counter()-started:.1f}s',
                flush=True,
            )
    if precomplete_limit:
        write_json(
            output / "retry-early-report.json",
            dict(
                attempted=len(previous),
                frame_limit=precomplete_limit,
                policy="independent early completed frames only; the main imagery ledger is left to its writer",
                elapsed_seconds=time.perf_counter() - started,
            ),
        )
        return
    changed = []
    for i, old in enumerate(current):
        proposed = previous.get(i, {}).get("record")
        if proposed and proposed[metric] > old[metric] + 1e-7:
            current[i] = dict(proposed, retrieval_retry_improved=True)
            changed.append(i)
    encoded = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in current)
    (output / "frames.jsonl").write_text(encoded, encoding="utf-8")
    (output / "frames.partial.jsonl").write_text(encoded, encoding="utf-8")
    write_json(
        output / "retry-primary-report.json",
        dict(
            attempted=len(previous),
            improved_frames=changed,
            merge_only=merge_only,
            policy="retry the better primary once; retain every already retrieved view unless its exact original-resolution objective improves",
            elapsed_seconds=time.perf_counter() - started,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=OUTPUT)
    p.add_argument("--workers", type=int, default=48)
    p.add_argument("--precomplete-limit", type=int)
    p.add_argument("--merge-only", action="store_true")
    a = p.parse_args()
    run(a.output, a.workers, a.precomplete_limit, a.merge_only)
