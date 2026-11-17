"""Sequential, resumable local production; never uploads or publishes externally."""

import argparse, json, os, subprocess, sys, time
from pathlib import Path
from .pipeline import write_json
from .precision_config import OUTPUT, DICTIONARY
from .data import digest


def verify_satellite_resume(output):
    settings = json.loads((output / "run.json").read_text())
    summary = json.loads((output / "monochrome-summary.json").read_text())
    verified = json.loads((output / "monochrome-native-verification.json").read_text())
    if not (output / "matches.jsonl").is_file():
        raise ValueError("Completed search is required to resume imagery")
    for report in [summary, verified]:
        if report["frames"] != settings["frames"] or report["match_run_id"] != settings["match_run_id"]:
            raise ValueError("Completed monochrome edition belongs to a different search")
        for key in [
            "exact_native_mask_differences",
            "regressed_miou_frames",
            "regressed_legacy_objective_frames",
            "regressed_precision_objective_frames",
            "regressed_structure_objective_frames",
        ]:
            if report.get(key, 0):
                raise ValueError("Completed monochrome verification did not pass")
    for key in ["lossless_pixel_audit", "mp4_silhouette_audit", "comparison_left_audit", "comparison_right_audit"]:
        audit = summary[key]
        if audit["frames"] != settings["frames"] or audit["decode_errors"] or audit["pixel_hash_differences"]:
            raise ValueError("Completed monochrome pixel audit did not pass")
    for filename, key in [
        ("badapple-land-water.mp4", "movie_sha256"),
        ("badapple-land-water-lossless.mkv", "lossless_sha256"),
        ("badapple-land-water-comparison.mp4", "comparison_sha256"),
    ]:
        if digest(output / filename) != summary[key]:
            raise ValueError("Completed monochrome movie changed")
    if digest(Path(settings["video_source"])) != settings["video_source_sha256"]:
        raise ValueError("Original video changed since monochrome verification")


def run(wait_dictionary=False, workers=48, include_satellite_photos=False, resume_satellite=False):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    os.environ.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", RAYON_NUM_THREADS="16", GDAL_NUM_THREADS="1")

    def stage(name, args):
        write_json(OUTPUT / "pipeline-status.json", dict(phase=name, status="running", pid=os.getpid()))
        print("precision production: " + name, flush=True)
        subprocess.run([sys.executable, "-u"] + args, check=True)

    try:
        if resume_satellite:
            if not include_satellite_photos:
                raise ValueError("Satellite resume requires --include-satellite-photos")
            verify_satellite_resume(OUTPUT)
            print("Verified completed monochrome movies; resuming saved real imagery.", flush=True)
        else:
            if wait_dictionary:
                write_json(
                    OUTPUT / "pipeline-status.json",
                    dict(phase="dictionary", status="waiting-for-running-dictionary", pid=os.getpid()),
                )
                print("Waiting for the independently running dual-profile dictionary.", flush=True)
                while not (DICTIONARY / "complete.json").is_file():
                    time.sleep(5)
            else:
                stage("dictionary", ["-m", "badapple_coastline.precision_dictionary"])
            stage("search", ["-m", "badapple_coastline.precision_search"])
            stage(
                "monochrome-verify-render-audit",
                ["-m", "badapple_coastline.precision_monochrome", "--output", str(OUTPUT)],
            )
        if not include_satellite_photos:
            write_json(
                OUTPUT / "pipeline-status.json",
                dict(phase="complete-monochrome", status="complete", new_satellite_downloads=False, pid=os.getpid()),
            )
            print("Monochrome edition completed. Satellite acquisition was not started.", flush=True)
            return
        stage(
            "imagery", ["-m", "badapple_coastline.native_imagery", "--output", str(OUTPUT), "--workers", str(workers)]
        )
        stage(
            "retry-primary",
            ["-m", "badapple_coastline.retry_native_primary", "--output", str(OUTPUT), "--workers", str(workers)],
        )
        stage("verify-render-audit", ["-m", "badapple_coastline.native_finalize", "--output", str(OUTPUT)])
        stage("comparison", ["scripts/preview_fullres_coastline.py", "--root", str(OUTPUT), "--delivered"])
        stage("publishing-assets", ["scripts/build_coastline_youtube.py", "--root", str(OUTPUT)])
        write_json(OUTPUT / "pipeline-status.json", dict(phase="complete", status="complete", pid=os.getpid()))
        print("All local production stages completed. No external upload performed.", flush=True)
    except Exception as exc:
        write_json(
            OUTPUT / "pipeline-status.json", dict(phase="failed", status="failed", error=str(exc), pid=os.getpid())
        )
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--wait-dictionary", action="store_true")
    p.add_argument("--workers", type=int, default=48)
    p.add_argument("--include-satellite-photos", action="store_true")
    p.add_argument("--resume-satellite", action="store_true")
    a = p.parse_args()
    run(a.wait_dictionary, a.workers, a.include_satellite_photos, a.resume_satellite)
