"""Early review of the already completed edition while the deeper search runs."""

import hashlib, json
from fractions import Fraction
from pathlib import Path
import numpy as np
from .data import digest, MASK_CREDIT
from .pipeline import read_records, write_json
from .precision_config import BASELINE, OUTPUT
from .precision_monochrome import SIZE, load_mask, encode, audit_pixels, comparison_planes


def run():
    output = OUTPUT / "baseline-review"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "baseline-monochrome-summary.json").is_file():
        return
    settings = json.loads((BASELINE / "run.json").read_text())
    records = read_records(BASELINE / "frames.jsonl")
    verification = json.loads((BASELINE / "native-verification.json").read_text())
    if verification["frames"] != len(records) or verification["native_pixel_differences"] != 0:
        raise ValueError("Baseline lacks completed native verification")
    if [r["frame"] for r in records] != list(range(len(records))):
        raise ValueError("Reordered baseline")
    source = Path(settings["video_source"])
    if digest(source) != settings["video_source_sha256"]:
        raise ValueError("Source changed")
    hashes = []
    for i, r in enumerate(records):
        path = Path(r["native_mask_path"])
        if digest(path) != r["native_mask_sha256"]:
            raise ValueError("Verified baseline mask changed")
        hashes.append(hashlib.sha256(load_mask(path).tobytes()).hexdigest())
        if (i + 1) % 500 == 0:
            print(f"verified existing baseline masks: {i+1}/{len(records)}", flush=True)
    raw = np.memmap(
        Path(settings["frames_directory"]) / "frames.u8", dtype=np.uint8, mode="r", shape=(len(records), 1080, 1444)
    )
    fps = Fraction(settings["fps"])
    movie = output / "badapple-land-water-baseline.mp4"
    comparison = output / "badapple-land-water-baseline-comparison.mp4"
    encode(
        (load_mask(r["native_mask_path"]) for r in records),
        SIZE,
        len(records),
        fps,
        source,
        movie,
        mirror_allowed=False,
    )
    main_audit = audit_pixels(movie, hashes, True)
    encode(
        comparison_planes(records, raw, False),
        (2888, 1120),
        len(records),
        fps,
        source,
        comparison,
        mirror_allowed=False,
    )
    right_audit = audit_pixels(comparison, hashes, True, "1444:1080:1444:40")
    original = [hashlib.sha256((np.uint8(r >= 128) * 255).tobytes()).hexdigest() for r in raw]
    left_audit = audit_pixels(comparison, original, True, "1444:1080:0:40", False)
    (output / "MONOCHROME-CREDITS.txt").write_text(
        MASK_CREDIT
        + "\nhttps://doi.org/10.5281/zenodo.14899246\nhttps://creativecommons.org/licenses/by/4.0/\n"
        + "Cropped, rotated and sampled from the original mask. White=land, black=water. Existing verified baseline only; no new satellite photographs acquired.\n"
        + "Underlying mask sources: ESA WorldCover, OpenStreetMap contributors (https://www.openstreetmap.org/copyright), Copernicus DEM.\n"
        + "Music and shadow animation: original Bad Apple!! material, respective creators retain their rights. Alstroemeria Records / Masayoshi Minoshima / Haruka / nomico / ZUN. Shadow PV: Anira (あにら), https://www.nicovideo.jp/watch/sm8628149 .\n",
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        """# 前版の白黒確認用

これは計算中の高精度・鏡像対応版ではなく、完成済みのcoastline-fullres-cuda-v2の実地点マスクです。
比較の基準を先に確認するために作成しました。新しい探索は別の工程で継続しています。

badapple-land-water-baseline.mp4:1444×1080、前版で採用した陸水マスクのみ。
badapple-land-water-baseline-comparison.mp4:左が元影絵、右が前版の陸水マスク。各画像は原寸。
音声は元動画からコピー。新しい写真は取得していません。
圧縮後の両動画の二値輪郭を、全フレームで元マスクと照合しています。
""",
        encoding="utf-8",
    )
    summary = dict(
        edition="verified-fullres-cuda-v2-baseline-only",
        new_precision_search_still_running=True,
        frames=len(records),
        evaluation_size=list(SIZE),
        mean_miou=verification["mean_miou"],
        new_satellite_downloads=False,
        baseline_frames_sha256=digest(BASELINE / "frames.jsonl"),
        native_verification_sha256=digest(BASELINE / "native-verification.json"),
        movie=str(movie.resolve()),
        movie_sha256=digest(movie),
        comparison_sha256=digest(comparison),
        main_audit=main_audit,
        comparison_left_audit=left_audit,
        comparison_right_audit=right_audit,
    )
    write_json(output / "baseline-monochrome-summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    run()
