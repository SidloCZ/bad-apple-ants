"""Decode every original frame at full size (1444x1080) into binary targets and 64x48 query features."""

import hashlib, json, subprocess, time
from pathlib import Path
import numpy as np
from PIL import Image
from badapple_coastline.data import digest
from badapple_media import ffmpeg_binary, probe

source = Path("assets/badapple.mp4").resolve()
metadata = probe(source)
sha = digest(source)
identity = hashlib.sha256(
    json.dumps(
        dict(
            sha=sha, width=metadata["width"], height=metadata["height"], fps=metadata["fps"], schema="original-size-v1"
        ),
        sort_keys=True,
    ).encode()
).hexdigest()[:20]
root = Path("cache/coastline-frames-full") / identity
root.mkdir(parents=True, exist_ok=True)
w, h = metadata["width"], metadata["height"]
if not (root / "frames.json").is_file():
    partial = root / "frames.u8.partial"
    with partial.open("wb") as sink:
        subprocess.run(
            [
                ffmpeg_binary("ffmpeg"),
                "-v",
                "error",
                "-nostdin",
                "-i",
                str(source),
                "-map",
                "0:v:0",
                "-an",
                "-pix_fmt",
                "gray",
                "-f",
                "rawvideo",
                "-fps_mode",
                "passthrough",
                "pipe:1",
            ],
            stdout=sink,
            check=True,
        )
    assert partial.stat().st_size % (w * h) == 0
    count = partial.stat().st_size // (w * h)
    if count != 6955:
        raise ValueError(f"Expected the 6,955-frame source animation, got {count} frames")
    partial.replace(root / "frames.u8")
    (root / "frames.json").write_text(
        json.dumps(
            dict(metadata, count=count, source=str(source), source_sha256=sha, width=w, height=h, spatial_resize=False),
            indent=2,
        ),
        encoding="utf-8",
    )
meta = json.loads((root / "frames.json").read_text())
n = meta["count"]
if not (root / "targets.bits").is_file():
    targets = np.memmap(root / "frames.u8", mode="r", dtype=np.uint8, shape=(n, h, w))
    words = (w * h + 31) // 32
    bits = np.memmap(root / "targets.bits.partial", mode="w+", dtype="<u4", shape=(n, words))
    features = np.lib.format.open_memmap(
        root / "query-features.npy.partial", mode="w+", dtype=np.int8, shape=(n, 64 * 48)
    )
    masses = np.zeros(n, np.int32)
    for i, image in enumerate(targets):
        binary = image >= 128
        flat = np.pad(binary.reshape(-1), (0, words * 32 - w * h))
        bits[i] = np.packbits(flat, bitorder="little").view("<u4")
        small = np.asarray(Image.fromarray(np.uint8(binary) * 255).resize((64, 48), Image.Resampling.BOX), np.float32)
        features[i] = np.rint(small / 255 * 16).astype(np.int8).reshape(-1)
        masses[i] = binary.sum()
        if (i + 1) % 1000 == 0:
            print("full-resolution targets", i + 1, "/", n, flush=True)
    bits.flush()
    features.flush()
    del bits, features, targets
    (root / "targets.bits.partial").replace(root / "targets.bits")
    (root / "query-features.npy.partial").replace(root / "query-features.npy")
    np.save(root / "target-masses.npy", masses)
(root / "fullres.json").write_text(
    json.dumps(
        dict(
            source=str(source),
            source_sha256=sha,
            frames=n,
            width=w,
            height=h,
            fps=metadata["fps"],
            target_format="binary threshold 128, all original spatial pixels, little-endian packed uint32 per frame",
            feature_shape=[64, 48],
            feature_levels=16,
            source_spatial_resize=False,
        ),
        indent=2,
    ),
    encoding="utf-8",
)
# Later stages find this directory through the pointer (its name is a hash of the source and settings).
(root.parent / "current.json").write_text(
    json.dumps(dict(directory=root.name, source_sha256=sha), indent=2), encoding="utf-8"
)
print(
    json.dumps(dict(directory=str(root.resolve()), frames=n, width=w, height=h, source_spatial_resize=False), indent=2),
    flush=True,
)
