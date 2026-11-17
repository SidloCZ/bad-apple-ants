"""Compare original-size target/old/new native views using actual source pixels."""

import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from badapple_coastline.cuda_backend import CudaAtlas
from badapple_coastline.cuda_search import OUTPUT, FRAMES, BASELINE, KEYS
from badapple_coastline.pipeline import read_records
from badapple_media import ui_font

p = argparse.ArgumentParser()
p.add_argument("--delivered", action="store_true")
p.add_argument("--root", type=Path, default=OUTPUT)
p.add_argument("--baseline", type=Path)
a = p.parse_args()
root = a.root
settings = json.loads((root / "run.json").read_text())
records = read_records(root / ("frames.jsonl" if a.delivered else "matches.jsonl"))
old = read_records((a.baseline or Path(settings.get("baseline", BASELINE))) / "frames.jsonl")
masses = np.load(FRAMES / "target-masses.npy")
evaluated_pixels = settings.get("evaluated_pixels", 1559520)
raw = np.memmap(FRAMES / "frames.u8", dtype=np.uint8, mode="r", shape=(len(records), 1080, 1444))
selected = []
for group in range(8):
    indices = [
        i
        for i in range(group * 900, min((group + 1) * 900, len(records)))
        if 0.15 < masses[i] / evaluated_pixels < 0.85
    ]
    if indices:
        best = max(
            indices, key=lambda i: records[i]["native_miou"] - records[i]["baseline_fullres_metrics"]["native_miou"]
        )
        selected.append(best)
selected = sorted(
    set(
        sorted(
            selected,
            key=lambda i: records[i]["native_miou"] - records[i]["baseline_fullres_metrics"]["native_miou"],
            reverse=True,
        )[:5]
        + [5292]
    )
)
geos = np.array(
    [[r[k] for k in KEYS] + [int(r.get("mirror_horizontal", False))] for i in selected for r in [old[i], records[i]]],
    np.float64,
)
with CudaAtlas() as cuda:
    masks = cuda.sample(geos, (1444, 1080))
canvas = Image.new("RGB", (990, 74 + len(selected) * 286), "#080f17")
draw = ImageDraw.Draw(canvas)
font = ui_font(16)
small = ui_font(14)
for x, title in [(8, "SOURCE SILHOUETTE"), (338, "PREVIOUS GEOGRAPHIC VIEW"), (668, "CUDA / ORIGINAL MASK")]:
    draw.text((x, 12), title, font=font, fill="#73e6d7")
draw.text(
    (8, 43),
    "All scores measured at 1444 x 1080. "
    + ("Delivered photographs." if a.delivered else "Search results before imagery fallback."),
    font=small,
    fill="#b5c0cc",
)
if "structure_objective" in settings:
    draw.text(
        (8, 60),
        "Content ROI y2..1077; fixed source letterbox excluded for both editions.",
        font=ui_font(11),
        fill="#b5c0cc",
    )
for row, i in enumerate(selected):
    y = 74 + row * 286
    target = np.uint8(raw[i] >= 128) * 255
    for col, data in enumerate([target, masks[row * 2] * 255, masks[row * 2 + 1] * 255]):
        image = Image.fromarray(data).convert("RGB").resize((320, 240), Image.Resampling.LANCZOS)
        canvas.paste(image, (col * 330 + 5, y))
    r = records[i]
    baseline = r["baseline_fullres_metrics"]
    draw.text((8, y + 244), f"Frame {i+1}, {i*1001/30000:.3f} s", font=small, fill="white")
    draw.text((338, y + 244), f"mIoU {baseline['native_miou']:.4f}", font=small, fill="white")
    draw.text(
        (668, y + 244),
        f"mIoU {r['native_miou']:.4f}  (+{r['native_miou']-baseline['native_miou']:.4f})",
        font=small,
        fill="#73e6d7",
    )
    draw.text(
        (338, y + 265),
        f"{r['latitude']:.5f}, {r['longitude']:.5f} / {r['width_km']:.2f} km",
        font=small,
        fill="#b5c0cc",
    )
path = root / ("comparison.png" if a.delivered else "search-comparison.png")
canvas.save(path)
print(path.resolve(), flush=True)
