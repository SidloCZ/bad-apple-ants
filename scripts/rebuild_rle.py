import json
from pathlib import Path
import rasterio
from badapple_coastline.data import DOI, digest
from badapple_coastline.native_rle import build_cache

src = Path("assets/coastline").resolve()
dest = Path("cache/coastline-atlas-meta")
dest.mkdir(parents=True, exist_ok=True)
tiles = []
for p in sorted(src.glob("landWater2020_*.tif")):
    with rasterio.open(p) as ds:
        tiles.append({"file": p.name, "sha256": digest(p), "bounds": list(ds.bounds)})
    print("hashed", p.name, flush=True)
assert len(tiles) == 72
(dest / "atlas.json").write_text(
    json.dumps(
        {
            "schema": 1,
            "doi": DOI,
            "source_directory": str(src),
            "global_coverage": True,
            "missing_tiles": [],
            "note": "Tile manifest only (no search array); used to rebuild the lossless RLE cache",
            "tiles": tiles,
        },
        indent=2,
    ),
    encoding="utf-8",
)
print(build_cache(dest, Path("cache/coastline-native-rle")), flush=True)
