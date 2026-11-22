# Bad Apple!! Ants

A GPU-accelerated Python prototype that matches every frame of the *Bad Apple!!* shadow-art animation to real ant photographs sourced from GBIF (iNaturalist Research-Grade observations and AntWeb museum specimens).

Built as a fork and adaptation of the Moon pipeline (`badapple_moon` and `cuda/moon`) from [**ikuto32/bad-apple-earth-moon**](https://github.com/ikuto32/bad-apple-earth-moon) by **ikuto32** (MIT License).

## How It Works

1. **Low-resolution catalog & thumbnail pack:** Either queries the GBIF REST API (`taxonKey=4342`, `Formicidae`) or streams a local GBIF Darwin Core Archive (`catalog-dwca`) for species-balanced, derivative-permitting iNaturalist field photos, AntWeb specimen views, Harvard MCZ IIIF images, and Smithsonian NMNH specimens (requiring genus- or species-level identification to exclude misclassified `Formicidae`-only museum bycatch, and always excluding labels and `ND` licenses). Downloads lightweight thumbnails into sharded directories and packs proportional `192x192` grayscale tensors into a memory-mapped cache (`cache/pack/`).
2. **GPU coarse search, refinement & CLIP ant-presence gate:** Runs an ant-first `32x24` normalized cross-correlation (NCC) search on CUDA in VRAM-bounded chunks across 16 rotations, 3 crop scales (`65%–94%` of max fit width, `--min-scale`), center-anchored shifts (`--max-center-offset 0.14` keeping crops near the subject center), horizontal mirroring, scale-invariant Laplacian anti-bokeh sharpness filtering (`--min-sharpness 0.25`), and runtime skipping of family-only (`genus == species == ""`) entries, followed by 5 rounds of `64x48` coordinate-neighborhood refinement on the top-32 candidates per distinct frame and an optional GPU OpenCLIP `ViT-B-32` ant-presence gate (`--min-ant-score 0.50`, pass `0` to disable) rejecting empty wall/background crops, flash glare, fingers, and labels.
3. **Anti-repetition assignment & high-res render:** Selects timeline matches with configurable cooldown and max-reuse limits (`cooldown=15`, `max_reuse=6`), downloads high-resolution originals only for the winning images, and renders a `1280x720` video with original audio, stretched grayscale main panel + color sidebar preview (`--main-mode grayscale-stretch` default, `grayscale`, or `color`), live NCC sparkline, species metadata, and full per-image attributions.

## Requirements & Setup

- Python 3.12 (provisioned automatically by [`uv`](https://docs.astral.sh/uv/))
- NVIDIA GPU with CUDA drivers (uses PyTorch `cu126` wheels with bundled CUDA runtime; runs comfortably within 4 GB VRAM; no local `nvcc`, Rust, or system `ffmpeg` installation required)

```powershell
uv sync --python 3.12
uv run pytest -v
```

## Usage

Run the full prototype pipeline end-to-end (defaults: `6,500` iNaturalist + `3,500` AntWeb records, `25s` clip starting at `0:25`):

```powershell
uv run python -m badapple_ants run-all
```

Or run individual stages:

```powershell
uv run python -m badapple_ants fetch-media
uv run python -m badapple_ants catalog --inat 6500 --antweb 3500
# Or stream a local GBIF Darwin Core Archive (250k default, or pass higher quotas + --max-per-occ 0 for all views):
uv run python -m badapple_ants catalog-dwca --inat 150000 --antweb 85000 --mcz 10000 --nmnh 5000
uv run python -m badapple_ants download-thumbs --concurrency 32
uv run python -m badapple_ants search --start 25 --duration 25 --cooldown 15 --max-reuse 6 --min-scale 0.65 --min-sharpness 0.25 --max-center-offset 0.14 --min-ant-score 0.50 --dedup-l1 1.2
# Or search the full video from 0:00 to EOF by passing --start 0 --duration 0:
uv run python -m badapple_ants search --start 0 --duration 0 --output output/full
uv run python -m badapple_ants fetch-winners
uv run python -m badapple_ants render --main-mode grayscale-stretch
```

## Workspace Layout

| Path | Purpose |
| --- | --- |
| `badapple_ants/geometry.py` | Shared `(cx_frac, cy_frac, w_frac, theta, mirrored)` crop math, bounds clamping, PyTorch `grid_sample` grids, and Pillow `AFFINE` rendering |
| `badapple_ants/media.py` | Pinned download & SHA-256 verification for `badapple.mp4` and Noto fonts, plus `30000/1001` fps frame extraction via `imageio-ffmpeg` (`--duration 0` for full video) |
| `badapple_ants/catalog.py` | GBIF REST API and streaming DwC-A (`catalog-dwca`) catalog builder across iNaturalist, AntWeb, Harvard MCZ, and Smithsonian NMNH (filters labels, `ND`, and `--commercial-only`) |
| `badapple_ants/download.py` | Per-host throttled async thumbnail downloader (sharded cache), low-RAM memmap `192x192` packer, and high-res winner fetcher |
| `badapple_ants/ant_gate.py` | Batched GPU OpenCLIP (`ViT-B-32`) ant-presence gate rejecting empty backgrounds, glare, bokeh, hands, and labels |
| `badapple_ants/search.py` | Memory-mapped / VRAM-chunked CUDA `32x24` NCC coarse search (chunk prefetch, `_generate_poses_batch_gpu`, `96x96`-pooled `_sample_patches_grouped`, `QUERY_TILE=2048` query tiling), consecutive near-dup frame merging (`--dedup-l1`), sharpness & center-offset bounds, family-only filtering, `64x48` refinement, CLIP gating, and cooldown/reuse timeline selector |
| `badapple_ants/render.py` | `1280x720` composite H.264/AAC renderer (with multi-script font fallback and parallel Pillow crop prefetch), diagnostic still exporter, and `CREDITS.txt` generator |
| `badapple_ants/__main__.py` | CLI entry point (`python -m badapple_ants`) |
| `tests/test_geometry.py`, `tests/test_catalog_download.py`, `tests/test_render_fonts.py`, `tests/test_search_filters.py` | Unit tests for GPU/Pillow alignment, license & label filters, DwC-A import, sharded cache packing, chunked refinement, font fallback, and anti-bokeh sharpness/scale bounds |
| `PLANS.md` | Roadmap for grayscale main rendering, flash stabilization, anti-bokeh filtering, and full-video scaling |
| `assets/`, `cache/`, `output/` | Local downloaded inputs, thumbnail/full-res caches, and rendered prototype outputs (gitignored) |

## Prototype Outputs (`output/proto/`)

- `badapple-ants-proto.mp4` — `1280x720` (`29.97` fps) H.264 + AAC video combining the matched `960x720` ant crop and `320x720` telemetry sidebar
- `stills/` — diagnostic PNG frames (`10%`, `35%`, `65%`, `90%`, best NCC, and worst NCC)
- `summary.json` — NCC quantiles (`32x24` and `64x48`), source distribution, and unique image/species counts
- `CREDITS.txt` — per-image species, creator, license, and GBIF occurrence URLs for every photograph used

## Credits and Licenses

- **Original project and architecture:** **ikuto32** — [**ikuto32/bad-apple-earth-moon**](https://github.com/ikuto32/bad-apple-earth-moon) (MIT License). The two-stage low-res GPU NCC search, local pose refinement, and high-res winner crop pipeline in this repository directly adapt the approach pioneered in `badapple_moon` and `cuda/moon`.
- **Source animation and music:** *Bad Apple!!* (Alstroemeria Records, ZUN / Team Shanghai Alice, shadow PV by Anira). Downloaded locally only for frame matching and audio muxing; not redistributed in this repository.
- **Ant photographs:** Sourced through [GBIF](https://www.gbif.org/) (`GBIF.org (20 November 2026) GBIF Occurrence Download https://doi.org/10.15468/dl.dyc97c`) from [iNaturalist](https://www.inaturalist.org/) Research-Grade observations, [AntWeb](https://www.antweb.org/) (California Academy of Sciences), and other GBIF data publishers, filtered to derivative-permitting licenses (`CC0-1.0`, `CC-BY-4.0`, `CC-BY-SA-4.0`, `CC-BY-NC-4.0`, `CC-BY-NC-SA-4.0`). Full image-level credits are generated in `output/proto/CREDITS.txt`.
- **UI fonts:** Google Noto Sans and Noto Sans Mono (SIL Open Font License 1.1).
