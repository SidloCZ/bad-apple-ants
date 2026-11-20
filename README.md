# Bad Apple!! Ants

A GPU-accelerated Python prototype that matches every frame of the *Bad Apple!!* shadow-art animation to real ant photographs sourced from GBIF (iNaturalist Research-Grade observations and AntWeb museum specimens).

Built as a fork and adaptation of the Moon pipeline (`badapple_moon` and `cuda/moon`) from [**ikuto32/bad-apple-earth-moon**](https://github.com/ikuto32/bad-apple-earth-moon) by **ikuto32** (MIT License).

## How It Works

1. **Low-resolution catalog & thumbnail pack:** Queries the GBIF REST API (`taxonKey=4342`, `Formicidae`) for permissive-license iNaturalist field photos and non-label AntWeb specimen views (`head`, `profile`, `dorsal`, `ventral`). Downloads lightweight `/small` and `_thumbview` thumbnails and packs proportional `192x192` grayscale tensors into `cache/pack/`.
2. **GPU coarse search & refinement:** Runs an ant-first `32x24` normalized cross-correlation (NCC) search on CUDA across 16 rotations, 3 crop scales (`55%–94%` of max fit width), shifts, and horizontal mirroring, followed by 5 rounds of `64x48` coordinate-neighborhood refinement on the top-16 candidates per distinct frame.
3. **Anti-repetition assignment & high-res render:** Selects timeline matches with configurable cooldown and max-reuse limits (`cooldown=15`, `max_reuse=4`), downloads high-resolution originals only for the winning images, and renders a `1280x720` video with original audio, sidebar previews, live NCC sparkline, species metadata, and full per-image attributions.

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
uv run python -m badapple_ants download-thumbs
uv run python -m badapple_ants search --start 25 --duration 25 --cooldown 15 --max-reuse 4
uv run python -m badapple_ants fetch-winners
uv run python -m badapple_ants render
```

## Workspace Layout

| Path | Purpose |
| --- | --- |
| `badapple_ants/geometry.py` | Shared `(cx_frac, cy_frac, w_frac, theta, mirrored)` crop math, bounds clamping, PyTorch `grid_sample` grids, and Pillow `AFFINE` rendering |
| `badapple_ants/media.py` | Pinned download & SHA-256 verification for `badapple.mp4` and Noto fonts, plus `30000/1001` fps frame extraction via `imageio-ffmpeg` |
| `badapple_ants/catalog.py` | GBIF REST API catalog builder for iNaturalist and AntWeb (filters labels and restrictive licenses) |
| `badapple_ants/download.py` | Async thumbnail downloader, `192x192` uint8 cache packer, and high-res winner fetcher |
| `badapple_ants/search.py` | CUDA `32x24` NCC coarse search, `64x48` refinement, and cooldown/reuse timeline selector |
| `badapple_ants/render.py` | `1280x720` composite H.264/AAC renderer, diagnostic still exporter, and `CREDITS.txt` generator |
| `badapple_ants/__main__.py` | CLI entry point (`python -m badapple_ants`) |
| `tests/test_geometry.py` | Unit tests verifying GPU grid sampling vs. Pillow affine alignment and mirror symmetry |
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
- **Ant photographs:** Sourced through [GBIF](https://www.gbif.org/) from [iNaturalist](https://www.inaturalist.org/) Research-Grade observations and [AntWeb](https://www.antweb.org/) (California Academy of Sciences), filtered to permissive licenses (`CC0-1.0`, `CC-BY-4.0`, `CC-BY-SA-4.0`, `CC-BY-NC-4.0`, `CC-BY-NC-SA-4.0`). Full image-level credits are generated in `output/proto/CREDITS.txt`.
- **UI fonts:** Google Noto Sans and Noto Sans Mono (SIL Open Font License 1.1).
