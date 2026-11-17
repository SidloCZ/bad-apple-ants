# Bad Apple!! on real Earth and the real Moon

**English** | [日本語](README.ja.md)

The code behind three videos in which every frame of the *Bad Apple!!* shadow-art animation is a real place:

| Video | What every frame is |
| --- | --- |
| [Every frame is a real place on Earth (Satellite Imagery)](https://youtu.be/nw02Enk8o1I) | a piece of Earth whose land/water boundary matches the silhouette, shown as the real Sentinel-2 or Landsat photograph of that view |
| [Every frame is real land & water on Earth (OpenStreetMap)](https://youtu.be/kB3fzm_Un2g) | the same views, with the matched land/water as the main picture and an OpenStreetMap panel |
| [Every frame is a real shadow on the Moon (Kaguya / SELENE)](https://youtu.be/oe94IDXho9g) | the place on real lunar images whose light and shadow best match the frame |

Nothing is drawn or generated: each frame is one real location, only cropped, scaled, rotated, sometimes
mirrored, and tone-mapped. The map of every Earth view is at
[bad-apple-earth-map](https://github.com/ikuto32/bad-apple-earth-map) ([live site](https://bad-apple-earth-map.ikutotest.xyz/)).

## How it works

**Earth.** The 10 m global land/water mask of Kindermann (2025) is searched for views — any position,
searched widths of 2 km to 1,024 km (refined down to about 0.7 km), any rotation, optionally mirrored — whose land/water pattern matches
each frame's silhouette. Rust/Rayon and CUDA score billions of candidate views on the original 10 m pixels
(balanced mean IoU plus contour and boundary agreement). For the winner, the Sentinel-2 L2A scenes of that
exact view are read from public cloud archives and screened for clouds, snow and white-outs; Landsat 8/9
fills views Sentinel-2 cannot. The 2560×1920 video shows the photograph, the original frame, the matched
mask, a world map of every view so far and a local map. Details: [docs/satellite.md](docs/satellite.md).

**Moon.** About 240,000 Kaguya Terrain Camera images and the LROC WAC global mosaic are searched exhaustively
on the GPU (393 billion candidate views, normalized cross-correlation on 32×24 cells), refined, checked for
sensor saturation at full resolution, and cut out of the original images. The footer gives the nearest
IAU-named feature and the Sun's elevation at that spot. Details: [docs/moon.md](docs/moon.md).

## Results

The per-frame results of the published videos are in [`results/`](results), as plain text (no imagery):

| File | Contents |
| --- | --- |
| [`results/satellite/frames.csv`](results/satellite/frames.csv) | every frame: centre, width, rotation, mirroring, footprint corners, match score, photo scenes and dates, country, continent, nearest place |
| [`results/moon/frames.csv`](results/moon/frames.csv) | every frame: Kaguya image, centre, width, rotation, mirroring, NCC, named feature, Sun |
| [`results/moon/matches.json`](results/moon/matches.json) | the chosen pose of every distinct frame; renders the moon video without the search ([how](docs/moon.md#re-rendering-from-the-published-results)) |
| [`results/moon/rasters.txt`](results/moon/rasters.txt) | the 2,570 Kaguya images those poses use |

## Requirements

- Windows 10/11 (the GPU parts are built with MSVC; the PowerShell runners assume Windows)
- An NVIDIA GPU with the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (built and run on an RTX 4090 with CUDA 13.4; set `CUDA_ARCH`, e.g. `sm_86`, for other GPUs)
- Visual Studio 2022 C++ build tools, [Rust](https://rustup.rs/), Python 3.11+ (or [uv](https://docs.astral.sh/uv/)), FFmpeg/FFprobe on `PATH` (with `av1_nvenc` for the final encodes)
- For the OpenStreetMap panel: Node.js and Playwright's Chromium (`npx playwright install chromium`)
- Disk and time: the full runs download hundreds of GB and take days of GPU time. Start with the tests and a single stage.

## Getting started

```powershell
.\setup.ps1                                     # .venv with the three Python packages
.venv\Scripts\python -m badapple_media fetch    # source video and fonts (pinned, SHA-256 checked)
.venv\Scripts\python -m pytest                  # 91 tests, no downloads or GPU needed (2 skip until the
                                                #   Rust matcher is built and Natural Earth is fetched)
```

Then run an edition. Every stage writes its own directory under `output/` and resumes where it stopped.

```powershell
.\run-satellite.ps1                    # all stages, in the order the published video was built
.\run-satellite.ps1 -Stage precision   # or one stage at a time
.\run-moon.ps1                         # data → frames → verify → search → refine → crops → encode
.\run-moon.ps1 -Stage crops
```

| `run-satellite.ps1 -Stage` | Output | |
| --- | --- | --- |
| `fetch` | `assets/` | land/water mask (7.3 GB), Natural Earth, video, fonts |
| `baseline` | `output/coastline` | first global match and Sentinel-2 photos |
| `wide` | `output/coastline-wide-rust` | 6–768 km views, Rust/Rayon, lossless 10 m cache |
| `repair` | `output/coastline-wide-rust-fixed` | re-fetch white-out and snow photos |
| `landsat` | `output/coastline-wide-rust-landsat` | Landsat 8/9 where Sentinel-2 could not cover a view |
| `fullres` | `output/coastline-fullres-cuda-v2` | CUDA search at the source's full 1444×1080 |
| `precision` | `output/coastline-precision-v4` | contour-aware CUDA search with mirroring, new photos |
| `edition` | `output/coastline-precision-v4-2560x1920-30p-native-av1` | the published satellite video |
| `landwater` | `output/landwater-osm` | the land & water + OpenStreetMap video |
| `site` | `output/landwater-site/web` | data for the place-map website |

Exact reproduction has limits — satellite archives and map builds change, and one input of the precision
search came from an earlier run. [docs/satellite.md](docs/satellite.md#reproducibility-notes) lists them.

## Repository layout

| Path | |
| --- | --- |
| [`badapple_media/`](badapple_media) | shared FFmpeg I/O, cached frame extraction, fonts, pinned downloads |
| [`badapple_coastline/`](badapple_coastline) | Earth editions: data, search drivers, photo retrieval, rendering |
| [`badapple_moon/`](badapple_moon) | Moon edition: catalogue, ingest, crops, rendering |
| [`rust/`](rust) | `coastline-matcher` (shape search), `coastline-native` (lossless 10 m cache and sampler), `coastline-native-index` (coastal centres), `moon-search` (moon search driver and CPU check) |
| [`cuda/`](cuda) | `coastline` (view sampling, descriptors, FFT refinement), `moon` (pose scoring), `video_flash` (photo-panel luminance limiting); [`cuda/common.ps1`](cuda/common.ps1) builds them |
| [`tools/landwater-osm/`](tools/landwater-osm) | local tile server and MapLibre page for the OpenStreetMap panel |
| [`scripts/`](scripts) | frame preparation, CPU-vs-GPU validation, YouTube assets, `export_results.py` |
| [`tests/`](tests) | pytest suite |

## Tests and validation

```powershell
.venv\Scripts\python -m pytest
cargo test --release --manifest-path rust/coastline-native-index/Cargo.toml
```

The `scripts/validate_*.py` scripts compare the CUDA kernels with CPU references on real data
(they need the stage outputs). During clean-up for publication, the edition renderers were checked against
the original code: still frames render pixel-identical.

## License and credits

Code: [MIT](LICENSE). Data keeps its own terms — see [docs/data-and-licenses.md](docs/data-and-licenses.md)
for every source and its credit line. The *Bad Apple!!* animation and music belong to their creators
(Alstroemeria Records, ZUN / Team Shanghai Alice, shadow PV by Anira); this repository contains none of it.

AI assistance (Claude by Anthropic) was used for parts of the code.
