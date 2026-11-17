# Moon edition — every frame is a real shadow on the Moon

**English** | [日本語](moon.ja.md)

Video: [YouTube oe94IDXho9g](https://youtu.be/oe94IDXho9g) · per-frame results: [`results/moon/frames.csv`](../results/moon/frames.csv)

For each frame of *Bad Apple!!*, an exhaustive GPU search finds the place on real lunar images whose
light and shadow correlate best (normalized cross-correlation, NCC) with the silhouette. That place is
then cut out of the full-resolution image and shown. The black is real shadow: every frame is one
real location, only cropped, scaled, rotated, sometimes mirrored and contrast-stretched.

Output: `output/moon/badapple-moon.mp4`, 2560×1920, 30000/1001 fps (the source rate, so the audio stays
in sync), AV1 10-bit via NVENC, with the original audio. The surface fills a 4:3 panel on the left; a
480 px column on the right holds ORIGINAL (the source frame), MAP and LOCAL.

- **MAP** — the WAC global mosaic. A yellow box marks the current view (a minimum-size box and marker
  when it is tiny); every earlier view stays in teal, counted by `VISITED`.
- **LOCAL** — a north-up orthographic WAC map five times the view's width (at least 20 km), with the
  view drawn to scale and earlier views inside it in teal.
- **NCC graph** — the score of every frame played so far (never ahead of playback). Per pixel column,
  a thin band shows min–max and a line the mean of the last second; the current value is a yellow dot.
- **Footer** — coordinates, nearest IAU-named feature, camera, morning/evening and local Sun elevation,
  image ID and resolution, view width, rotation, mirroring and NCC. Every item has a fixed origin,
  baseline, monospaced font and fixed-width number format, so nothing shifts from frame to frame.

## Results of the published run

| | |
| --- | --- |
| Images searched | Kaguya TC ingested at ≤ 95 m/px: 30,498 single images + 208,729 stereo images (of 220,660; 9,688 flat and 2,243 low-signal images excluded), plus the WAC mosaic |
| Candidate poses | 393 billion (16 rotations, 10 shards, RTX 4090, about 5.8 million poses/s, about 19 hours) |
| Queries | 6,955 frames → 6,469 distinct; 6,460 textured × 2 (as-is and mirrored) = 12,920 |
| NCC per frame (32×24) | median 0.896, 10th percentile 0.790, 90th percentile 0.957 |
| Frames with NCC < 0.80 | 12.4 % |
| View width (TC) | median 3.7 km, 10th percentile 2.3 km, minimum 1.9 km |
| Sources used | TC 6,454 distinct frames (2,569 images; 4,028 from the stereo series), WAC 6, single-colour frames 9 |
| CPU check | best match identical for 12,920/12,920 queries; largest score difference 5.5e-4 |

## Pipeline

```powershell
./run-moon.ps1                 # data → frames → verify → search → refine → crops → encode
./run-moon.ps1 -Stage search   # each stage can be re-run or resumed on its own
```

Needs Windows, an NVIDIA GPU with the CUDA Toolkit (cuBLAS), Visual Studio C++ tools, Rust, FFmpeg and
the `.venv` from `setup.ps1`. The CUDA DLL is built for `sm_89` (RTX 40); set `CUDA_ARCH` for other GPUs.
`--size` changes the output size (4:3 or wider, e.g. `python -m badapple_moon encode --size 3840x2160`).
Crops are cut at 1944×1458 and scaled to the panel.

| Language | Part |
| --- | --- |
| Python + rasterio | catalogue, ingest, crops and UI ([`badapple_moon/`](../badapple_moon)) |
| Rust + Rayon | frame extraction, search driver, CPU verification ([`rust/moon-search/`](../rust/moon-search)) |
| CUDA C++ | pose sampling, scoring and top-k ([`cuda/moon/moon.cu`](../cuda/moon/moon.cu)) |

### Data

| Source | Use | View width |
| --- | --- | --- |
| Kaguya TC single images (USGS STAC, CC0) | from all 33,402 COGs, only the internal overview at ≤ 95 m/px, read with HTTP range requests (about 15 GB) | 2.4–27 km (8 steps) |
| Kaguya TC stereo (TC1W/TC2W, USGS STAC, CC0) | 220,660 items, ingested the same way in 3 parallel parts (`ingest-tc --part k --parts 3`, about 78 GB) | 2.4–27 km (8 steps) |
| LROC WAC global mosaic, 100 m | one 6 GB file, decimated to 800 m for the search | 38–434 km (8 steps) |
| USGS Gazetteer of Planetary Nomenclature | IAU-approved feature names (`fetch-names`) | captions |

Each TC image's geometry is a quadratic polynomial from the tangent plane at the image centre (km) to
full-resolution pixels, fitted through GDAL/PROJ (equirectangular and polar stereographic alike,
residual ≤ 0.25 px). WAC uses the exact inverse orthographic projection, so no per-latitude correction is
needed and the poles use the same code. Pixel values are stretched from each image's 1–99th percentile to
1..255 with 0 as no-data. `qa-images` excludes observations whose 1–99th percentile range is under 250 DN
(1,013 single and 2,365 stereo images): stretched, their weak signal shows detector seams, stripes and
compression blocks.

The Spectral Profiler support series (204,690 items) is not used: its strong onboard compression shows as
large blocks and stripes at full resolution. The 474 m TC global orthomosaic was also dropped: its stripes
and block gaps produced false matches.

### Search

1. **Queries** — Rust reads `ffmpeg … -pix_fmt gray` directly, removes exact duplicate frames by hash and
   area-averages each to 32×24 (and 64×48) with fractional-pixel weights. Each textured frame is used
   as-is and mirrored. The 9 single-colour frames (standard deviation < 2) are handled separately.
2. **Candidates** — view widths 0.15·√2ᵏ km, stride 1/16 of the width, 16 rotations (22.5°) per position.
   A step is skipped for an image whose data is coarser than one cell (1/32 of the width). Mirrored
   queries and rotated candidates together cover as-is, left-right, up-down and 180°.
3. **GPU** ([`moon.cu`](../cuda/moon/moon.cu)) — one block per pose. Each 32×24 cell averages 2×2 bilinear
   samples from the pyramid level that matches the cell size; cells are normalized to zero mean and unit
   norm and stored as fp16. Poses with missing data, a standard deviation under 3 (flat), or more than
   10 % of cells at the stretch ceiling (255) are invalid — the last rule keeps TC sensor saturation from
   passing as a white silhouette (without it, 387 distinct frames showed over 40 % blown-out white; with
   it, 7). cuBLAS `GemmEx` (fp16 in, fp32 accumulate) scores every query against every pose, and a top-32
   kernel (exact radix select plus bitonic merge) keeps each query's list. Rust/Rayon generates poses on
   another thread with double buffering and checkpoints every 5 minutes. With pyramids the data is 53 GB,
   so it is split into shards of ≤ 14 GB (`--shard-gb`) loaded one at a time; the top-32 lists carry over
   between shards and give the same scores as a single pass.
4. **Refinement** — from each query's top 32, up to 12 well-separated poses are refined in fp32 over a 3⁴
   neighbourhood of position, scale and rotation, halving the step six times. Scale stops one step below
   the searched width (about 1.7 km for TC) and never below the data's resolution. The best pose over
   both mirror states is chosen per frame.
5. **Single-colour frames** — taken in turn from dark (low mean and spread) and bright lists collected
   during the same search.
6. **Verification** (`moon-search verify`) — a Rust/Rayon CPU implementation is compared with the GPU.
7. **Merging** (`python -m badapple_moon merge`) — several runs over the same queries are pooled and the
   best 32×24 score wins. The published result merges four runs: the 2.4 km step and the 3.4–27 km steps at
   95 m/px, an earlier 180 m/px run with the stereo series, and the first single-image-only run. A larger
   search can push a good candidate out of a full top-32 list, so no frame scores below any input run.
   `run-moon.ps1` performs one run at the final settings.

### Rendering

Each pose is inverse-mapped onto a 1944×1458 grid and sampled with cubic interpolation from the
full-resolution COG (the overview level matching the output pixel, read by range requests) or the 100 m
WAC file; 2×2 supersampling is used only where an output pixel is coarser than a source pixel. The
same rotation and mirroring as the match are applied. Isolated no-data pixels inside TC images (−32768)
are filled from the nearest valid value before interpolation (gaps up to 64 px), otherwise cubic
overshoot shows small white squares in the shadows.

Every crop is checked at full resolution. If more than 2 % of it is at or above 2,750 DN (TC saturation)
or it has missing data, the frame's next refined candidate is used and the decision is written to
`matches.json` (`qa.rejected`). The stereo series uses different exposure settings and saturates at
about 2,850–6,300 DN, so the uniform 2,750 DN rule reliably removes those and also some merely bright
ground (it errs on the safe side).

The black point is the larger of the source image's 1st percentile (as at ingest) and the crop's 1st
percentile; the white point is the crop's 99.5th percentile but not below 20 % of the source range, so
noise inside shadows is not lifted. Gamma 1.3 hides Kaguya TC's onboard compression blocks in deep shadow.

The nearest feature is the smallest named feature containing the point, else the one whose rim is
nearest, with distance and diameter. The camera line shows morning or evening and the Sun's elevation
at the matched point, computed on a sphere from the sub-solar point at the observation time (JPL
Horizons, MOON_ME, east longitude; [`sun.py`](../badapple_moon/sun.py), cached in
`cache/moon/subsolar.json`). STAC's `view:sun_elevation` is one value per observation and barely changes
along an orbit, so it is not the Sun's elevation at the imaged place (88.56°S showed 6.0° where the true
value is 0.45°). The computation agrees with Horizons' own azimuth/elevation for surface sites within
0.01°. Some frames have the Sun just below the horizon (down to −3.6°): only high rims and peaks just past
the terminator are lit.

`python -m badapple_moon refooter` rewrites only the camera line of a finished video (no crops needed):
it renders the old text the same way, checks that it matches the decoded band, replaces it and
re-encodes to AV1 with the audio copied.

## Re-rendering from the published results

[`results/moon/matches.json`](../results/moon/matches.json) holds the chosen pose of every distinct frame
(after the QA above), so the video can be rebuilt without the search. Only the 2,570 images listed in
[`results/moon/rasters.txt`](../results/moon/rasters.txt) need to be ingested:

```powershell
.venv\Scripts\python -m badapple_media fetch
.venv\Scripts\python -m badapple_moon fetch-names
.venv\Scripts\python -m badapple_moon catalog
.venv\Scripts\python -m badapple_moon catalog --series stereo
.venv\Scripts\python -m badapple_moon ingest-tc --ids results/moon/rasters.txt
.venv\Scripts\python -m badapple_moon ingest-tc --catalog cache/moon/tc_catalog_stereo.jsonl --output cache/moon/tiles-stereo-0 --ids results/moon/rasters.txt
.venv\Scripts\python -m badapple_moon fetch-wac
.venv\Scripts\python -m badapple_moon ingest-global
mkdir output/moon; copy results/moon/matches.json output/moon/matches.json
.venv\Scripts\python -m badapple_moon crops
.venv\Scripts\python -m badapple_moon encode
```

`crops` writes its QA decisions back into the matches file it reads, hence the copy. Checked on frames
4077 and 5580 (single-image series) and 1135 and 5531 (stereo series): crops re-cut this way from freshly downloaded data
match the published video at 51–55 dB PSNR (the difference is compression).

## Credits

- Kaguya (SELENE) TC: JAXA/SELENE, via USGS Astrogeology Analysis Ready Data (CC0).
- LROC WAC: NASA/GSFC/Arizona State University.
- Feature names: IAU / USGS Gazetteer of Planetary Nomenclature.
- See [data-and-licenses.md](data-and-licenses.md) for every source and the animation's rights.
