# Satellite edition

**English** | [日本語](satellite.ja.md)

This part of the repository made "Bad Apple!! but every frame is a real place on Earth (Satellite Imagery)"
(YouTube `nw02Enk8o1I`, 2560x1920, 30p, AV1) and its land & water + OpenStreetMap edition (YouTube `kB3fzm_Un2g`).
For every frame of the Bad Apple!! shadow-art animation, the code searches the whole Earth for a real place whose
land/water boundary matches the silhouette, using G. E. Kindermann (2025), *Global Land / Water Mask Map with 0.3 arc sec.
(10 m) Resolution* ([doi:10.5281/zenodo.14899246](https://doi.org/10.5281/zenodo.14899246), CC BY 4.0). It then shows the
real Sentinel-2 (or Landsat 8/9) photograph of exactly that view. Land is white in the mask, water is black, and the
silhouette is the source frame thresholded at luminance 128. No geography is generated, warped or stitched: each frame is
one rigid view (position, width, rotation, optional mirror) of the real mask, and every photo pixel comes from a real scene.

## How the published video was built

The satellite edition was built incrementally. Each search stage starts from the previous stage's result and keeps a
frame's previous match unless the new search scores better on the original 10 m mask. The stage runner
[`run-satellite.ps1`](../run-satellite.ps1) runs the stages in the same order; each stage writes its own directory under
`output/` and resumes from completed work.

```powershell
./setup.ps1                            # .venv with the Python packages
./run-satellite.ps1                    # every stage, in order
./run-satellite.ps1 -Stage precision   # one stage; earlier stages' outputs must exist
```

| Stage | Output directory | Summary |
| --- | --- | --- |
| `fetch` | `assets/` | Mask tiles (7.3 GB), Natural Earth, source video and fonts |
| `baseline` | `output/coastline` | First global match + Sentinel-2 photos |
| `wide` | `output/coastline-wide-rust` | 6-768 km views, Rust/Rayon, lossless 10 m cache |
| `repair` | `output/coastline-wide-rust-fixed` | Re-fetch white-out / snow photos |
| `landsat` | `output/coastline-wide-rust-landsat` | Landsat 8/9 for frames Sentinel-2 could not cover |
| `fullres` | `output/coastline-fullres-cuda-v2` | CUDA search at the source's full 1444x1080 |
| `precision` | `output/coastline-precision-v4` | Contour-aware CUDA search, mirroring, new photos |
| `edition` | `output/coastline-precision-v4-2560x1920-30p-native-av1` | The published video |
| `landwater` | `output/landwater-osm` | Land & water + OpenStreetMap edition |
| `site` | `output/landwater-site/web` | Data for the place-map website (`-Site` changes the path) |

Each search is a finite, sampled search (fixed seed points, widths and angles, then local refinement). It is not a proof
of the best possible view anywhere on Earth.

## Method

### Data and search grid (`fetch`, `baseline`)

`fetch` downloads the 72 GeoTIFF tiles of the mask from Zenodo (about 7.26 GB, 1-bit, CCITT-compressed) and checks each
file's MD5 and size against the Zenodo record. Expanded, the globe is about 9.3 trillion pixels, so the 10 m data is never
held as one uncompressed array. For the first search, [`atlas.py`](../badapple_coastline/atlas.py) reduces it to a
0.01° grid (about 1.1 km at the equator, about 648 MB). [`matching.py`](../badapple_coastline/matching.py) samples
32,000 points on land/water boundaries (fixed seed) and builds 1,151,896 candidate windows of 24/48/96 km width in
30° rotation steps, with a fixed 4:3 aspect and a latitude correction of the longitude span. Candidate centres were limited
to 54°S-80°N in this stage, where optical imagery is usually available.

### Matching objective

The objective was made stricter over the stages; every later stage re-measures the previous result with its own objective
before deciding whether a frame changes.

1. `baseline`, `wide`: land IoU. Coarse candidates are ranked by small shape descriptors, then the translation is searched
   with FFT correlation.
2. `fullres`: 0.85 × balanced mIoU (mean of land IoU and water IoU) + 0.15 × boundary F1 with a 3-pixel tolerance in
   source pixels, at the full 1444x1080. Land IoU alone can look high while the dark figure is badly placed.
3. `precision`: 0.70 × mIoU + 0.25 × soft contour F1 + 0.05 × boundary F1, minus 0.10 × the largest mismatch fraction
   along the four frame borders. The soft contour term measures the L1 distance between contours in source pixels
   (truncated at 256) in both directions (missing and extra contour) and converts it with `exp(-d/8)` and `exp(-d/32)`.
   Weights are scaled to integers so the GPU sum does not depend on the order of additions. The border term discourages
   views in which, for example, the land runs out of the top of the picture while the head stays inside it.

The source frames have a fixed 2-pixel black band at the top and bottom. The precision stage scores only the
1444x1076 picture area (rows 2-1077) and extends the nearest picture row into the band for the search descriptors;
source frames are not modified. Frames that are a single colour (171 frames: 115 black, 56 white) use views that are
verified to be entirely water or entirely land.

### Rust/Rayon (`wide`)

[`rust/coastline-matcher`](../rust/coastline-matcher) moved candidate search, coordinate sampling and FFT matching to
Rust with Rayon. The search grid became 0.0025° (about 278 m at the equator; the global array is about 10.37 GB) and the
widths 6/12/24/48/96/192/384/768 km. Local refinement tries 7 rotations (±15° in 5° steps) and 7 scales (about 0.707-1.414x);
the top 12 candidates are re-read from the 10 m tiles at 260x195, and the top 4 are refined at 1.25° and 2 % steps. Over
the same 260x195 10 m evaluation, the mean IoU went from 0.7089 to 0.7473, with 6,800 frames improved and none worse.
On 512 real queries the Rust local matching took 0.106 s against 3.511 s for the Python reference (median of 3 runs);
that is the local matching step only.

### Lossless 10 m RLE cache

[`rust/coastline-native`](../rust/coastline-native) decodes each 1-bit tile row by row with libtiff and stores it as
run-length transitions (format `LWMRLE01`: width, height, per-row offsets, then the x position of each water/land
transition). The 72 tiles convert in about 66 s into about 10.64 GB. Lookups through
[`native_rle.py`](../badapple_coastline/native_rle.py) return the exact 10 m pixel without downsampling; a comparison of
512 windows (25,958,400 pixels) against GDAL reads found no difference. [`scripts/rebuild_rle.py`](../scripts/rebuild_rle.py)
rebuilds the cache.

### CUDA search (`fullres`)

[`rust/coastline-native-index`](../rust/coastline-native-index) scans every row of the RLE cache (about 2.6 billion
boundary transitions) and picks 64,000 spatially spread seeds between 80°S and 84°N. The view bank has 16 widths from
4 to 768 km and 24 rotations (15° steps): 24,576,000 views. [`cuda/coastline/coastline.cu`](../cuda/coastline/coastline.cu),
called through [`cuda_backend.py`](../badapple_coastline/cuda_backend.py), uploads the whole RLE cache to GPU memory,
samples each view at 256x192 directly from the 10 m data, pools it to a 64x48 INT8 descriptor and correlates all frames
with cuBLAS (INT8 products, INT32 sums). Descriptors are streamed, not stored. The survivors (4 spread candidates per
scale plus the previous view) are refined with cuFFT at 128x96, 256x192 and finally the full 1444x1080, with the final
pixels re-read from the RLE cache. The DLL is built with `--fmad=false` so geometry is bit-identical to the CPU path;
25,128,780 sampled pixels matched the CPU reference with 0 differences. On an RTX 4090 the bank and correlation took
215 s and the refinement of all frames 1,490.89 s. On the delivered photo views, mean balanced mIoU went from 0.7756
(previous views, re-measured at 1444x1080) to 0.8038.

### Precision search (`precision`)

[`precision_search.py`](../badapple_coastline/precision_search.py) and
[`precision_dictionary.py`](../badapple_coastline/precision_dictionary.py) widen the grid
([`precision_config.py`](../badapple_coastline/precision_config.py)): 512,000 seeds, 32 widths from 2 to 1,024 km,
5° rotations and mirrored views, 2,359,296,000 combinations in total. Descriptors are 96x72 (from 384x288 samples); the
GPU ranks them both by area and by a signed contour-gradient descriptor and keeps the top 8 per tile and scale for each.
192 candidates survive the coarse stage, 64 go to full-resolution refinement (192x144, 512x384, then 1444x1080), with up
to 3 FFT translation peaks each. Two temporal passes (forward and backward) pass candidates from frames ±1, 2, 4, 8, 16,
32 and 64 away, limited by silhouette similarity. A new view is accepted only if none of mIoU, the `fullres` objective,
the precision objective and the border-penalised objective drops below the freshly measured previous view. The stage also
writes black-and-white land/water videos, then fetches photos for changed views. On the photographed views, mean mIoU
over the 1444x1076 picture area went from 0.8070 (previous views, re-measured) to 0.8297.

### Photo retrieval

- Sentinel-2 L2A from the public [Earth Search](https://earth-search.aws.element84.com/v1) STAC and AWS Open Data COGs
  ([`satellite.py`](../badapple_coastline/satellite.py), [`native_imagery.py`](../badapple_coastline/native_imagery.py)).
  The photo has the same position, rotation and aspect as the matched view. Summer dates around 2020 for the hemisphere are
  preferred to reduce snow and sea ice. The scene classification layer (SCL) marks cloud, missing data and, from the
  `repair` stage on, snow/ice (class 11); clear pixels are chosen across several dates of the same place. Remaining cloud
  is kept as photographed and its fraction is shown. From `fullres` on, windows are read from the original COGs at
  source resolution or an overview finer than the output.
- White-out checks ([`quality.py`](../badapple_coastline/quality.py), [`repair.py`](../badapple_coastline/repair.py)):
  at 96x72, a photo is rejected if 35 % of its area has all of R, G, B ≥ 225, if 60 % has mean brightness ≥ 210, or if it
  is a bright, nearly uniform image. Rejected frames try other summer dates and then saved alternative views that do not
  score lower.
- Landsat 8/9 Collection 2 Level-2 ([`landsat.py`](../badapple_coastline/landsat.py)) from Microsoft Planetary Computer
  (`landsat-c2-l2`, raw 16-bit crops through the data API), screened with QA_PIXEL and converted with the USGS scale factor
  (DN × 0.0000275 − 0.2) and one sRGB transform for all scenes. The `landsat` stage replaced the 92 frames the `repair`
  stage had left without a photo; 91 kept their view and one (frame 5185) used a saved alternative view.
- From `fullres` on, a Sentinel-2 photo with at least 95 % real coverage can have its missing pixels filled with
  same-coordinate Landsat pixels (labelled S2+LANDSAT); the `fullres` result also used official ESA WorldCover Sentinel-2
  annual RGB composites for 26 frames (none of them remains in the published video, and the code that fetched them
  is not included, so a rerun of `fullres` gives other photos for those frames).
  [`retry_native_primary.py`](../badapple_coastline/retry_native_primary.py) retries
  the best candidate once when its photo failed at first.
- No pixel is invented: a photo is used only if real pixels cover every output pixel; missing areas are never painted.
  Mask, photo footprint and image hashes are re-checked for every frame before rendering. Credits per frame go to
  `frames.jsonl` and `CREDITS.txt`.

### Rendering the 2560x1920 edition (`edition`)

[`native_2560.py`](../badapple_coastline/native_2560.py) draws every panel at output size from lossless sources and
encodes once (output boxes as x0, y0, x1, y1):

| Panel | Box | Content |
| --- | --- | --- |
| Photo | 40,140,1960,1580 | Satellite photo, 1444x1080 source resampled once to 1920x1440 |
| Original | 2000,184,2520,574 | Source frame, decoded and resized only |
| Matched mask | 2000,650,2520,1040 | The 10 m mask of the same view |
| World map | 2000,1116,2520,1376 | Natural Earth 1:110m land; current footprint and every earlier one ([`footprint_map.py`](../badapple_coastline/footprint_map.py)) |
| Local map | 2000,1488,2520,1880 | North-up azimuthal equidistant map of the 10 m mask, at least 20 km or 5x the photo width; gold = current photo, cyan = earlier nearby views ([`local_map.py`](../badapple_coastline/local_map.py)) |

Below the photo are the nearest mapped place (Natural Earth populated places, with distance), latitude and longitude,
capture dates, view width and rotation, mIoU, imagery source and cloud fraction; the right column marks mirrored views
and the frame counter. Tone mapping
([`cuda/video_flash`](../cuda/video_flash/video_flash.cu), [`flash_cuda.py`](../badapple_coastline/flash_cuda.py)) uses
the current frame only: black and white levels from the 0.5th/99.5th luminance percentiles, gamma 0.75, mapped into
relative luminance 0.002 to 0.092 (span 0.09), saturation 1.15, and a chroma gain of at most 1 for near-black pixels so
codec noise in dark water is not turned into blue or magenta. No earlier frame's pixels are used.

Flash reduction applies to the photo panel only. After encoding, every pixel of the decoded photo panel in every frame is
screened against the WCAG 2.2 SC 2.3.1 general-flash and red-flash thresholds (conservative reference-view method in
[`flash_reduction.py`](../badapple_coastline/flash_reduction.py)); if it failed, the span would be narrowed and the video
re-encoded. The published video passed at span 0.09 (largest candidate area 1,134 px and red area 277 px against a limit
of 95,480 px). The original animation and the mask panel are not processed and can still flash, so the whole video is
not claimed to comply and the title carries a warning.

Output: 2560x1920, each of the 6,955 source frames shown once at 30.000 fps (231.833 s), AV1 `av1_nvenc` preset p7
two-pass, 10-bit, 45 Mbps target / 90 Mbps maximum, AAC 192 kbps. From frame 6562 (0-based) on, during the original's
ending credits, one fixed real view (that of frame 6570) is held. In the published video 2,731 frames are mirrored and 544
use Landsat pixels.

### Land & water + OpenStreetMap edition (`landwater`, `site`)

[`landwater_osm.py`](../badapple_coastline/landwater_osm.py) makes the matched 10 m land/water view the main picture
(1920x1440), with the original, the satellite photo of the same view, the world map and an OpenStreetMap panel on the
right. [`landwater_main.py`](../badapple_coastline/landwater_main.py) samples the RLE cache through the exact match
transform: area supersampling for wide views, a Gaussian-weighted field (σ = 0.55 cell) contoured at 0.5 for zoomed-in
views, so rotated 10 m cells do not show as stairs. Compared with the native masks on every 25th frame of the encoded
video, mean IoU is 0.9951. Some areas have a stepped coastline in the mask itself.

The OpenStreetMap panel is rendered once per view by MapLibre GL JS 5.24.0 in headless Chromium
([`osm_maps.py`](../badapple_coastline/osm_maps.py), [`tools/landwater-osm`](../tools/landwater-osm)) with a recoloured
`@protomaps/basemaps` style. Tiles come from a Protomaps planet build (PMTiles v3) read by HTTP range requests through
[`tools/landwater-osm/server.mjs`](../tools/landwater-osm/server.mjs), a dependency-free Node server that caches every tile
on disk; the public OSM tile servers do not allow bulk rendering. Footprints, scale bar and north arrow are drawn in Python
with the same Web Mercator maths.

[`geostats.py`](../badapple_coastline/geostats.py) assigns each view to a country or territory by the largest geodesic land
overlap with Natural Earth 1:10m Admin-0 map units; views with less than 2 % land take the nearest unit within 200 nautical
miles. Continent comes from Natural Earth `CONTINENT`, with Russia split at 60°E. Views overlapping by quad IoU > 0.5 form
one place. Result: 6,562 matched frames, 6,355 views, 1,680 places, 138 countries and territories, 7 continents. An
animated statistics card covers the ending credits. Its "area shown" figure on the published video (24.2 million km²)
counts overlaps repeatedly; the union of all views is 8.31 million km² (1.63 % of Earth's surface), which the website uses.

Frames are composed on a thread pool and streamed to `av1_nvenc` (10-bit, 45 Mbps). The photo panel keeps the tone window
of the satellite edition; the main panel reproduces the original animation's light/dark changes and is not
flash-reduced. `site` ([`landwater_web.py`](../badapple_coastline/landwater_web.py)) writes the website data: every view
with corners, frame ranges, country, continent, place and score, every frame, statistics, and an original-vs-mask
thumbnail per view. Timestamps follow `nw02Enk8o1I` (frame n at (n−1)/30 s); `--youtube` changes the video id.

## Reproducibility notes

- The `precision` search also merged the regional candidate lists of an earlier v3 dictionary and v3 seed views, whose
  exact code is not in this repository. The code treats that dictionary as optional (`legacy_available()` in
  [`precision_search.py`](../badapple_coastline/precision_search.py)), so a fresh run uses only the v4 dictionary and can
  choose different views.
- Satellite archives, STAC catalogues and the Protomaps daily builds change over time, so photos and maps may differ.
  The video used the Protomaps build of 2026-11-09; old builds are removed, so set `BADAPPLE_PMTILES` to a current build
  or a local extract.
- The exact per-frame results of the published video are in [`results/satellite/frames.csv`](../results/satellite/frames.csv):
  `frame`, `time_s`, `longitude`, `latitude`, `width_km`, `height_km`, `rotation_clockwise_deg`, `mirrored`, `miou`,
  `precision_objective`, the four footprint corners (`corner1_lon` … `corner4_lat`), `photo_collection`, `photo_scenes`,
  `photo_dates`, `photo_cloud_fraction`, `country`, `territory`, `continent`, `nearest_place`, `nearest_place_country`,
  `nearest_place_km`, `note`. [`scripts/export_results.py`](../scripts/export_results.py) writes it from a finished run.
- The cleaned code was checked against the original: with the original intermediate outputs, stills of frames 145, 2500
  and 6600 of the edition (`python -m badapple_coastline.native_2560 --stills 145,2500,6600`) and frames 3000 and 6790 of
  the land & water edition (`python -m badapple_coastline.landwater_osm stills --frames 3000,6790`) render pixel-identical.
- Tests are in [`tests/`](../tests) (`python -m pytest`); GPU/CPU comparisons are in `scripts/validate_*.py`.
- Requirements: Windows, an NVIDIA GPU with the CUDA Toolkit (cuBLAS, cuFFT) and MSVC, a GPU whose NVENC supports AV1
  for the final encodes, Rust, FFmpeg, and Node.js with Playwright Chromium for the OpenStreetMap panel. The published run
  used an RTX 4090. The whole RLE cache is loaded into GPU memory for the CUDA stages.

## Environment variables

| Variable | Meaning |
| --- | --- |
| `BADAPPLE_CACHE` | Cache root for OpenStreetMap tiles and land & water panels (default `cache`) |
| `BADAPPLE_PMTILES` | Protomaps PMTiles URL or local file (default the 2026-11-09 build) |
| `BADAPPLE_CHROME` | Chromium executable for MapLibre; otherwise the newest Playwright Chromium (`PLAYWRIGHT_BROWSERS_PATH` is honoured) |
| `CUDA_PATH` | CUDA Toolkit root; otherwise the newest installed toolkit |
| `CUDA_ARCH` | GPU architecture for the CUDA DLLs (default `sm_89`; e.g. `sm_86`, `sm_120`) |

`run-satellite.ps1` also sets `OMP_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2`, `RAYON_NUM_THREADS=16` and
`GDAL_NUM_THREADS=1`. GDAL runs in separate single-threaded processes.

## Disk and time

- Mask tiles: about 7.26 GB. Lossless 10 m RLE cache: about 10.64 GB. Fine 0.0025° search atlas: about 10.37 GB;
  the 0.01° atlas about 648 MB. Full-resolution frame cache (6,955 × 1444x1080 bytes): about 10.85 GB.
- Photo caches, per-frame masks and the intermediate videos of each stage come on top; check free space before a full run.
- A full run takes days of downloads and GPU time; photo retrieval is limited by the remote services. Every stage resumes
  from cached work, so an interrupted stage can be restarted with the same command.

## Data and credits

Kindermann land/water mask (CC BY 4.0; built from ESA WorldCover, OpenStreetMap and Copernicus DEM), Copernicus Sentinel-2
data ("Contains modified Copernicus Sentinel data [year]"), Landsat ("Data available from the U.S. Geological Survey", public domain),
ESA WorldCover annual composites (CC BY 4.0), OpenStreetMap data (ODbL, "© OpenStreetMap contributors") via Protomaps,
and Natural Earth (public domain). Each render writes the full notices to `CREDITS.txt`. The mask's water class includes
lakes and rivers, so a matched "coastline" may be a lake or river shore.
