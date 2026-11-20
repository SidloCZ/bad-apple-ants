# Plans & Roadmap — Bad Apple!! Ants

Planning notes and next steps following the initial GPU prototype (`8,989` thumbnails, `751` frames / `25s` clip, median `64x48` NCC `0.789`).

Forked from upstream reference repository [`ikuto32/bad-apple-earth-moon`](https://github.com/ikuto32/bad-apple-earth-moon).

---

## 1. Rendering & Visual Readability (Top Priority)

- [x] **Swap main and sidebar photo modes (grayscale main / color in sidebar)**
  - **Implemented:** `badapple_ants/render.py` (`--main-mode grayscale` default, `grayscale-stretch` 1%–99% percentile tone-mapping, and `color` legacy mode) now renders the `960x720` main panel in grayscale and moves the matched color crop to the `288x216` `COLOR MATCH` sidebar preview.
- [ ] **Flash and luminance stabilization**
  - Inspect `cuda/video_flash/` in upstream `ikuto32/bad-apple-earth-moon` and add temporal luminance smoothing / flash-limiting across hard cuts where needed so sudden bright-dark swings remain comfortable to watch.
- [ ] **Optional sidebar telemetry enhancements**
  - Add a mini world map plotting specimen/observation coordinates (`lat`/`lon` already preserved in `cache/catalog.jsonl`), a running visited-locations counter, and fixed-width footer alignment matching the upstream editions.

---

## 2. Matching Quality & Ant Visibility

- [ ] **Tighter ant-first crop bounds and anti-bokeh filtering**
  - Prototype visual QA (`still_f0069_best_ncc.png`, `still_f0675_p90.png`) showed that `50-52%` crops on simple diagonal wipes or bright-on-black frames can latch onto out-of-focus background bokeh or stem stripes with no ant visible.
  - Raise the minimum crop width floor (e.g., `>= 65-75%` of `max_fit_crop_w`) and add a local sharpness / edge-energy check (such as Laplacian variance inside the candidate patch) to reject pure out-of-focus background crops.
  - Evaluate blending a contour/boundary agreement term with `64x48` NCC (mirroring the `precision` objective in `bad-apple-earth-moon`) so silhouettes stay crisp on the edges.
- [ ] **Curate dark-background ant photography**
  - AntWeb studio specimens naturally cover black-on-white frames, while white-on-black frames depend heavily on iNaturalist macro shots. Deliberately enrich the pool with dark-background ant photos (night flash macros, ants on dark bark/charcoal soil, pale ants/larvae/brood on dark substrates) rather than inverting image colors.
- [ ] **Tune timeline anti-repetition & continuity**
  - Benchmark `cooldown` and `max_reuse` tradeoffs once the corpus scales up, and test optional soft temporal pose/luminance continuity penalties between consecutive distinct frames.

---

## 3. Scaling to the Full *Bad Apple!!* Video (`~6,562-6,955` frames)

- [ ] **Bulk dataset ingestion (Darwin Core Archive + full AntWeb)**
  - Replace the prototype's GBIF REST API paging (`~9,000` items) with a single **GBIF Occurrence Download** (Darwin Core Archive) for Formicidae (`taxonKey=4342`, `mediaType=StillImage`, `1M+` records) plus full non-label AntWeb museum views (`~207k` head/profile/dorsal images).
  - Target `100,000-250,000` low-res thumbnails (`~3-6 GB` on disk) during search, fetching full-resolution originals only for the `~3,000-6,500` matched winners.
- [ ] **Search throughput & upstream engine evaluation**
  - Evaluate scaling the PyTorch CUDA sharded search vs. adapting the native Rust/CUDA search engine (`cuda/moon/moon.cu`, `rust/moon-search/`) from upstream `ikuto32/bad-apple-earth-moon` with configurable `CUDA_ARCH` and VRAM sharding.
  - Optionally support hardware NVENC (`h264_nvenc` / `hevc_nvenc`) alongside CPU `libx264` for faster full-length 1080p/1440p renders on supported GPUs.
- [ ] **Attribution, licensing, and export**
  - Provide a clean `--commercial-only` flag filtering strictly to `CC0-1.0`, `CC-BY-4.0`, and `CC-BY-SA-4.0` (excluding `CC-BY-NC*` for commercial-compatible releases).
  - Export full-run `CREDITS.txt` and per-frame CSV/JSON metadata suitable for an interactive species/map explorer.
