# Plans & Roadmap — Bad Apple!! Ants

Planning notes and next steps following the initial GPU prototype (`8,989` thumbnails, `751` frames / `25s` clip, median `64x48` NCC `0.789`).

Forked from upstream reference repository [`ikuto32/bad-apple-earth-moon`](https://github.com/ikuto32/bad-apple-earth-moon).

---

## 1. Rendering & Visual Readability (Top Priority)

- [x] **Swap main and sidebar photo modes (grayscale main / color in sidebar)**
  - **Implemented:** `badapple_ants/render.py` (`--main-mode grayscale` default, `grayscale-stretch` 1%–99% percentile tone-mapping, and `color` legacy mode) now renders the `960x720` main panel in grayscale and moves the matched color crop to the `288x216` `COLOR MATCH` sidebar preview.
- [ ] **Flash and luminance stabilization**
  - Inspect `cuda/video_flash/` in upstream `ikuto32/bad-apple-earth-moon` and add temporal luminance smoothing / flash-limiting across hard cuts where needed so sudden bright-dark swings remain comfortable to watch.
- [x] **Optional sidebar telemetry enhancements**
  - Add a mini world map plotting specimen/observation coordinates (`lat`/`lon` already preserved in `cache/catalog.jsonl`), a running visited-locations counter, and fixed-width footer alignment matching the upstream editions.

---

## 2. Matching Quality & Ant Visibility

- [x] **Tighter ant-first crop bounds and anti-bokeh filtering**
  - Raised default ant-first crop floor to `65%` (`--min-scale 0.65`, `COARSE_SCALES = (0.94, 0.795, 0.65)`) and added GPU-side `64x48` Laplacian-to-std sharpness filtering (`--min-sharpness 0.25`) in `badapple_ants/search.py` so smooth out-of-focus bokeh (`~0.08`) is rejected while keeping sharp ant silhouettes (`>= 0.48`).
  - Future option: evaluate blending a contour/boundary agreement term with `64x48` NCC (mirroring the `precision` objective in `bad-apple-earth-moon`) so silhouettes stay crisp on the edges.
- [x] **Center-anchored crop bounds & GPU CLIP ant-presence gate**
  - Anchored crops within `±14%` of image center (`--max-center-offset 0.14`), expanded coarse/refined top-K depth (`48`/`32`), and added batched GPU OpenCLIP `ViT-B-32` (`laion2b_s34b_b79k`) ant-presence scoring (`badapple_ants/ant_gate.py`, `--min-ant-score 0.50`) so crops missing ants (empty walls/shadows, flash glare, fingers, labels) are filtered out before timeline selection.
- [ ] **Curate dark-background ant photography**
  - AntWeb studio specimens naturally cover black-on-white frames, while white-on-black frames depend heavily on iNaturalist macro shots. Deliberately enrich the pool with dark-background ant photos (night flash macros, ants on dark bark/charcoal soil, pale ants/larvae/brood on dark substrates) rather than inverting image colors.
- [ ] **Tune timeline anti-repetition & continuity**
  - Benchmark `cooldown` and `max_reuse` tradeoffs once the corpus scales up, and test optional soft temporal pose/luminance continuity penalties between consecutive distinct frames.

---

## 3. Scaling to the Full *Bad Apple!!* Video (`~6,562-6,955` frames)

- [x] **Bulk dataset ingestion (Darwin Core Archive + full AntWeb)**
  - Added streaming `catalog-dwca` importer in `badapple_ants/catalog.py` (`iNaturalist` + `AntWeb` + `Harvard MCZ` IIIF + `Smithsonian NMNH` thumbnails, species-diverse bounded selection, media-level `ND` exclusion, and `--commercial-only`), sharded/memmap caching in `badapple_ants/download.py`, and VRAM-chunked GPU search in `badapple_ants/search.py`.
  - Next: generate `250,000`-image catalog, download thumbnails (`~7.7 GB` thumbs + `~9.2 GB` pack), and scale matching to the full video.
- [ ] **Search throughput & upstream engine evaluation**
  - **PyTorch CUDA vectorization & full-song batching (done):** Hoisted vectorized `_generate_poses_batch_gpu` once per `4,096`-image chunk, switched coarse sampling to zero-copy grouped `F.grid_sample` (`_sample_patches_grouped`) over `2x2` anti-aliased `96x96` canvases (`831 img/s` on `16,384` images, `735 img/s` across the full `1.68M` pack), and added `QUERY_TILE=2048` query tiling, centered-only `0.795` mid-scale pose trimming (`304 -> 176` poses/img), consecutive near-duplicate frame merging (`--dedup-l1 1.2`), background chunk prefetch, GPU solid-frame stats, and parallel Pillow crop prefetching in `badapple_ants/render.py`.
  - Remaining: evaluate adapting the native Rust/CUDA search engine (`cuda/moon/moon.cu`, `rust/moon-search/`) from upstream `ikuto32/bad-apple-earth-moon` or optional NVENC (`h264_nvenc` / `hevc_nvenc`) if faster repeat full-length runs are needed.
- [ ] **Attribution, licensing, and export**
  - Provide a clean `--commercial-only` flag filtering strictly to `CC0-1.0`, `CC-BY-4.0`, and `CC-BY-SA-4.0` (excluding `CC-BY-NC*` for commercial-compatible releases).
  - Export full-run `CREDITS.txt` and per-frame CSV/JSON metadata suitable for an interactive species/map explorer.
