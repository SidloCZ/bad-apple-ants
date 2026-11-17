# Data sources and their terms

The code in this repository is MIT-licensed (see [LICENSE](../LICENSE)). The data it downloads is not
part of the repository and keeps its own terms. Terms were checked at the providers' own pages before
use (2026-10-30 to 2026-11-10); check them again before you publish anything you make.

The `results/` directory holds derived per-frame records (coordinates, scene and image identifiers,
scores, place names). No imagery is included.

## Both editions

| Data | Use | Terms |
| --- | --- | --- |
| *Bad Apple!!* shadow-art animation and song | the frames to match and the soundtrack | The animation, music and lyrics belong to their creators (song: Bad Apple!! feat. nomico, Alstroemeria Records; original composition ZUN / Team Shanghai Alice; shadow PV: Anira). `python -m badapple_media fetch` downloads one public encode (the [bad-apple-lab/Bad-Apple](https://github.com/bad-apple-lab/Bad-Apple) mirror, pinned commit, SHA-256 checked) for local use only; nothing in this repository redistributes it. Follow the creators' guidelines and the rules of the site where you publish. |
| Noto Sans, Noto Sans Mono | moon captions | SIL Open Font License 1.1 (saved to `assets/licenses/Noto-OFL.txt`) |

## Satellite edition

| Data | Use | Terms and credit line |
| --- | --- | --- |
| G. E. Kindermann (2025), *Global Land / Water Mask Map with 0.3 arc sec. (10 m) Resolution*, v1, [doi:10.5281/zenodo.14899246](https://doi.org/10.5281/zenodo.14899246) | the land/water boundaries that are matched | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Credit the author, dataset, DOI and licence, and say the mask was cropped, resampled and rotated. The fetch stops if Zenodo reports a different licence. The mask is built from ESA WorldCover, OpenStreetMap and Copernicus DEM; keep those acknowledgements. Water includes lakes and rivers. |
| Copernicus Sentinel-2 L2A (visual and SCL COGs via [Earth Search / AWS Open Data](https://registry.opendata.aws/sentinel-2-l2a-cogs/); some frames kept from early runs came via the [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/)) | the photographs | [Copernicus Sentinel data legal notice](https://cds.climate.copernicus.eu/licences/ec-sentinel): free reproduction, distribution and modification. Show `Contains modified Copernicus Sentinel data [year]` with the capture year(s). |
| Landsat 8/9 Collection 2 Level-2 (via the [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/) STAC) | photographs for views Sentinel-2 could not cover | [Public domain](https://www.usgs.gov/faqs/are-landsat-data-cloud-still-considered-be-within-public-domain). Credit `Data available from the U.S. Geological Survey.` |
| [Natural Earth](https://www.naturalearthdata.com/) 1:110m land, 1:10m populated places, admin-0 map units, disputed areas, marine areas | world map, nearest place, country and continent | Public domain. "Made with Natural Earth". Boundaries are Natural Earth's de facto view; no position on sovereignty is implied. |
| OpenStreetMap via a [Protomaps](https://protomaps.com/) planet build (land & water edition only) | the OpenStreetMap panel | Map data © OpenStreetMap contributors, [ODbL](https://www.openstreetmap.org/copyright). The basemap style (@protomaps/basemaps) and MapLibre GL JS are BSD-3-Clause; the map fonts (Noto Sans, SIL OFL) and sprites come from [protomaps/basemaps-assets](https://github.com/protomaps/basemaps-assets). |
| [ESA WorldCover](https://esa-worldcover.org/) Sentinel-2 annual RGB composites | fallback photographs in the earlier `fullres` run only; none remains in the published video, but its ending credits carry the line, and the code that fetched them is not included | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). `© ESA WorldCover project 2020/2021`. |

Pictures are only cropped, scaled, rotated, reprojected, mirrored where marked, tone-mapped and, where
one date had gaps, filled with the same place on another date. No geography is invented and no
synthetic pixels are added.

## Moon edition

| Data | Use | Terms and credit line |
| --- | --- | --- |
| Kaguya (SELENE) Terrain Camera, single-image and stereo series, via the USGS Astrogeology [STAC](https://stac.astrogeology.usgs.gov/) (Analysis Ready Data) | the lunar surface | Distributed by USGS as CC0. Credit JAXA/SELENE. |
| LRO LROC WAC global mosaic, 100 m (June 2013) | search base for wide views, MAP and LOCAL panels | NASA/GSFC/Arizona State University; USGS Astrogeology distribution. |
| [USGS Gazetteer of Planetary Nomenclature](https://planetarynames.wr.usgs.gov/) | the nearest named feature | IAU / USGS. Regenerated daily by USGS; `fetch-names` records the download date and checksum. |
| [JPL Horizons](https://ssd.jpl.nasa.gov/horizons/) | sub-solar point at each observation time, for the Sun elevation | NASA/JPL service. |
