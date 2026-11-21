# Suggestions & Technical Observations

## Bugy a potenciální rizika

- [x] **Podtržítkové kódy licencí v `normalize_license` (např. `CC_BY_4_0`, `CC_BY_NC_4_0`)**
  - **Kde:** `badapple_ants/catalog.py` (`normalize_license`, řádky 52–84)
  - **Kontext:** GBIF DwC-A `occurrence.txt` používá kódy s podtržítky (`CC_BY_4_0`, `CC_BY_NC_4_0`), zatímco `normalize_license` kontroloval pouze pomlčkové varianty. Opraveno normalizací `_` na `-` a přidány regresní testy.

## UI, UX a frontend

- [x] **Fallback font pro nelatinková jména autorů (CJK znaky v `CREDITS`/sidebaru)**
  - **Kde:** `badapple_ants/render.py` (`FontChain`, `render_video`)
  - **Kontext:** Ve snímku [still_f0544_worst_ncc.png](output/proto-250k/stills/still_f0544_worst_ncc.png) se jméno autora zobrazovalo jako `□□□ | CC-BY-NC-4.0`, protože `NotoSans-Regular.ttf` obsahuje pouze latinku/cyrilici. Opraveno pomocí `FontChain` s detekcí chybějících glyfů a baseline-zarovnaným systémovým fallbackem (`msyh.ttc`, `msjh.ttc`, `YuGothR.ttc`, `malgun.ttf`, `segoeui.ttf`, `arial.ttf`, atd.).
