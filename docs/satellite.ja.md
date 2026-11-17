# 衛星写真版

[English](satellite.md) | **日本語**

このリポジトリのこの部分は、「Bad Apple!! but every frame is a real place on Earth (Satellite Imagery)」
（YouTube `nw02Enk8o1I`、2560×1920、30p、AV1）と、その「陸と水＋OpenStreetMap」版（YouTube `kB3fzm_Un2g`）を作ったコードです。
Bad Apple!!の影絵アニメの各フレームについて、陸と水の境界がシルエットに一致する実在の場所を地球全体から探します。
探索に使うのは G. E. Kindermann (2025)『Global Land / Water Mask Map with 0.3 arc sec. (10 m) Resolution』
（[doi:10.5281/zenodo.14899246](https://doi.org/10.5281/zenodo.14899246)、CC BY 4.0）です。見つけた範囲と
まったく同じ範囲の、実際のSentinel-2（またはLandsat 8/9）の写真を表示します。マスクでは陸が白、水が黒に対応し、
シルエットは元フレームを輝度128で二値化したものです。地形の生成・変形・継ぎ合わせは行いません。各フレームは実際の
マスクの1つの剛体的な画角（位置・幅・回転・必要なら鏡像）で、写真の画素はすべて実在のシーンから取ったものです。

## 公開動画ができるまで

衛星写真版は段階的に作りました。各探索段階は前の段階の結果から始め、元の10mマスク上で新しい探索の評価が
上回ったフレームだけを置き換え、それ以外は前の結果を残します。段階実行スクリプト
[`run-satellite.ps1`](../run-satellite.ps1) は同じ順序で各段階を実行します。各段階は `output/` の下に専用の
ディレクトリを持ち、完了済みの処理を再利用して再開できます。

```powershell
./setup.ps1                            # Pythonパッケージ入りの .venv を作成
./run-satellite.ps1                    # 全段階を順に実行
./run-satellite.ps1 -Stage precision   # 1段階だけ実行（前の段階の出力が必要）
```

| 段階 | 出力先 | 内容 |
| --- | --- | --- |
| `fetch` | `assets/` | マスクのタイル（7.3 GB）、Natural Earth、元動画、フォント |
| `baseline` | `output/coastline` | 最初の全球探索とSentinel-2写真 |
| `wide` | `output/coastline-wide-rust` | 6〜768 kmの画角、Rust/Rayon、可逆の10 mキャッシュ |
| `repair` | `output/coastline-wide-rust-fixed` | 白飛び・雪の写真を再取得 |
| `landsat` | `output/coastline-wide-rust-landsat` | Sentinel-2で写真が揃わなかったフレームをLandsat 8/9で補う |
| `fullres` | `output/coastline-fullres-cuda-v2` | 元動画の原寸1444×1080でのCUDA探索 |
| `precision` | `output/coastline-precision-v4` | 輪郭を考慮したCUDA探索、鏡像、新しい写真 |
| `edition` | `output/coastline-precision-v4-2560x1920-30p-native-av1` | 公開した動画 |
| `landwater` | `output/landwater-osm` | 陸と水＋OpenStreetMap版 |
| `site` | `output/landwater-site/web` | 場所マップのサイト用データ（`-Site` で出力先を変更） |

どの探索も、決まった開始地点・幅・角度から局所的に詰める有限のサンプリング探索です。地球上のあらゆる画角の中で
最良であることを保証するものではありません。

## 手法

### データと探索格子（`fetch`、`baseline`）

`fetch` はZenodoからマスクの72枚のGeoTIFFタイル（約7.26 GB、1ビット、CCITT圧縮）を取得し、Zenodoの記録にある
MD5とサイズを照合します。展開すると全球で約9.3兆画素になるため、10 mデータを1つの非圧縮配列として持つことは
しません。最初の探索では [`atlas.py`](../badapple_coastline/atlas.py) が0.01度格子（赤道で約1.1 km、約648 MB）へ縮小し、
[`matching.py`](../badapple_coastline/matching.py) が陸と水の境界上の32,000点を固定シードで選び、幅24/48/96 km・
30度刻みの候補窓を1,151,896個作ります。縦横比は4:3に固定し、緯度に応じて経度方向の幅を補正します。
この段階では、光学画像を得やすい南緯54度〜北緯80度に候補の中心を限っていました。

### 一致度の指標

指標は段階を追うごとに厳しくしました。後の段階は、前の結果を自分の指標で測り直してから置き換えるかどうかを決めます。

1. `baseline`、`wide`：陸のIoU。小さな形状記述子で候補を絞り、平行移動はFFT相関で探します。
2. `fullres`：0.85 ×（陸IoUと水IoUの平均であるmIoU）＋ 0.15 × 境界F1（元動画の画素で3画素の許容幅）を、
   原寸1444×1080で計算します。陸IoUだけでは、黒い人物の位置がずれていても高く出ることがあるためです。
3. `precision`：0.70 × mIoU ＋ 0.25 × ソフト輪郭F1 ＋ 0.05 × 境界F1 から、画面四辺の不一致率の最大値 × 0.10 を
   引きます。ソフト輪郭は、輪郭同士のL1距離（元動画の画素単位、256で打ち切り）を両方向（欠けた輪郭と余分な輪郭）で
   測り、`exp(-d/8)` と `exp(-d/32)` で一致度に変換します。重みは整数化し、GPUでの加算順によって値が変わらない
   ようにしています。四辺の項は、たとえば頭は画面内にあるのに陸地だけが上端から抜けている、といった候補を抑えます。

元動画の上下には常に2画素の黒い帯があります。`precision` 段階では映像部分の1444×1076（2〜1077行）だけを採点し、
探索用の記述子では帯の部分を直近の映像行で延長します。元フレーム自体は変更しません。全面が単色のフレーム
（171枚：黒115、白56）には、全域が水または全域が陸であることを確認した画角を使います。

### Rust/Rayon（`wide`）

[`rust/coastline-matcher`](../rust/coastline-matcher) で、候補探索・座標サンプリング・FFT照合をRayon並列のRustへ
移しました。探索格子は0.0025度（赤道で約278 m、全球配列は約10.37 GB）、幅は6/12/24/48/96/192/384/768 kmです。
局所探索は回転7段階（±15度、5度刻み）と縮尺7段階（約0.707〜1.414倍）。上位12候補は10 mタイルから260×195で
読み直し、上位4候補は1.25度・2 %刻みでさらに詰めます。同じ260×195の10 mマスクで比べた平均IoUは0.7089から
0.7473へ上がり、6,800フレームで改善、悪化は0でした。実データ512件の局所照合はPython参照実装の3.511秒に対して
Rustは0.106秒（3回の中央値）。これは局所照合だけの比較です。

### 可逆の10 m RLEキャッシュ

[`rust/coastline-native`](../rust/coastline-native) は1ビットのタイルをlibtiffで1行ずつ復号し、ランレングス形式
（`LWMRLE01`：幅、高さ、行ごとのオフセット、水と陸が切り替わるx座標の列）で保存します。72タイルの変換は約66秒、
容量は約10.64 GBです。[`native_rle.py`](../badapple_coastline/native_rle.py) からの参照は10 m画素を間引かずに返し、
512窓・25,958,400画素でGDALの読み取りと比べて差はありませんでした。キャッシュは
[`scripts/rebuild_rle.py`](../scripts/rebuild_rle.py) で作り直せます。

### CUDA探索（`fullres`）

[`rust/coastline-native-index`](../rust/coastline-native-index) はRLEキャッシュの全行（境界の切り替わり約26億点）を
走査し、南緯80度〜北緯84度で空間的に散らばった64,000地点を選びます。画角は4〜768 kmの16段階の幅と24方向
（15度刻み）で、合計24,576,000ビューです。[`cuda/coastline/coastline.cu`](../cuda/coastline/coastline.cu) を
[`cuda_backend.py`](../badapple_coastline/cuda_backend.py) から呼び、RLEキャッシュ全体をGPUメモリに載せます。各ビューを
10 mデータから直接256×192で標本化して64×48のINT8記述子にまとめ、全フレームとの相関をcuBLAS（INT8積、INT32累積）で
計算します。記述子は流しながら処理し、保存しません。残った候補（縮尺ごとに分散させた4候補と前回の画角）は
cuFFTで128×96、256×192、最後に原寸1444×1080で詰め、最終画素はRLEキャッシュから読み直します。DLLは
`--fmad=false` でビルドし、幾何計算をCPU版とビット単位で一致させています。標本化した25,128,780画素はCPU参照と
差0でした。RTX 4090で、ビュー構築と相関に215秒、全フレームの詳細探索に1,490.89秒かかりました。実際に写真を
採用した画角の平均mIoUは、原寸で測り直した前回の画角の0.7756から0.8038になりました。

### 精度優先の探索（`precision`）

[`precision_search.py`](../badapple_coastline/precision_search.py) と
[`precision_dictionary.py`](../badapple_coastline/precision_dictionary.py) で探索範囲を広げました
（設定は [`precision_config.py`](../badapple_coastline/precision_config.py)）。開始地点512,000、2〜1,024 kmの32段階の幅、
5度刻みの回転と鏡像で、組み合わせは合計2,359,296,000通りです。記述子は96×72（384×288点の標本から作成）で、
GPUが面積と符号付きの輪郭勾配の2系統で順位をつけ、それぞれタイル・縮尺ごとの上位8件を残します。粗い段階で
192候補が残り、64候補が原寸での調整（192×144、512×384、1444×1080）へ進み、各候補でFFTの平行移動の極大を
最大3つ調べます。さらに前後方向の2回のパスで、±1、2、4、8、16、32、64フレーム離れたフレームの候補を
シルエットの類似度で制限しつつ受け渡します。新しい画角は、mIoU・`fullres` の指標・精度指標・四辺の項を含む指標の
どれもが、測り直した前回の画角を下回らない場合だけ採用します。この段階では白黒の陸水動画も出力し、その後に
変わった画角の写真を取得します。写真を採用した画角の平均mIoU（映像部分1444×1076）は、測り直した前回の画角の
0.8070から0.8297になりました。

### 写真の取得

- Sentinel-2 L2Aは公開の [Earth Search](https://earth-search.aws.element84.com/v1) STACとAWS Open DataのCOGから取得します
  （[`satellite.py`](../badapple_coastline/satellite.py)、[`native_imagery.py`](../badapple_coastline/native_imagery.py)）。
  写真は一致した画角と同じ位置・回転・縦横比です。雪や海氷を減らすため、半球ごとに2020年前後の夏の撮影日を
  優先します。シーン分類レイヤー（SCL）で雲と欠測、`repair` 段階以降は雪・氷（クラス11）も判別し、同じ地点の
  複数の撮影日から晴れた画素を選びます。残った雲は撮影されたまま残し、その割合を表示します。`fullres` 以降は
  元のCOGから、元解像度または出力より細かいオーバービューで必要な範囲を直接読みます。
- 白飛びの検査（[`quality.py`](../badapple_coastline/quality.py)、[`repair.py`](../badapple_coastline/repair.py)）：
  96×72に縮小して、RGBすべてが225以上の面積が35 %以上、平均輝度210以上の面積が60 %以上、または明るくほぼ単色の
  画像を除外します。除外したフレームは別の夏の撮影日を試し、次に評価が下がらない保存済みの別画角を試します。
- Landsat 8/9 Collection 2 Level-2（[`landsat.py`](../badapple_coastline/landsat.py)）はMicrosoft Planetary Computerの
  `landsat-c2-l2` から、データAPIで16ビットの元画素を切り出して取得します。QA_PIXELで検査し、USGSの係数
  （DN × 0.0000275 − 0.2）と全シーン共通のsRGB変換で表示します。`landsat` 段階では、`repair` 段階で写真が得られず
  残っていた92フレームを置き換えました。91フレームは同じ画角のまま、1フレーム（5185）は保存済みの別画角です。
- `fullres` 以降は、実画素で95 %以上を覆うSentinel-2写真の欠けた画素だけを、同じ座標のLandsat画素で補う場合が
  あります（S2+LANDSATと表示）。`fullres` の結果では、26フレームにESA WorldCoverの公式Sentinel-2年次RGB合成画像も
  使いました（公開した動画には残っていません。取得したコードは含めていないので、`fullres` を再実行するとそれらのフレームの写真は変わります）。[`retry_native_primary.py`](../badapple_coastline/retry_native_primary.py) は、最初に写真が取れなかった
  第一候補を一度だけ再試行します。
- 画素を想像で補うことはしません。出力の全画素を実画素が覆う写真だけを使い、欠けた部分を塗りつぶしません。
  描画の前に、全フレームのマスク・写真の範囲・画像のハッシュを再確認します。フレームごとの出典は `frames.jsonl` と
  `CREDITS.txt` に残します。

### 2560×1920版の描画（`edition`）

[`native_2560.py`](../badapple_coastline/native_2560.py) は、各パネルを可逆の素材から出力サイズで直接描き、
1回だけエンコードします（座標は x0, y0, x1, y1）。

| パネル | 範囲 | 内容 |
| --- | --- | --- |
| 写真 | 40,140,1960,1580 | 衛星写真。1444×1080の素材を1回だけ1920×1440へリサンプリング |
| 原作 | 2000,184,2520,574 | 元フレームを復号して縮小しただけのもの |
| 一致マスク | 2000,650,2520,1040 | 同じ画角の10 mマスク |
| 世界地図 | 2000,1116,2520,1376 | Natural Earth 1:110mの陸地。現在の範囲とこれまでのすべての範囲（[`footprint_map.py`](../badapple_coastline/footprint_map.py)） |
| 周辺図 | 2000,1488,2520,1880 | 10 mマスクによる北が上の正距方位図。最小20 kmまたは写真幅の5倍。金色が現在の写真、水色が過去の近傍の画角（[`local_map.py`](../badapple_coastline/local_map.py)） |

写真の下には、最寄りの地名（Natural Earthの集落と距離）、緯度・経度、撮影日、画角の幅と回転、mIoU、画像の出典と
雲の割合を表示し、右欄には鏡像かどうかとフレーム番号を表示します。トーン処理
（[`cuda/video_flash`](../cuda/video_flash/video_flash.cu)、[`flash_cuda.py`](../badapple_coastline/flash_cuda.py)）は
現在のフレームだけを使います。輝度の0.5/99.5パーセンタイルから黒と白のレベルを決め、ガンマ0.75で相対輝度
0.002〜0.092（幅0.09）に収め、彩度は1.15倍です。暗い画素では色差の倍率を1以下にし、暗い水面の圧縮ノイズが
青やマゼンタに強調されないようにしています。前のフレームの画素は使いません。

点滅の低減は写真パネルだけに適用します。エンコード後に、全フレームの写真パネルの全画素を復号し、
WCAG 2.2 SC 2.3.1の一般閃光・赤色閃光の閾値に対して保守的な参照視野の方法で検査します
（[`flash_reduction.py`](../badapple_coastline/flash_reduction.py)）。不合格なら幅を狭めて再エンコードする仕組みです。
公開動画は幅0.09で合格しました（閃光候補の最大面積1,134 px、赤色277 px、上限95,480 px）。原作と一致マスクの
パネルは処理しておらず点滅し得るため、動画全体の適合は主張せず、タイトルに注意書きを入れています。

出力は2560×1920で、元の6,955フレームを各1回、30.000 fpsで表示します（231.833秒）。AV1 `av1_nvenc` のプリセット
p7・2パス、10ビット、目標45 Mbps・最大90 Mbps、音声はAAC 192 kbpsです。原作のエンディングクレジットに当たる
フレーム6562（0始まり）以降は、1つの実在の画角（フレーム6570のもの）を固定して表示します。公開動画では
2,731フレームが鏡像、544フレームがLandsatの画素を使っています。

### 陸と水＋OpenStreetMap版（`landwater`、`site`）

[`landwater_osm.py`](../badapple_coastline/landwater_osm.py) は、一致した10 mの陸水の画角を主画面（1920×1440）にし、
右欄に原作、同じ画角の衛星写真、世界地図、OpenStreetMapを並べます。
[`landwater_main.py`](../badapple_coastline/landwater_main.py) は一致に使った変換そのものでRLEキャッシュを標本化します。
広い画角では面積の超標本化、拡大した画角ではガウス重み付きの場（σ = 0.55セル）を0.5で等値線化し、回転した10 mの
セルが階段状に見えないようにしています。エンコード後の動画の25フレームごとに元のマスクと比べた平均IoUは
0.9951です。一部の地域では、マスク自体の海岸線が階段状になっています。

OpenStreetMapのパネルは、画角ごとに1回、ヘッドレスChromium上のMapLibre GL JS 5.24.0で描きます
（[`osm_maps.py`](../badapple_coastline/osm_maps.py)、[`tools/landwater-osm`](../tools/landwater-osm)）。スタイルは
`@protomaps/basemaps` を動画の配色に合わせたものです。タイルはProtomapsの全球ビルド（PMTiles v3）から、
依存パッケージのないNodeサーバー [`tools/landwater-osm/server.mjs`](../tools/landwater-osm/server.mjs) がHTTPの範囲
リクエストで読み、すべてディスクにキャッシュします。公開のOSMタイルサーバーは大量の描画を認めていないためです。
範囲の枠、縮尺、方位記号は同じWebメルカトルの計算でPythonが描きます。

[`geostats.py`](../badapple_coastline/geostats.py) は、Natural Earth 1:10mのAdmin-0 map unitsとの測地的な陸地の重なりが
最も大きい国・地域を各画角に割り当てます。陸地が2 %未満の画角は、200海里以内で最も近いものを使います。大陸は
Natural Earthの `CONTINENT` により、ロシアは東経60度で分けます。四辺形のIoUが0.5を超えて重なる画角は1つの
場所にまとめます。結果は一致したフレーム6,562、画角6,355、場所1,680、国・地域138、大陸7です。エンディングでは
統計のカードを動きのある表示で重ねます。公開動画のカードにある「表示した面積」（2,420万km²）は重なりを重複して
数えた値で、全画角の和集合は831万km²（地表の1.63 %）です。サイトでは後者を使っています。

フレームはスレッドプールで合成し、`av1_nvenc`（10ビット、45 Mbps）へ流します。写真パネルは衛星写真版と同じ
トーン幅を使い、主画面は原作の明暗の変化をそのまま再現するため点滅の低減はしていません。`site`
（[`landwater_web.py`](../badapple_coastline/landwater_web.py)）はサイト用のデータを書き出します。すべての画角の四隅、
フレーム範囲、国、大陸、場所、評価値、全フレームの一覧、統計、画角ごとの原作とマスクのサムネイルです。
タイムスタンプは `nw02Enk8o1I` に合わせています（フレームnが (n−1)/30秒）。動画IDは `--youtube` で変えられます。

## 再現性について

- `precision` の探索は、以前のv3辞書の地域別候補リストとv3の開始画角も合わせて使いましたが、それらを作った
  コードはこのリポジトリにありません。コードではこの辞書を任意扱いにしているため
  （[`precision_search.py`](../badapple_coastline/precision_search.py) の `legacy_available()`）、新たに実行すると
  v4辞書だけを使い、選ばれる画角が変わることがあります。
- 衛星画像のアーカイブ、STACカタログ、Protomapsの日次ビルドは時間とともに変わるため、写真や地図は異なる場合が
  あります。動画は2026-11-09のProtomapsビルドを使いました。古いビルドは削除されるので、`BADAPPLE_PMTILES` で
  現在のビルドかローカルの抽出ファイルを指定してください。
- 公開動画のフレームごとの正確な結果は [`results/satellite/frames.csv`](../results/satellite/frames.csv) にあります。
  列は `frame`、`time_s`、`longitude`、`latitude`、`width_km`、`height_km`、`rotation_clockwise_deg`、`mirrored`、
  `miou`、`precision_objective`、写真範囲の四隅（`corner1_lon` … `corner4_lat`）、`photo_collection`、`photo_scenes`、
  `photo_dates`、`photo_cloud_fraction`、`country`、`territory`、`continent`、`nearest_place`、`nearest_place_country`、
  `nearest_place_km`、`note` です。完了した実行から [`scripts/export_results.py`](../scripts/export_results.py) で作れます。
- 整理したコードは元のコードと照合済みです。元の中間出力を使い、本編のフレーム145、2500、6600の静止画
  （`python -m badapple_coastline.native_2560 --stills 145,2500,6600`）と、陸と水版のフレーム3000、6790
  （`python -m badapple_coastline.landwater_osm stills --frames 3000,6790`）が画素単位で一致します。
- テストは [`tests/`](../tests)（`python -m pytest`）、GPUとCPUの比較は `scripts/validate_*.py` にあります。
- 必要な環境：Windows、CUDA Toolkit（cuBLAS、cuFFT）とMSVCを使えるNVIDIA GPU、最終エンコード用にAV1対応の
  NVENC、Rust、FFmpeg、OpenStreetMapパネル用のNode.jsとPlaywrightのChromium。公開版はRTX 4090で作りました。
  CUDAの段階ではRLEキャッシュ全体をGPUメモリに載せます。

## 環境変数

| 変数 | 意味 |
| --- | --- |
| `BADAPPLE_CACHE` | OpenStreetMapのタイルと陸水パネルのキャッシュの置き場所（既定は `cache`） |
| `BADAPPLE_PMTILES` | ProtomapsのPMTilesのURLまたはローカルファイル（既定は2026-11-09のビルド） |
| `BADAPPLE_CHROME` | MapLibre用のChromium。未指定なら最新のPlaywright Chromium（`PLAYWRIGHT_BROWSERS_PATH` も参照） |
| `CUDA_PATH` | CUDA Toolkitのルート。未指定ならインストール済みの最新版 |
| `CUDA_ARCH` | CUDA DLLのGPUアーキテクチャ（既定 `sm_89`。例：`sm_86`、`sm_120`） |

`run-satellite.ps1` は `OMP_NUM_THREADS=2`、`OPENBLAS_NUM_THREADS=2`、`RAYON_NUM_THREADS=16`、
`GDAL_NUM_THREADS=1` も設定します。GDALは単一スレッドの別プロセスで動かします。

## 容量と時間

- マスクのタイル：約7.26 GB。可逆の10 m RLEキャッシュ：約10.64 GB。0.0025度の探索用配列：約10.37 GB、
  0.01度の配列：約648 MB。原寸フレームのキャッシュ（6,955 × 1444×1080バイト）：約10.85 GB。
- これに各段階の写真キャッシュ、フレームごとのマスク、中間動画が加わります。全工程を実行する前に空き容量を
  確認してください。
- 全工程の実行には、ダウンロードとGPU処理で何日もかかります。写真の取得は配信サービス側の速度に左右されます。
  各段階は完了済みの処理を再利用するので、中断しても同じコマンドで再開できます。

## データと出典

Kindermannの陸水マスク（CC BY 4.0。ESA WorldCover、OpenStreetMap、Copernicus DEMから作成）、Copernicus Sentinel-2の
データ（「Contains modified Copernicus Sentinel data [年]」）、米国地質調査所（USGS）提供のLandsat（パブリック
ドメイン）、ESA WorldCoverの年次合成画像（CC BY 4.0）、Protomaps経由のOpenStreetMapデータ（ODbL、
「© OpenStreetMap contributors」）、Natural Earth（パブリックドメイン）を使っています。完全な表記は描画のたびに
`CREDITS.txt` へ書き出します。マスクの水域には湖や川も含まれるため、一致した「海岸線」が湖岸や河岸のこともあります。
