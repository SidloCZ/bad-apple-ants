# Bad Apple!! を実在する地球と月で

[English](README.md) | **日本語**

影絵アニメーション「Bad Apple!!」のすべてのフレームを実在の場所で置き換えた、3 本の動画のコードです。

| 動画 | 各フレームの中身 |
| --- | --- |
| [すべてのフレームが地球上の実在の場所（衛星写真）](https://youtu.be/nw02Enk8o1I) | 陸と水の境界がシルエットと一致する地球上の範囲を探し、その範囲の実際の Sentinel-2 / Landsat の写真を表示 |
| [すべてのフレームが地球の実在の陸と水（OpenStreetMap）](https://youtu.be/kB3fzm_Un2g) | 同じ範囲の陸と水そのものをメインにし、OpenStreetMap の地図を添えた版 |
| [すべてのフレームが月面の本物の影（かぐや / SELENE）](https://youtu.be/oe94IDXho9g) | 実在の月面画像から光と影が最もよく一致する場所 |

描いたり生成したりした部分はありません。各フレームは実在する 1 か所を切り出し・拡大縮小・回転（一部は左右反転）・
トーン調整しただけのものです。地球版の全範囲の地図は
[bad-apple-earth-map](https://github.com/ikuto32/bad-apple-earth-map)（[サイト](https://bad-apple-earth-map.ikutotest.xyz/)）にあります。

## しくみ

**地球**：Kindermann（2025）の 10 m 全球陸海マスクから、各フレームのシルエットと陸と水の形が一致する範囲を探します。
位置は全球、幅は約 2〜1,000 km、回転は任意で、左右反転も許します。Rust/Rayon と CUDA で数十億の候補を元の 10 m 画素で
評価し（バランス平均 IoU に輪郭・境界の一致を加えた指標）、最良の範囲について Sentinel-2 L2A の撮影シーンを公開クラウドから
読み込み、雲・雪・白飛びを検査します。Sentinel-2 で覆えない範囲は Landsat 8/9 で補います。2560×1920 の動画には、写真・元フレーム・
一致したマスク・それまでの全範囲の世界地図・周辺図を表示します。詳細は [docs/satellite.ja.md](docs/satellite.ja.md)。

**月**：かぐや地形カメラの約 24 万枚の画像と LROC WAC 全球モザイクを GPU で全探索し（候補 3,933 億、32×24 セルの正規化相互相関）、
再探索・全解像度での飽和検査をしてから元画像から切り出します。下部には最寄りの IAU 地名と、その場所の太陽高度を表示します。
詳細は [docs/moon.ja.md](docs/moon.ja.md)。

## 結果データ

公開した動画のフレームごとの結果を [`results/`](results) にテキストで置いています（画像は含みません）。

| ファイル | 内容 |
| --- | --- |
| [`results/satellite/frames.csv`](results/satellite/frames.csv) | 全フレーム：中心・幅・回転・反転・四隅の座標・一致度・写真のシーンと撮影日・国・大陸・最寄りの地名 |
| [`results/moon/frames.csv`](results/moon/frames.csv) | 全フレーム：かぐやの画像・中心・幅・回転・反転・NCC・地名・太陽 |
| [`results/moon/matches.json`](results/moon/matches.json) | 全種類のフレームで採用したポーズ。探索なしで月面版を作り直せます（[手順](docs/moon.ja.md#公開版の結果から作り直す)） |
| [`results/moon/rasters.txt`](results/moon/rasters.txt) | それらのポーズが使う 2,570 枚のかぐや画像 |

## 必要なもの

- Windows 10/11（GPU 部分は MSVC でビルドし、実行スクリプトは PowerShell です）
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) 付きの NVIDIA GPU（RTX 4090・CUDA 13.4 で作成。他の GPU では `CUDA_ARCH` を設定、例 `sm_86`）
- Visual Studio 2022 の C++ ビルドツール、[Rust](https://rustup.rs/)、Python 3.11 以上（または [uv](https://docs.astral.sh/uv/)）、PATH 上の FFmpeg/FFprobe（最終エンコードには `av1_nvenc`）
- OpenStreetMap の地図には Node.js と Playwright の Chromium（`npx playwright install chromium`）
- ディスクと時間：全工程では数百 GB をダウンロードし、GPU で数日かかります。まずテストと 1 工程から試してください。

## はじめ方

```powershell
.\setup.ps1                                     # 3 つの Python パッケージを入れた .venv を作成
.venv\Scripts\python -m badapple_media fetch    # 元動画とフォント（コミット固定・SHA-256 検査）
.venv\Scripts\python -m pytest                  # 92 件のテスト。ダウンロードも GPU も不要
```

各工程は `output/` の下に自分のフォルダーを作り、中断した所から再開します。

```powershell
.\run-satellite.ps1                    # 公開版を作ったときと同じ順に全工程
.\run-satellite.ps1 -Stage precision   # 1 工程ずつでも実行できる
.\run-moon.ps1                         # data → frames → verify → search → refine → crops → encode
.\run-moon.ps1 -Stage crops
```

| `run-satellite.ps1 -Stage` | 出力 | |
| --- | --- | --- |
| `fetch` | `assets/` | 陸海マスク（7.3 GB）・Natural Earth・動画・フォント |
| `baseline` | `output/coastline` | 最初の全球照合と Sentinel-2 写真 |
| `wide` | `output/coastline-wide-rust` | 6〜768 km の範囲、Rust/Rayon、可逆の 10 m キャッシュ |
| `repair` | `output/coastline-wide-rust-fixed` | 白飛び・雪の写真を再取得 |
| `landsat` | `output/coastline-wide-rust-landsat` | Sentinel-2 で覆えない範囲を Landsat 8/9 で補完 |
| `fullres` | `output/coastline-fullres-cuda-v2` | 元動画の 1444×1080 そのままでの CUDA 探索 |
| `precision` | `output/coastline-precision-v4` | 輪郭を考慮し左右反転も含む CUDA 探索、新しい写真 |
| `edition` | `output/coastline-precision-v4-2560x1920-30p-native-av1` | 公開した衛星写真版の動画 |
| `landwater` | `output/landwater-osm` | 陸と水＋OpenStreetMap 版の動画 |
| `site` | `output/landwater-site/web` | 場所マップのサイト用データ |

完全な再現には限界があります。衛星画像のアーカイブや地図のビルドは更新され、precision 探索の入力の一部は以前の実行に由来します。
[docs/satellite.ja.md](docs/satellite.ja.md) に詳しく書いています。

## 構成

| パス | |
| --- | --- |
| [`badapple_media/`](badapple_media) | 共通の FFmpeg 入出力・フレームのキャッシュ・フォント・固定ダウンロード |
| [`badapple_coastline/`](badapple_coastline) | 地球版：データ・探索・写真の取得・描画 |
| [`badapple_moon/`](badapple_moon) | 月面版：カタログ・取り込み・切り出し・描画 |
| [`rust/`](rust) | `coastline-matcher`（形の探索）、`coastline-native`（可逆の 10 m キャッシュと標本化）、`coastline-native-index`（海岸の中心点）、`moon-search`（月面探索と CPU 検証） |
| [`cuda/`](cuda) | `coastline`（範囲の標本化・記述子・FFT 再探索）、`moon`（ポーズの採点）、`video_flash`（写真の輝度制限）。[`cuda/common.ps1`](cuda/common.ps1) でビルド |
| [`tools/landwater-osm/`](tools/landwater-osm) | OpenStreetMap の地図用のローカルタイルサーバーと MapLibre のページ |
| [`scripts/`](scripts) | フレームの準備・CPU と GPU の照合・YouTube 用素材・`export_results.py` |
| [`tests/`](tests) | pytest |

## テストと検証

```powershell
.venv\Scripts\python -m pytest
cargo test --release --manifest-path rust/coastline-native-index/Cargo.toml
```

`scripts/validate_*.py` は実データで CUDA カーネルを CPU の参照実装と照合します（各工程の出力が必要）。
公開用の整理では、動画を描画するコードを整理前のコードと比較し、静止画が画素単位で一致することを確認しました。

## ライセンスとクレジット

コードは [MIT](LICENSE)。データにはそれぞれの利用条件があります。出典と表記は [docs/data-and-licenses.md](docs/data-and-licenses.md) を参照してください。
Bad Apple!! の映像と音楽の権利は各権利者（Alstroemeria Records、ZUN / 上海アリス幻樂団、影絵 PV あにら氏）にあり、このリポジトリには含まれていません。

コードの一部に AI（Anthropic の Claude）の支援を受けています。
