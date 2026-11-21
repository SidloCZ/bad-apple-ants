"""Tests for license normalization, DwC-A URL/label filtering, sharded caching, and chunked refinement."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from badapple_ants.catalog import build_catalog_dwca, normalize_license, parse_media_candidate
from badapple_ants.download import _run_worker_pool, _thumb_path, pack_thumbs
from badapple_ants.search import _refine_chunked


def test_license_normalization_and_media_override() -> None:
    assert normalize_license("http://creativecommons.org/publicdomain/zero/1.0/") == "CC0-1.0"
    assert normalize_license("http://creativecommons.org/licences/by/4.0/") == "CC-BY-4.0"
    assert normalize_license("CC_BY_4_0") == "CC-BY-4.0"
    assert normalize_license("CC_BY_NC_4_0") == "CC-BY-NC-4.0"
    assert normalize_license("CC_BY_NC_4_0", commercial_only=True) is None
    assert normalize_license("http://creativecommons.org/licenses/by-nc/4.0/") == "CC-BY-NC-4.0"
    assert normalize_license("http://creativecommons.org/licenses/by-nc/4.0/", commercial_only=True) is None
    assert normalize_license("http://creativecommons.org/licenses/by-nc-nd/4.0/") is None
    assert normalize_license("© All rights reserved") is None

    # Media-level ND must reject even when occurrence license is CC-BY-NC.
    rejected = parse_media_candidate(
        "https://inaturalist-open-data.s3.amazonaws.com/photos/12345/original.jpg",
        media_license="http://creativecommons.org/licenses/by-nc-nd/4.0/",
        occ_license="CC_BY_NC_4_0",
    )
    assert rejected is None

    # Empty media-level license inherits permissive occurrence license.
    inherited = parse_media_candidate(
        "https://iiif.mcz.harvard.edu/iiif/3/187251/full/max/0/default.jpg",
        title="MCZ:Ent:23043 habitus dorsal view",
        description="habitus dorsal view",
        media_license="",
        occ_license="CC_BY_4_0",
    )
    assert inherited is not None
    assert inherited["source"] == "mcz"
    assert inherited["view"] == "dorsal"
    assert inherited["thumb_url"] == "https://iiif.mcz.harvard.edu/iiif/3/187251/full/!256,256/0/default.jpg"


def test_label_filtering_across_museum_sources() -> None:
    assert (
        parse_media_candidate(
            "https://static.antweb.org/images/casent0001/casent0001_l_1_high.jpg",
            occ_license="CC_BY_4_0",
        )
        is None
    )
    assert (
        parse_media_candidate(
            "https://iiif.mcz.harvard.edu/iiif/3/187254/full/max/0/default.jpg",
            title="MCZ:Ent:23043 labels",
            description="labels",
            occ_license="CC_BY_NC_4_0",
        )
        is None
    )
    assert (
        parse_media_candidate(
            "https://collections.nmnh.si.edu/media/?i=11630430&h=960",
            title="00529465_L",
            description="Type specimen, labels",
            occ_license="CC0_1_0",
        )
        is None
    )
    assert (
        parse_media_candidate(
            "https://collections.nmnh.si.edu/media/?i=11630431&h=1920",
            title="00529465_O",
            description="Type specimen, pin",
            occ_license="CC0_1_0",
        )
        is None
    )
    nmnh_head = parse_media_candidate(
        "https://collections.nmnh.si.edu/media/?i=11630428&h=925",
        title="00529465_H",
        description="Type specimen, head",
        occ_license="CC0_1_0",
    )
    assert nmnh_head is not None
    assert nmnh_head["thumb_url"] == "https://collections.nmnh.si.edu/media/?i=11630428&w=320"


def test_dwca_build_and_sharded_pack(tmp_path: Path) -> None:
    zip_path = tmp_path / "mini.zip"
    occ_tsv = "\n".join(
        [
            "gbifID\tspecies\tgenus\tcountryCode\tlevel0Name\tdecimalLatitude\tdecimalLongitude\tlicense\trecordedBy\tdatasetKey\treferences",
            "101\tLasius niger\tLasius\tCZ\tCzechia\t50.0\t14.4\tCC_BY_NC_4_0\tAlice\tds-inat\thttps://example.org/101",
            "102\tFormica rufa\tFormica\tCZ\tCzechia\t49.2\t16.6\tCC_BY_4_0\tBob\tds-aw\thttps://example.org/102",
        ]
    )
    mm_tsv = "\n".join(
        [
            "gbifID\ttype\tidentifier\treferences\ttitle\tdescription\tcreator\tlicense\trightsHolder",
            "101\tStillImage\thttps://inaturalist-open-data.s3.amazonaws.com/photos/1/original.JPG\t\t\t\tAlice\thttp://creativecommons.org/licenses/by-nc/4.0/\tAlice",
            "101\tStillImage\thttps://inaturalist-open-data.s3.amazonaws.com/photos/2/original.JPG\t\t\t\tAlice\thttp://creativecommons.org/licenses/by-nc-nd/4.0/\tAlice",
            "102\tStillImage\thttps://static.antweb.org/images/casent1/casent1_d_1_med.jpg\t\t\t\tBob\tCC BY 4.0\tCAS",
            "102\tStillImage\thttps://static.antweb.org/images/casent1/casent1_l_1_med.jpg\t\t\t\tBob\tCC BY 4.0\tCAS",
        ]
    )
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("occurrence.txt", occ_tsv)
        zf.writestr("multimedia.txt", mm_tsv)

    cat_path = tmp_path / "catalog.jsonl"
    stats = build_catalog_dwca(
        cat_path,
        zip_path=zip_path,
        inat_target=2,
        antweb_target=2,
        mcz_target=0,
        nmnh_target=0,
        fill_remainder=False,
    )
    assert stats["total"] == 2
    assert stats["sources"] == {"antweb": 1, "inat": 1}

    stats_unlim = build_catalog_dwca(
        tmp_path / "catalog_unlim.jsonl",
        zip_path=zip_path,
        inat_target=10,
        antweb_target=10,
        mcz_target=0,
        nmnh_target=0,
        fill_remainder=False,
        max_per_occurrence=0,
    )
    assert stats_unlim["total"] == 2

    # Verify legacy flat file reuse and sharded pack creation
    cache_dir = tmp_path / "cache"
    items = [__import__("json").loads(line) for line in cat_path.read_text(encoding="utf-8").splitlines()]
    img = Image.new("L", (160, 120), color=220)
    ImageDraw.Draw(img).rectangle((40, 30, 110, 90), fill=20)

    # Save first item at legacy flat path, second item at sharded path
    legacy_path = cache_dir / "thumbs" / items[0]["source"] / f"{items[0]['id']}.jpg"
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(legacy_path)

    sharded_path = _thumb_path(cache_dir, items[1])
    sharded_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(sharded_path)
    assert _thumb_path(cache_dir, items[0]) == legacy_path

    meta = pack_thumbs(items, cache_dir, canvas_size=192)
    assert meta["count"] == 2
    assert (cache_dir / "pack" / "gray.npy").is_file()


def test_refine_chunked_consistency() -> None:
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    rng = np.random.default_rng(7)
    gray_np = rng.integers(0, 255, size=(6, 192, 192), dtype=np.uint8)
    dims_gpu = torch.tensor([[192, 144]] * 6, dtype=torch.int32, device=device)
    q64 = torch.randn((2, 3072), device=device)
    q64 = q64 / q64.norm(dim=-1, keepdim=True)
    q32 = torch.randn((2, 768), device=device)
    q32 = q32 / q32.norm(dim=-1, keepdim=True)
    cands = torch.tensor(
        [[float(i), 0.5, 0.5, 0.7, 0.0, 0.0] for i in range(6)],
        dtype=torch.float32,
        device=device,
    )
    qidx = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long, device=device)

    p1, s64_1, s32_1 = _refine_chunked(gray_np, dims_gpu, q64, q32, cands, qidx, 192, chunk_imgs=6)
    p2, s64_2, s32_2 = _refine_chunked(gray_np, dims_gpu, q64, q32, cands, qidx, 192, chunk_imgs=2)
    assert torch.allclose(p1, p2)
    assert torch.allclose(s64_1, s64_2)
    assert torch.allclose(s32_1, s32_2)


def test_worker_pool_circuit_breaker(monkeypatch, tmp_path: Path) -> None:
    import asyncio
    from tqdm import tqdm

    called: list[str] = []

    async def fake_fetch(client, host_sems, default_sem, url, dest) -> str:
        called.append(url)
        return "transport_err"

    monkeypatch.setattr("badapple_ants.download._fetch_one", fake_fetch)
    pairs = [(f"https://example.org/{i}.jpg", tmp_path / f"{i}.jpg") for i in range(50)]
    with tqdm(total=len(pairs), disable=True) as pbar:
        ok = asyncio.run(
            _run_worker_pool(
                pairs,
                client=None,  # type: ignore[arg-type]
                host_sems={},
                default_sem=asyncio.Semaphore(4),
                num_workers=2,
                pbar=pbar,
                label="example.org",
                max_consec_transport_errors=5,
            )
        )
    assert ok == 0
    assert len(called) < len(pairs)


def test_dwca_excludes_family_only_records(tmp_path: Path) -> None:
    zip_path = tmp_path / "fam.zip"
    occ_tsv = "\n".join(
        [
            "gbifID\tspecies\tgenus\tcountryCode\tlevel0Name\tdecimalLatitude\tdecimalLongitude\tlicense\trecordedBy\tdatasetKey\treferences",
            "201\t\t\tMG\tMadagascar\t-18.9\t47.5\tCC_BY_4_0\tCollector\tds-aw\thttps://example.org/201",
            "202\t\tCamponotus\tMG\tMadagascar\t-18.9\t47.5\tCC_BY_4_0\tCollector\tds-aw\thttps://example.org/202",
            "203\tLasius niger\tLasius\tCZ\tCzechia\t50.0\t14.4\tCC_BY_4_0\tAlice\tds-inat\thttps://example.org/203",
        ]
    )
    mm_tsv = "\n".join(
        [
            "gbifID\ttype\tidentifier\treferences\ttitle\tdescription\tcreator\tlicense\trightsHolder",
            "201\tStillImage\thttps://static.antweb.org/images/casent0838376/casent0838376_p_1_med.jpg\t\t\t\tCollector\tCC BY 4.0\tCAS",
            "202\tStillImage\thttps://static.antweb.org/images/casent202/casent202_p_1_med.jpg\t\t\t\tCollector\tCC BY 4.0\tCAS",
            "203\tStillImage\thttps://inaturalist-open-data.s3.amazonaws.com/photos/203/original.jpg\t\t\t\tAlice\tCC BY 4.0\tAlice",
        ]
    )
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("occurrence.txt", occ_tsv)
        zf.writestr("multimedia.txt", mm_tsv)

    cat_path = tmp_path / "filtered.jsonl"
    stats = build_catalog_dwca(
        cat_path,
        zip_path=zip_path,
        inat_target=5,
        antweb_target=5,
        mcz_target=0,
        nmnh_target=0,
        fill_remainder=False,
    )
    assert stats["total"] == 2
    ids = {__import__("json").loads(line)["id"] for line in cat_path.read_text(encoding="utf-8").splitlines()}
    assert "antweb_casent0838376_p_1" not in ids
    assert "antweb_casent202_p_1" in ids
    assert "inat_203" in ids

