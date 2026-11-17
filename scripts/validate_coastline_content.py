"""Independent original-size CPU references for content ROI and frame edges."""

import json, struct
from pathlib import Path
import numpy as np
from scipy.ndimage import maximum_filter, distance_transform_cdt
from badapple_coastline.cuda_backend import CudaAtlas

rng = np.random.default_rng(19851)
root = Path(".tmp/content-validation")
root.mkdir(parents=True, exist_ok=True)


def atlas(mask, name):
    rows = [np.flatnonzero(np.diff(np.r_[0, row])).astype("<u4") for row in mask]
    offsets = np.r_[0, np.cumsum([len(r) for r in rows])].astype("<u8")
    path = (root / (name + ".rle")).resolve()
    path.write_bytes(
        b"LWMRLE01"
        + struct.pack("<II", mask.shape[1], mask.shape[0])
        + offsets.tobytes()
        + b"".join(r.tobytes() for r in rows)
    )
    cuda = CudaAtlas(load_globe=False)
    for row in range(6):
        for col in range(12):
            left = -180 + col * 30
            top = 90 - row * 30
            cuda.check(
                cuda.library.coast_cuda_add_tile(
                    cuda.handle, row * 12 + col, str(path).encode(), left, top - 30, left + 30, top
                )
            )
    cuda.check(cuda.library.coast_cuda_finish_tiles(cuda.handle))
    return cuda


def edges(a, border):
    e = np.zeros_like(a, bool)
    v = a[border : a.shape[0] - border] if border else a
    out = np.zeros_like(v, bool)
    out[:, 1:] |= v[:, 1:] != v[:, :-1]
    out[:, :-1] |= v[:, 1:] != v[:, :-1]
    out[1:] |= v[1:] != v[:-1]
    out[:-1] |= v[1:] != v[:-1]
    e[border : a.shape[0] - border if border else a.shape[0]] = out
    return e


def distance(e):
    return np.minimum(distance_transform_cdt(~e, metric="taxicab"), 256) if e.any() else np.full(e.shape, 256)


def reference(scene, query, border, tolerance):
    h, w = scene.shape
    valid = np.ones_like(scene, bool)
    if border:
        valid[:border] = False
        valid[-border:] = False
    s = scene.astype(bool)
    q = query.astype(bool)
    se = edges(s, border)
    qe = edges(q, border)
    sd = distance(se)
    qd = distance(qe)
    weights = np.stack([np.rint(1e6 * np.exp(-np.arange(257) / (tau * w / 1444))).astype(np.uint64) for tau in [8, 32]])
    weights[:, 256] = 0
    values = [
        (s & valid).sum(),
        (s & q & valid).sum(),
        0,
        se.sum(),
        qe.sum(),
        (se & maximum_filter(qe, size=2 * tolerance + 1, mode="constant")).sum(),
        (qe & maximum_filter(se, size=2 * tolerance + 1, mode="constant")).sum(),
        weights[0, qd[se]].sum(),
        weights[0, sd[qe]].sum(),
        weights[1, qd[se]].sum(),
        weights[1, sd[qe]].sum(),
        qd[se].sum(),
        sd[qe].sum(),
    ]
    last = h - border - 1
    values += [
        (s[border] != q[border]).sum(),
        (s[last] != q[last]).sum(),
        (s[border : h - border, 0] != q[border : h - border, 0]).sum(),
        (s[border : h - border, -1] != q[border : h - border, -1]).sum(),
    ]
    return np.array(values, np.uint64)


geos = np.array([[15, 15, 1200, a, m] for m in [0, 1] for a in [0, 15, 67]], np.float64)
checks = 0
with atlas(rng.integers(0, 2, (64, 64), dtype=np.uint8), "mixed") as cuda:
    for w, h, border in [(64, 48, 2), (512, 384, 1), (1444, 1080, 2)]:
        scenes = cuda.sample(geos, (w, h), aspect=1080 / 1444)
        query = scenes[1].copy()
        query[h // 4 : h // 3, w // 4 : w // 3] ^= 1
        query[:border] = 0
        query[-border:] = 0
        tolerance = max(1, round(3 * w / 1444))
        result = cuda.score_structure(geos, query, tolerance=tolerance, border_y=border)
        expected = np.stack([reference(scene, query, border, tolerance) for scene in scenes])
        np.testing.assert_array_equal(result["structure_counts"], expected)
        altered = query.copy()
        altered[:border] ^= 1
        altered[-border:] ^= 1
        other = cuda.score_structure(geos, altered, tolerance=tolerance, border_y=border)
        for key in result:
            np.testing.assert_array_equal(result[key], other[key])
        normal = cuda.score_structure(geos[:3], query[:, ::-1], tolerance=tolerance, border_y=border)
        np.testing.assert_array_equal(result["shape_counts"][3:], normal["shape_counts"])
        np.testing.assert_array_equal(
            result["structure_counts"][3:, 13:], normal["structure_counts"][:, [13, 14, 16, 15]]
        )
        if w == 64:
            full = cuda.score_structure(geos, query, tolerance=tolerance, border_y=0)
            legacy = cuda.score_shape(geos, query, tolerance=tolerance)
            np.testing.assert_array_equal(full["shape_counts"], legacy["shape_counts"])
        checks += len(scenes)
        print(
            f"All17 ROI/contour/border integer counts match CPU; padding invariance and mirror identity: {w}x{h}",
            flush=True,
        )
    build = cuda.build
for land in [0, 1]:
    with atlas(np.full((64, 64), land, np.uint8), "constant" + str(land)) as cuda:
        query = np.full((1080, 1444), land, np.uint8)
        query[:2] = 0
        query[-2:] = 0
        result = cuda.score_structure(geos, query, border_y=2)
        np.testing.assert_array_equal(
            result["structure_counts"], np.stack([reference(np.full_like(query, land), query, 2, 3) for _ in geos])
        )
        assert (result["miou"] == 1).all() and (result["structure_objective"] == 1).all()
        if land:
            assert (cuda.score_shape(geos, query)["miou"] < 0.5).all()
        h, w = 48, 64
        step = 4 * (1080 / 1444) / h / 111.32
        missing = np.array([[15, 90 - (h / 2 - 1.5) * step, 4, 0, 0]])
        invalid = cuda.score_structure(missing, np.zeros((h, w), np.uint8), border_y=2)
        assert invalid["missing"][0] > 0 and invalid["structure_objective"][0] == -1
print("Constant white/black content scores1; missing source pixels outside ROI still reject a view", flush=True)
report = dict(
    integer_count_differences=0,
    padding_invariance_differences=0,
    mirror_identity_differences=0,
    constant_content_score=1,
    missing_source_outside_roi_rejected=True,
    validated_views=checks,
    sizes=[[64, 48], [512, 384], [1444, 1080]],
    cuda_build=build,
)
Path("output/coastline-content-cuda-validation.json").write_text(json.dumps(report, indent=2))
