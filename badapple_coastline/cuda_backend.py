"""Native original-mask access and integer descriptor search on CUDA."""

from __future__ import annotations
import ctypes, json, os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def metrics_from_counts(out, qm, pixels):
    """Balanced binary-region and tolerant-boundary metrics from exact integers."""
    values = out.astype(np.float64)
    land, intersection, missing, edges, qedges, matched, qmatched = values.T
    lu = land + qm - intersection
    wu = pixels - intersection
    li = np.divide(intersection, lu, out=np.ones_like(lu), where=lu > 0)
    wi = np.divide(pixels - land - qm + intersection, wu, out=np.ones_like(wu), where=wu > 0)
    precision = np.divide(matched, edges, out=np.ones_like(edges), where=edges > 0)
    recall = np.divide(qmatched, qedges, out=np.ones_like(qedges), where=qedges > 0)
    bf = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(recall), where=precision + recall > 0)
    bf[(edges == 0) != (qedges == 0)] = 0
    miou = (li + wi) / 2
    objective = 0.85 * miou + 0.15 * bf
    objective[missing > 0] = -1
    return dict(
        land_iou=li,
        water_iou=wi,
        miou=miou,
        boundary_f1=bf,
        objective=objective,
        mse=(land + qm - 2 * intersection) / pixels,
        missing=missing,
        counts=out,
    )


def shape_metrics_from_counts(out, qm, pixels):
    metrics = metrics_from_counts(out[:, :7], qm, pixels)
    edges = out[:, 3].astype(np.float64)
    qedges = out[:, 4].astype(np.float64)
    soft = []
    for start in [7, 9]:
        precision = np.divide(out[:, start], edges * 1e6, out=np.ones_like(edges), where=edges > 0)
        recall = np.divide(out[:, start + 1], qedges * 1e6, out=np.ones_like(edges), where=qedges > 0)
        score = np.divide(
            2 * precision * recall, precision + recall, out=np.zeros_like(edges), where=precision + recall > 0
        )
        score[(edges == 0) != (qedges == 0)] = 0
        soft.append(score)
    contour = (soft[0] + soft[1]) / 2
    objective = 0.70 * metrics["miou"] + 0.25 * contour + 0.05 * metrics["boundary_f1"]
    objective[metrics["missing"] > 0] = -1
    metrics.update(
        soft_contour_f1=contour,
        soft_contour_f1_8=soft[0],
        soft_contour_f1_32=soft[1],
        precision_objective=objective,
        shape_counts=out,
    )
    return metrics


def structure_metrics_from_counts(out, qm, width, height, border_y, edge_weight=0.10):
    """Exact content-area/contour scores plus four content-border errors."""
    valid_height = height - 2 * border_y
    if border_y < 0 or valid_height <= 0:
        raise ValueError("Invalid content bounds")
    metrics = shape_metrics_from_counts(out[:, :13], qm, width * valid_height)
    errors = out[:, 13:17].astype(np.float64) / np.array([width, width, valid_height, valid_height])
    worst = errors.max(axis=1)
    objective = metrics["precision_objective"] - edge_weight * worst
    objective[metrics["missing"] > 0] = -1
    metrics.update(
        structure_objective=objective,
        frame_edge_error=worst,
        frame_edge_errors=errors,
        structure_counts=out,
        content_border_y=np.full(len(out), border_y, np.int32),
        evaluated_pixels=np.full(len(out), width * valid_height, np.int64),
    )
    return metrics


class CudaAtlas:
    def __init__(self, manifest=ROOT / "cache/coastline-native-rle/manifest.json", load_globe=True):
        build = json.loads((ROOT / "cuda/coastline/build/current.json").read_text(encoding="utf-8-sig"))
        cuda = Path(build["cuda_root"])
        self._dll_directories = (
            [os.add_dll_directory(str(p)) for p in [cuda / "bin", cuda / "bin/x64"] if p.is_dir()]
            if os.name == "nt"
            else []
        )
        self.library = ctypes.CDLL(build["library"])
        self.build = build
        self.library.coast_cuda_error.restype = ctypes.c_char_p
        self.library.coast_cuda_create.restype = ctypes.c_void_p
        self.library.coast_cuda_destroy.argtypes = [ctypes.c_void_p]
        self.library.coast_cuda_add_tile.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p] + [
            ctypes.c_double
        ] * 4
        self.library.coast_cuda_finish_tiles.argtypes = [ctypes.c_void_p]
        double = ctypes.POINTER(ctypes.c_double)
        byte = ctypes.POINTER(ctypes.c_uint8)
        signed = ctypes.POINTER(ctypes.c_int8)
        self.library.coast_cuda_sample.argtypes = [
            ctypes.c_void_p,
            double,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            byte,
        ]
        self.library.coast_cuda_features.argtypes = (
            [ctypes.c_void_p, double] + [ctypes.c_int] * 4 + [ctypes.c_double, signed, byte]
        )
        self.library.coast_cuda_queries.argtypes = [ctypes.c_void_p, signed, ctypes.c_int, ctypes.c_int]
        self.library.coast_cuda_start_region.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.library.coast_cuda_rank.argtypes = [ctypes.c_void_p, signed, byte, ctypes.c_int, ctypes.c_int]
        self.library.coast_cuda_region_results.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int32),
        ]
        self.library.coast_cuda_refine.argtypes = (
            [ctypes.c_void_p, double, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
            + [ctypes.c_int] * 4
            + [ctypes.c_double, ctypes.POINTER(ctypes.c_float)]
        )
        self.library.coast_cuda_score.argtypes = [
            ctypes.c_void_p,
            double,
            ctypes.c_int,
            byte,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint64),
        ]
        if hasattr(self.library, "coast_cuda_score_shape"):
            self.library.coast_cuda_score_shape.argtypes = self.library.coast_cuda_score.argtypes
        if hasattr(self.library, "coast_cuda_refine_topk"):
            self.library.coast_cuda_refine_topk.argtypes = self.library.coast_cuda_refine.argtypes[:-1] + [
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_float),
            ]
        if hasattr(self.library, "coast_cuda_native_rank_profiles"):
            self.library.coast_cuda_native_rank_profiles.argtypes = (
                [ctypes.c_void_p, ctypes.c_void_p, double] + [ctypes.c_int] * 4 + [ctypes.c_double] + [ctypes.c_int] * 3
            )
        if hasattr(self.library, "coast_cuda_sample_mirrored"):
            integer = ctypes.POINTER(ctypes.c_int32)
            self.library.coast_cuda_sample_mirrored.argtypes = (
                [ctypes.c_void_p, double, integer] + [ctypes.c_int] * 3 + [ctypes.c_double, byte]
            )
            self.library.coast_cuda_refine_mirrored.argtypes = (
                [ctypes.c_void_p, double, integer, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
                + [ctypes.c_int] * 4
                + [ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
            )
            self.library.coast_cuda_score_shape_mirrored.argtypes = [
                ctypes.c_void_p,
                double,
                integer,
                ctypes.c_int,
                byte,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_double,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_uint64),
            ]
        if hasattr(self.library, "coast_cuda_score_structure"):
            self.library.coast_cuda_score_structure.argtypes = [
                ctypes.c_void_p,
                double,
                ctypes.POINTER(ctypes.c_int32),
                ctypes.c_int,
                byte,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_double,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_uint64),
            ]
        self.handle = self.library.coast_cuda_create()
        if not self.handle:
            raise RuntimeError(self.error())
        self.query_count = self.top_k = 0
        if load_globe:
            manifest = Path(manifest)
            self.metadata = json.loads(manifest.read_text(encoding="utf-8"))
            for i, tile in enumerate(self.metadata["tiles"]):
                left, bottom, right, top = tile["bounds"]
                col = round((left + 180) / 30)
                row = round((90 - top) / 30)
                path = manifest.parent / (tile["file"] + ".rle")
                self.check(
                    self.library.coast_cuda_add_tile(
                        self.handle, row * 12 + col, str(path).encode(), left, bottom, right, top
                    )
                )
                if (i + 1) % 12 == 0:
                    print(f"CUDA original globe: {i+1}/72 tiles", flush=True)
            self.check(self.library.coast_cuda_finish_tiles(self.handle))

    def error(self):
        return self.library.coast_cuda_error().decode(errors="replace")

    def check(self, status):
        if status:
            raise RuntimeError(self.error())

    @staticmethod
    def pointer(array, kind):
        return array.ctypes.data_as(ctypes.POINTER(kind))

    @staticmethod
    def split_geometry(geometry):
        geometry = np.asarray(geometry, dtype=np.float64)
        if geometry.ndim == 1:
            geometry = geometry.reshape(1, -1)
        if geometry.ndim != 2 or geometry.shape[1] not in (4, 5):
            raise ValueError("Expected longitude,latitude,width,rotation[,mirror]")
        mirrors = (
            np.ascontiguousarray(geometry[:, 4], dtype=np.int32)
            if geometry.shape[1] == 5
            else np.zeros(len(geometry), np.int32)
        )
        if geometry.shape[1] == 5 and not np.isin(geometry[:, 4], [0, 1]).all():
            raise ValueError("Mirror flags must be 0 or 1")
        return geometry, np.ascontiguousarray(geometry[:, :4]), mirrors

    def close(self):
        if self.handle:
            self.library.coast_cuda_destroy(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def sample(self, geometry, size=(260, 195), aspect=0.0):
        geometry, source, mirrors = self.split_geometry(geometry)
        w, h = size
        out = np.empty((len(geometry), h, w), np.uint8)
        if mirrors.any():
            self.check(
                self.library.coast_cuda_sample_mirrored(
                    self.handle,
                    self.pointer(source, ctypes.c_double),
                    self.pointer(mirrors, ctypes.c_int32),
                    len(geometry),
                    w,
                    h,
                    aspect,
                    self.pointer(out, ctypes.c_uint8),
                )
            )
        else:
            self.check(
                self.library.coast_cuda_sample(
                    self.handle,
                    self.pointer(source, ctypes.c_double),
                    len(geometry),
                    w,
                    h,
                    aspect,
                    self.pointer(out, ctypes.c_uint8),
                )
            )
        return out

    def features(self, geometry, size=(64, 48), aa=4, aspect=1080 / 1444):
        geometry = np.ascontiguousarray(geometry, dtype=np.float64).reshape(-1, 4)
        w, h = size
        out = np.empty((len(geometry), w * h), np.int8)
        valid = np.empty(len(geometry), np.uint8)
        self.check(
            self.library.coast_cuda_features(
                self.handle,
                self.pointer(geometry, ctypes.c_double),
                len(geometry),
                w,
                h,
                aa,
                aspect,
                self.pointer(out, ctypes.c_int8),
                self.pointer(valid, ctypes.c_uint8),
            )
        )
        return out, valid

    def queries(self, features):
        features = np.ascontiguousarray(features, dtype=np.int8)
        self.query_count, self.dimension = features.shape
        self.check(self.library.coast_cuda_queries(self.handle, self.pointer(features, ctypes.c_int8), *features.shape))

    def start_region(self, k=4):
        self.top_k = k
        self.check(self.library.coast_cuda_start_region(self.handle, k))

    def rank(self, features, valid=None, offset=0):
        features = np.ascontiguousarray(features, dtype=np.int8)
        valid = np.ones(len(features), np.uint8) if valid is None else np.ascontiguousarray(valid, dtype=np.uint8)
        self.check(
            self.library.coast_cuda_rank(
                self.handle,
                self.pointer(features, ctypes.c_int8),
                self.pointer(valid, ctypes.c_uint8),
                len(features),
                offset,
            )
        )

    def region_results(self):
        scores = np.empty((self.query_count, self.top_k), np.float32)
        indices = np.empty(scores.shape, np.int32)
        self.check(
            self.library.coast_cuda_region_results(
                self.handle, self.pointer(scores, ctypes.c_float), self.pointer(indices, ctypes.c_int32)
            )
        )
        return scores, indices

    def rank_native_profiles(
        self, contour_context, geometry, size=(96, 72), aa=4, offset=0, area=True, contours=True, aspect=1080 / 1444
    ):
        geometry = np.ascontiguousarray(geometry, dtype=np.float64).reshape(-1, 4)
        w, h = size
        self.check(
            self.library.coast_cuda_native_rank_profiles(
                self.handle,
                contour_context.handle,
                self.pointer(geometry, ctypes.c_double),
                len(geometry),
                w,
                h,
                aa,
                aspect,
                offset,
                int(area),
                int(contours),
            )
        )

    def refine(self, geometry, query, margin, aspect=1080 / 1444):
        return self.refine_topk(geometry, query, margin, 1, 0, aspect)

    def score(self, geometry, query, aspect=1080 / 1444, tolerance=3):
        geometry, source, mirrors = self.split_geometry(geometry)
        if mirrors.any():
            return self.score_shape(geometry, query, aspect, tolerance)
        query = np.ascontiguousarray(query, dtype=np.uint8)
        h, w = query.shape
        out = np.empty((len(geometry), 7), np.uint64)
        self.check(
            self.library.coast_cuda_score(
                self.handle,
                self.pointer(source, ctypes.c_double),
                len(geometry),
                self.pointer(query, ctypes.c_uint8),
                w,
                h,
                aspect,
                tolerance,
                self.pointer(out, ctypes.c_uint64),
            )
        )
        return metrics_from_counts(out, float(query.sum()), w * h)

    def score_shape(self, geometry, query, aspect=1080 / 1444, tolerance=3):
        geometry, source, mirrors = self.split_geometry(geometry)
        query = np.ascontiguousarray(query, dtype=np.uint8)
        h, w = query.shape
        out = np.empty((len(geometry), 13), np.uint64)
        if mirrors.any():
            self.check(
                self.library.coast_cuda_score_shape_mirrored(
                    self.handle,
                    self.pointer(source, ctypes.c_double),
                    self.pointer(mirrors, ctypes.c_int32),
                    len(geometry),
                    self.pointer(query, ctypes.c_uint8),
                    w,
                    h,
                    aspect,
                    tolerance,
                    self.pointer(out, ctypes.c_uint64),
                )
            )
        else:
            self.check(
                self.library.coast_cuda_score_shape(
                    self.handle,
                    self.pointer(source, ctypes.c_double),
                    len(geometry),
                    self.pointer(query, ctypes.c_uint8),
                    w,
                    h,
                    aspect,
                    tolerance,
                    self.pointer(out, ctypes.c_uint64),
                )
            )
        return shape_metrics_from_counts(out, float(query.sum()), w * h)

    def score_structure(self, geometry, query, aspect=1080 / 1444, tolerance=3, border_y=2, edge_weight=0.10):
        geometry, source, mirrors = self.split_geometry(geometry)
        query = np.ascontiguousarray(query, dtype=np.uint8)
        h, w = query.shape
        if border_y < 0 or 2 * border_y >= h:
            raise ValueError("Invalid content bounds")
        out = np.empty((len(geometry), 17), np.uint64)
        self.check(
            self.library.coast_cuda_score_structure(
                self.handle,
                self.pointer(source, ctypes.c_double),
                self.pointer(mirrors, ctypes.c_int32),
                len(geometry),
                self.pointer(query, ctypes.c_uint8),
                w,
                h,
                aspect,
                tolerance,
                border_y,
                self.pointer(out, ctypes.c_uint64),
            )
        )
        region = query[border_y : h - border_y] if border_y else query
        return structure_metrics_from_counts(out, int(region.sum()), w, h, border_y, edge_weight)

    def refine_topk(self, geometry, query, margin, top_k=3, separation=3, aspect=1080 / 1444):
        geometry, source, mirrors = self.split_geometry(geometry)
        query = np.ascontiguousarray(query, dtype=np.float32)
        h, w = query.shape
        mx, my = (margin, margin) if np.isscalar(margin) else margin
        result = np.empty((len(geometry), top_k, 3), np.float32)
        if mirrors.any():
            self.check(
                self.library.coast_cuda_refine_mirrored(
                    self.handle,
                    self.pointer(source, ctypes.c_double),
                    self.pointer(mirrors, ctypes.c_int32),
                    len(geometry),
                    self.pointer(query, ctypes.c_float),
                    w,
                    h,
                    mx,
                    my,
                    aspect,
                    top_k,
                    separation,
                    self.pointer(result, ctypes.c_float),
                )
            )
        elif top_k == 1:
            self.check(
                self.library.coast_cuda_refine(
                    self.handle,
                    self.pointer(source, ctypes.c_double),
                    len(geometry),
                    self.pointer(query, ctypes.c_float),
                    w,
                    h,
                    mx,
                    my,
                    aspect,
                    self.pointer(result, ctypes.c_float),
                )
            )
        else:
            self.check(
                self.library.coast_cuda_refine_topk(
                    self.handle,
                    self.pointer(source, ctypes.c_double),
                    len(geometry),
                    self.pointer(query, ctypes.c_float),
                    w,
                    h,
                    mx,
                    my,
                    aspect,
                    top_k,
                    separation,
                    self.pointer(result, ctypes.c_float),
                )
            )
        result = result.reshape(-1, 3)
        shifted = np.repeat(geometry, top_k, axis=0)
        x = (result[:, 0] - mx) * shifted[:, 2] / w
        y = (result[:, 1] - my) * shifted[:, 2] * aspect / h
        x *= np.repeat(1 - 2 * mirrors, top_k)
        a = np.deg2rad(shifted[:, 3])
        c = np.cos(a)
        s = np.sin(a)
        shifted[:, 0] += (c * x - s * y) / (111.32 * np.cos(np.deg2rad(shifted[:, 1])))
        shifted[:, 1] += (-s * x - c * y) / 111.32
        return shifted, result[:, 2]
