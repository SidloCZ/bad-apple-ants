"""North-up native-10m context maps of the actual displayed photograph geometry."""

from __future__ import annotations
import ctypes, hashlib, json, os, time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from pyproj import CRS, Geod, Transformer
from pyproj.enums import TransformDirection
from badapple_media import ui_font
from .data import digest
from .footprint_map import footprint
from .pipeline import write_json

ROOT = Path(__file__).resolve().parents[1]
SIZE = (260, 162)
HEADER = 26
LOCAL_BOX = (1000, 780, 1260, 942)
WORLD_BOX = (1000, 608, 1260, 738)
MASK_BOX = (1000, 352, 1260, 547)
ORIGINAL_BOX = (1000, 92, 1260, 287)
FOOTER_BOX = (20, 810, 980, 930)
NAVY = (8, 15, 23)
GOLD = (255, 207, 108)
CYAN = (67, 174, 188)
GEOD = Geod(ellps="WGS84")


class NativePointSampler:
    """Rust/Rayon point sampler over the lossless 10 m RLE globe."""

    def __init__(self):
        os.environ.setdefault("RAYON_NUM_THREADS", "8")
        self.manifest = ROOT / "cache/coastline-native-rle/manifest.json"
        self.library_path = ROOT / "rust/coastline-native/target/release/coastline_native.dll"
        self.dll = ctypes.CDLL(str(self.library_path))
        self.dll.coastline_native_open.argtypes = [ctypes.c_char_p]
        self.dll.coastline_native_open.restype = ctypes.c_void_p
        self.dll.coastline_native_close.argtypes = [ctypes.c_void_p]
        self.dll.coastline_native_sample_points.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint8),
        ]
        self.handle = self.dll.coastline_native_open(str(self.manifest).encode())
        if not self.handle:
            raise ValueError("Cannot open verified native 10 m globe")

    def sample(self, longitude, latitude):
        longitude = (np.asarray(longitude) + 180) % 360 - 180
        coordinates = np.ascontiguousarray(np.stack([longitude, latitude], axis=-1), np.float64)
        output = np.empty(coordinates.shape[:-1], np.uint8)
        result = self.dll.coastline_native_sample_points(
            self.handle,
            coordinates.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            output.size,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        )
        if result or not np.isin(output, [0, 1]).all():
            raise ValueError("Local map has missing/invalid native land-water coverage")
        return output

    def close(self):
        if self.handle:
            self.dll.coastline_native_close(self.handle)
            self.handle = None


def centre(points):
    radians = np.deg2rad(np.asarray(points)[:, 0])
    return float(np.rad2deg(np.arctan2(np.sin(radians).mean(), np.cos(radians).mean()))), float(
        np.asarray(points)[:, 1].mean()
    )


def dense_boundary(points):
    """Follow the actual affine-geographic photo edges, including across the seam."""
    p = np.asarray(points, np.float64).copy()
    p[:, 0] = np.rad2deg(np.unwrap(np.deg2rad(p[:, 0])))
    t = np.arange(8) / 8
    return np.concatenate([p[k] + t[:, None] * (p[(k + 1) % 4] - p[k]) for k in range(4)])


def geometry(points, size=SIZE):
    lon, lat = centre(points)
    projection = CRS.from_proj4(f"+proj=aeqd +lat_0={lat:.12f} +lon_0={lon:.12f} +datum=WGS84 +units=m +no_defs")
    transformer = Transformer.from_crs("EPSG:4326", projection, always_xy=True)
    x, y = transformer.transform(np.asarray(points)[:, 0], np.asarray(points)[:, 1])
    xy = np.column_stack([x, y]) / 1000
    width = GEOD.inv(*points[0], *points[1])[2] / 1000
    # Five photograph widths, at least 20 km; leave margin even for polar distortion.
    span = max(20.0, 5 * width, 4 * np.abs(xy[:, 0]).max(), 4 * np.abs(xy[:, 1]).max() * size[0] / size[1])
    return transformer, lon, lat, float(span)


def map_xy(transformer, points, span, size=SIZE):
    p = np.asarray(points)
    x, y = transformer.transform(p[..., 0].ravel(), p[..., 1].ravel())
    xy = np.column_stack(
        [(np.asarray(x) / 1000 / span + 0.5) * size[0], size[1] / 2 - np.asarray(y) / 1000 / span * size[0]]
    )
    return xy.reshape(p.shape)


class LocalMaps:
    def __init__(self, records, size=SIZE, sampler=None):
        self.records = records
        self.size = size
        self.sampler = sampler or NativePointSampler()
        self.points = np.array([footprint(r) for r in records])
        self.centres = np.array([centre(p) for p in self.points])
        radians = np.deg2rad(self.centres)
        lon, lat = radians.T
        self.units = np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
        self.first = []
        seen = set()
        for i, p in enumerate(self.points):
            key = tuple(sorted(map(tuple, p.tolist())))
            if key not in seen:
                self.first.append(i)
                seen.add(key)
        self.first = np.asarray(self.first)
        self.dense = np.array([dense_boundary(p) for p in self.points[self.first]])
        cp = self.points[self.first]
        cs = self.centres[self.first]
        _, _, distance = GEOD.inv(
            np.repeat(cs[:, 0], 4), np.repeat(cs[:, 1], 4), cp[:, :, 0].ravel(), cp[:, :, 1].ravel()
        )
        self.radii = np.asarray(distance).reshape(-1, 4).max(1) / 1000
        self.fonts = {n: ui_font(n) for n in [8, 10, 11, 12, 14]}

    @lru_cache(maxsize=24)
    def base(self, key):
        points = np.asarray(key).reshape(4, 2)
        transform, lon, lat, span = geometry(points, self.size)
        width, height = self.size
        xx, yy = np.meshgrid(
            (np.arange(width) + 0.5 - width / 2) * span * 1000 / width,
            (height / 2 - np.arange(height) - 0.5) * span * 1000 / width,
        )
        longitude, latitude = transform.transform(xx, yy, direction=TransformDirection.INVERSE)
        values = self.sampler.sample(longitude, latitude)
        colors = np.array([[12, 29, 42], [46, 67, 66]], np.uint8)
        image = Image.fromarray(colors[values])
        draw = ImageDraw.Draw(image)
        for q in [0.25, 0.5, 0.75]:
            draw.line((int(width * q), 0, int(width * q), height), fill=(36, 53, 60), width=1)
        for q in [0.25, 0.5, 0.75]:
            draw.line((0, int(height * q), width, int(height * q)), fill=(36, 53, 60), width=1)
        return image, transform, lon, lat, span

    def render(self, index):
        points = self.points[index]
        width, height = self.size
        image, transform, lon, lat, span = self.base(tuple(points.ravel()))
        image = image.copy()
        draw = ImageDraw.Draw(image)
        # Conservative sphere test; retain EVERY earlier unique footprint in reach.
        unit = self.units[index]
        dot = (self.units[self.first] * unit).sum(1)
        distances = 6371.01 * np.arccos(np.clip(dot, -1, 1))
        nearby = np.flatnonzero(
            (self.first < index) & (distances < 1.02 * (span * np.hypot(1, height / width) / 2 + self.radii) + 2)
        )
        if len(nearby):
            past = map_xy(transform, self.dense[nearby], span, self.size)
            for polygon in past:
                if np.isfinite(polygon).all():
                    draw.line([*map(tuple, polygon), tuple(polygon[0])], fill=(44, 118, 130), width=1)
        boundary = map_xy(transform, dense_boundary(points), span, self.size)
        draw.line([*map(tuple, boundary), tuple(boundary[0])], fill=(8, 15, 23), width=4, joint="curve")
        draw.line([*map(tuple, boundary), tuple(boundary[0])], fill=GOLD, width=2, joint="curve")
        draw.line((width - 14, 26, width - 14, 11), fill="#bdcfdb", width=1)
        draw.line((width - 18, 16, width - 14, 11, width - 10, 16), fill="#bdcfdb", width=1)
        draw.text((width - 18, 0), "N", font=self.fonts[10], fill="#bdcfdb")
        target = span / 4
        power = 10 ** np.floor(np.log10(target))
        bar = max(v * power for v in [1, 2, 5] if v * power <= target)
        pixels = width * bar / span
        draw.line((9, height - 10, 9 + pixels, height - 10), fill="#bdcfdb", width=2)
        draw.text((9, height - 26), f"{bar:g} km", font=self.fonts[10], fill="#bdcfdb")
        draw.rectangle((0, 0, width - 1, height - 1), outline=(71, 95, 109), width=1)
        panel = Image.new("RGB", (width, height + HEADER), NAVY)
        panel.paste(image, (0, HEADER))
        d = ImageDraw.Draw(panel)
        d.text((0, 2), "LOCAL", font=self.fonts[14], fill="#73e6d7")
        label = f"{span:.1f} km across" if span < 1000 else f"{span:.0f} km across"
        d.text((width - d.textlength(label, font=self.fonts[11]), 5), label, font=self.fonts[11], fill="#9eafc0")
        meta = dict(
            frame=index,
            centre_lonlat=[lon, lat],
            span_km=span,
            nearby_past_unique_views=len(nearby),
            current_corners_lonlat=points.tolist(),
            current_corners_map_xy=map_xy(transform, points, span, self.size).tolist(),
            photograph_sha256=self.records[index]["satellite"]["image_sha256"],
        )
        return np.asarray(panel), meta


class LocalMapCache:
    def __init__(self, path, metadata):
        self.metadata = metadata
        self.path = Path(path)
        self.frames = np.memmap(path, np.uint8, "r", shape=(metadata["frames"], SIZE[1] + HEADER, SIZE[0], 3))
        self.visited = np.array(metadata["visited_counts"])
        self.static = Image.new("RGB", (260, 666), NAVY)
        d = ImageDraw.Draw(self.static)
        d.text((0, 28), "MATCHED MASK", font=ui_font(14), fill="#73e6d7")
        d.text((0, 286), "WORLD MAP", font=ui_font(14), fill="#73e6d7")
        d.text((0, 446), "Gold=current; cyan=past; small world boxes enlarged", font=ui_font(8), fill="#9eafc0")
        self.static = np.asarray(self.static)
        self.headings = {}

    def compose(self, frame, index, mirror=False):
        # Capture both before moving anything: their old and new boxes overlap.
        mask = frame[352:547, 1000:1260].copy()
        world = frame[608:738, 1000:1260].copy()
        # Fit the COMPLETE footer below the photo, retaining its aspect ratio.
        # It occupies only the left column; no panel or caption can cover it.
        footer = Image.fromarray(frame[800:960]).resize((960, 120), Image.Resampling.LANCZOS)
        sidebar = self.static.copy()
        sidebar[58:253] = mask
        sidebar[314:444] = world
        sidebar[460:648] = self.frames[index]
        # Tiny changing labels are cached, without touching the original comparison.
        key = (int(self.visited[index]), bool(mirror))
        if key not in self.headings:
            if len(self.headings) > 64:
                self.headings.clear()
            labels = []
            for text in ["MIRRORED" if mirror else "TRUE VIEW", f"USED {key[0]:04d}"]:
                strip = Image.new("RGB", (95, 18), NAVY)
                ImageDraw.Draw(strip).text((0, 1), text, font=ui_font(11), fill="#9eafc0")
                labels.append(np.asarray(strip))
            self.headings[key] = labels
        labels = self.headings[key]
        sidebar[28:46, 165:] = labels[0]
        sidebar[286:304, 165:] = labels[1]
        frame[800:960] = NAVY
        frame[810:930, 20:980] = np.asarray(footer)
        frame[294:960, 1000:1260] = sidebar
        return frame


def prepare(records, output, workers=4):
    output = Path(output)
    assets = output / "render-assets"
    assets.mkdir(parents=True, exist_ok=True)
    path = assets / "local-maps.u8"
    manifest = output / "local-maps.json"
    identity = dict(
        schema="native10m-local-aeqd-v1",
        size=SIZE,
        header=HEADER,
        native_source_manifest_sha256=digest(ROOT / "cache/coastline-native-rle/manifest.json"),
        renderer_sha256=digest(Path(__file__)),
        sampler_sha256=digest(ROOT / "rust/coastline-native/target/release/coastline_native.dll"),
        views=[dict(corners=footprint(r).tolist(), photo=r["satellite"]["image_sha256"]) for r in records],
    )
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    shape = (len(records), SIZE[1] + HEADER, SIZE[0], 3)
    if manifest.is_file() and path.is_file():
        old = json.loads(manifest.read_text())
        if old["identity"] == key and path.stat().st_size == int(np.prod(shape)):
            return LocalMapCache(path, old)
    renderer = LocalMaps(records)
    start = time.perf_counter()
    metadata = []
    frames = np.memmap(path, np.uint8, "w+", shape=shape)
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, (panel, info) in enumerate(pool.map(renderer.render, range(len(records)))):
                frames[i] = panel
                metadata.append(info)
                if (i + 1) % 100 == 0 or i + 1 == len(records):
                    elapsed = time.perf_counter() - start
                    write_json(
                        output / "status.json",
                        dict(phase="native10m-local-maps", complete=i + 1, total=len(records), elapsed_seconds=elapsed),
                    )
                    print(f"Native10m local maps: {i+1}/{len(records)}; {elapsed:.1f}s", flush=True)
        frames.flush()
    finally:
        renderer.sampler.close()
    result = dict(
        identity=key,
        frames=len(records),
        source={k: v for k, v in identity.items() if k != "views"},
        local_box=list(LOCAL_BOX),
        world_box=list(WORLD_BOX),
        mask_box=list(MASK_BOX),
        original_box=list(ORIGINAL_BOX),
        footer_box=list(FOOTER_BOX),
        footer_source_box=[0, 800, 1280, 960],
        footer_treatment="Complete original footer proportionally resized 0.75x, placed only below the left photo; no cropping or overlap",
        projection="WGS84 azimuthal equidistant centred on actual photograph; north up; equal metric pixel scale",
        native_source="G.E. Kindermann global land/water mask, 0.3 arcsec; lossless original pixels, nearest neighbour",
        span_rule="At least five actual photograph widths and 20 km; polar extent can enlarge context",
        current="Gold true photograph footprint, not enlarged",
        history="All preceding unique nearby photo footprints, no future views or truncation",
        visited_counts=np.searchsorted(renderer.first, np.arange(len(records)), side="right").tolist(),
        views=metadata,
        elapsed_seconds=time.perf_counter() - start,
    )
    write_json(manifest, result)
    del frames
    return LocalMapCache(path, result)
