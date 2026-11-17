from __future__ import annotations

import concurrent.futures as futures
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .geometry import apply_quadratic, ortho_inverse
from .ingest import GDAL_ENV

WIDTH, HEIGHT = 1944, 1458  # crop size; MoonLayout resizes it to the moon view (1899x1424 at 2560x1920)
WAC_FILE = "assets/moon/raw/wac_100m.tif"
WAC_METRES = 100.0
WAC_STRETCH = (1.0, 193.0)  # 1-99 % of the global mosaic (cache/moon/tiles/wac.json)


CROP_VERSION = 5  # bump when the crop rendering itself changes
TONE_GAMMA = 1.3
MAX_HOLE_PX = 64  # nodata specks up to this size inside an image are filled, not treated as gaps
SAT_DN = 2750  # TC calibrated DN at which the detector saturates
MAX_SATURATED = 0.02  # crops with more saturated area than this fall back to the next alternative
LOCAL_SIZE = (600, 450)  # local context map (pixels at 2560x1920)
LOCAL_SPAN = 5.0  # local map width as a multiple of the patch width
LOCAL_MIN_KM = 20.0


def crop_key(best: dict) -> str:
    fields = [CROP_VERSION] + [best[k] for k in ("raster", "a", "b", "width_km", "theta_deg", "mirror")]
    return hashlib.sha1(json.dumps(fields).encode()).hexdigest()[:16]


def _output_grid(best: dict, ss: int, size: tuple[int, int] = (WIDTH, HEIGHT)):
    """Patch-frame km of every supersampled output pixel, rotated into the map frame."""
    w, h = size[0] * ss, size[1] * ss
    width = best["width_km"]
    u = ((np.arange(w) + 0.5) / w - 0.5) * width
    v = (0.5 - (np.arange(h) + 0.5) / h) * 0.75 * width
    uu, vv = np.meshgrid(u, v)
    if best["mirror"]:  # the frame was mirrored to match, so mirror the surface back for display
        uu = -uu
    t = math.radians(best["theta_deg"])
    return uu * math.cos(t) - vv * math.sin(t), uu * math.sin(t) + vv * math.cos(t)


def _fill_nodata(data: np.ndarray, valid: np.ndarray):
    """Give every nodata pixel its nearest valid value, so cubic interpolation never rings around a
    -32768 speck (that drew small bright squares in shadows). Specks up to MAX_HOLE_PX count as
    valid afterwards; larger gaps stay invalid."""
    from scipy import ndimage

    invalid = ~valid
    if not invalid.any() or valid.sum() == 0:
        return data, valid
    labels, count = ndimage.label(invalid)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    small = (sizes <= MAX_HOLE_PX)[labels] & invalid
    nearest = ndimage.distance_transform_edt(invalid, return_distances=False, return_indices=True)
    return data[tuple(nearest)], valid | small


def _sample(data: np.ndarray, valid: np.ndarray, rows: np.ndarray, cols: np.ndarray):
    from scipy.ndimage import map_coordinates

    data, valid = _fill_nodata(data, valid)
    coords = [rows.ravel() - 0.5, cols.ravel() - 0.5]  # pixel-edge -> pixel-centre indices
    values = map_coordinates(data, coords, order=3, mode="nearest").reshape(rows.shape)
    ok = map_coordinates(valid.astype(np.float32), coords, order=1, mode="constant", cval=0).reshape(rows.shape) > 0.999
    return values, ok


def _native_km(best: dict, record: dict | None) -> float:
    if best["source"] != "tc":
        return WAC_METRES / 1000
    fwd = np.asarray(record["forward"])
    return 1 / math.sqrt(abs(fwd[0][1] * fwd[1][2] - fwd[0][2] * fwd[1][1]))


def _read_tc(best: dict, record: dict, dx, dy, out_km: float):
    import rasterio

    x, y = best["a"] + dx, best["b"] + dy
    cols, rows = apply_quadratic(np.asarray(record["forward"]), x, y)
    native_km = _native_km(best, record)
    url = "/vsicurl/" + record["href"]
    with rasterio.Env(**GDAL_ENV):
        with rasterio.open(url) as full:
            overviews = full.overviews(1)
            fw, fh = full.width, full.height
        level = -1
        for i, factor in enumerate(overviews):
            if native_km * factor <= out_km:
                level = i
        kwargs = {} if level < 0 else {"overview_level": level}
        with rasterio.open(url, **kwargs) as src:
            sx, sy = src.width / fw, src.height / fh
            c, r = cols * sx, rows * sy
            c0, r0 = max(int(c.min()) - 3, 0), max(int(r.min()) - 3, 0)
            c1, r1 = min(int(c.max()) + 4, src.width), min(int(r.max()) + 4, src.height)
            window = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
            data = src.read(1, window=window)
            nodata = src.nodata
    valid = data != nodata
    return _sample(data.astype(np.float32), valid, r - r0, c - c0)


def _read_wac(best: dict, dx, dy, out_km: float):
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import Window

    lon, lat = ortho_inverse(dx, dy, math.degrees(best["a"]), math.degrees(best["b"]))
    with rasterio.open(WAC_FILE) as src:
        W, H = src.width, src.height
        col = (lon + 180) / 360 * W
        centre = (math.degrees(best["a"]) + 180) / 360 * W
        col = centre + ((col - centre + W / 2) % W - W / 2)  # unwrap across the seam
        row = (90 - lat) / 180 * H
        f = max(1, int(out_km / (WAC_METRES / 1000)))
        c0, c1 = int(math.floor(col.min())) - 3 * f, int(math.ceil(col.max())) + 3 * f
        r0, r1 = max(int(math.floor(row.min())) - 3 * f, 0), min(int(math.ceil(row.max())) + 3 * f, H)
        c1 = c0 + (c1 - c0 + f - 1) // f * f
        r1 = r0 + max((r1 - r0) // f, 1) * f
        pieces, start = [], c0
        while start < c1:  # split at the +-180 seam
            s = start % W
            n = min(c1 - start, W - s)
            pieces.append(
                src.read(
                    1,
                    window=Window(s, r0, n, r1 - r0),
                    out_shape=(max((r1 - r0) // f, 1), max(n // f, 1)),
                    resampling=Resampling.average,
                )
            )
            start += n
        data = np.concatenate(pieces, axis=1)
    sx = data.shape[1] / (c1 - c0)
    sy = data.shape[0] / (r1 - r0)
    return _sample(data.astype(np.float32), data > 0, (row - r0) * sy, (col - c0) * sx)


def render_crop(best: dict, record: dict | None) -> np.ndarray:
    # 2x2 supersampling only where an output pixel covers more than one source pixel.
    s = 2 if best["width_km"] / WIDTH > _native_km(best, record) else 1
    dx, dy = _output_grid(best, s)
    out_km = best["width_km"] / (WIDTH * s)
    values, ok = _read_tc(best, record, dx, dy, out_km) if best["source"] == "tc" else _read_wac(best, dx, dy, out_km)
    values = values.reshape(HEIGHT, s, WIDTH, s).mean(axis=(1, 3))
    ok = ok.reshape(HEIGHT, s, WIDTH, s).all(axis=(1, 3))
    if ok.mean() < 0.5:
        raise ValueError(f"crop mostly outside data ({ok.mean():.2f})")
    # Black point: the larger of the source image's 1 % (as at ingest) and this crop's 1 %, so shadows
    # stay black. White point: this crop's highlights, but never below 20 % of the image's range so
    # noise inside a shadow is not amplified. A mild gamma keeps TC's onboard-compression blocks
    # in deep shadow out of sight.
    image_low, image_high = record["stretch"] if best["source"] == "tc" else WAC_STRETCH
    low = max(image_low, float(np.percentile(values[ok], 1)))
    high = max(float(np.percentile(values[ok], 99.5)), low + 0.2 * (image_high - image_low))
    out = np.clip((values - low) / max(high - low, 1e-6), 0, 1) ** TONE_GAMMA * 255
    out[~ok] = 0
    # TC saturates around 2,800-3,000 DN; such plateaus render as streaky flat white.
    saturated = float((values[ok] >= SAT_DN).mean()) if best["source"] == "tc" else 0.0
    return np.rint(out).astype(np.uint8), {"coverage": float(ok.mean()), "saturated": saturated}


def local_span_km(best: dict) -> float:
    return max(LOCAL_SPAN * best["width_km"], LOCAL_MIN_KM)


def render_local_map(best: dict) -> np.ndarray:
    """North-up orthographic WAC context around the patch centre, LOCAL_SPAN patch widths across."""
    span = local_span_km(best)
    centre = {
        "a": math.radians(best["lon"]),
        "b": math.radians(best["lat"]),
        "width_km": span,
        "theta_deg": 0.0,
        "mirror": False,
    }
    dx, dy = _output_grid(centre, 1, LOCAL_SIZE)
    values, ok = _read_wac(centre, dx, dy, span / LOCAL_SIZE[0])
    low, high = WAC_STRETCH
    out = np.clip((values - low) / (high - low), 0, 1) * 255
    out[~ok] = 0
    return np.rint(out).astype(np.uint8)


def footprint(best: dict, record: dict | None) -> np.ndarray:
    """The displayed patch outline as 4 corners (lon, lat) in degrees."""
    w, t = best["width_km"], math.radians(best["theta_deg"])
    u = np.array([-1, 1, 1, -1]) * w / 2
    v = np.array([1, 1, -1, -1]) * 0.375 * w
    dx, dy = u * math.cos(t) - v * math.sin(t), u * math.sin(t) + v * math.cos(t)
    if best["source"] == "tc":
        lon, lat = ortho_inverse(best["a"] + dx, best["b"] + dy, record["lon0"], record["lat0"])
    else:
        lon, lat = ortho_inverse(dx, dy, math.degrees(best["a"]), math.degrees(best["b"]))
    return np.stack([lon, lat], axis=1)


def _crop_job(args):
    """Render (or reuse) one crop and its local map; returns the crop's QA statistics."""
    best, record, path = args
    stats_path = path.with_suffix(".json")
    if path.exists() and path.with_suffix(".map.jpg").exists() and stats_path.exists():
        return json.loads(stats_path.read_text(encoding="utf-8"))
    image, stats = render_crop(best, record)
    Image.fromarray(image).save(path, quality=95)
    Image.fromarray(render_local_map(best)).save(path.with_suffix(".map.jpg"), quality=92)
    stats_path.write_text(json.dumps(stats), encoding="utf-8")
    return stats


def qa_failure(stats: dict) -> str | None:
    if stats["saturated"] > MAX_SATURATED:
        return f"saturated {stats['saturated']:.1%}"
    if stats["coverage"] < 0.999:
        return f"coverage {stats['coverage']:.1%}"
    return None


def tc_index_paths(tiles_root: Path = Path("cache/moon")) -> list[Path]:
    """The monoscopic index first, then every extra series shard (tiles-stereo-*, tiles-sp-*)."""
    return [tiles_root / "tiles" / "tc_index.jsonl"] + sorted(tiles_root.glob("tiles-*/tc_index.jsonl"))


def load_tc_records(index: Path | list[Path] | None = None) -> dict[str, dict]:
    paths = tc_index_paths() if index is None else [index] if isinstance(index, Path) else index
    records = {}
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    r = json.loads(line)
                    records.setdefault(r["id"], r)
    return records


def merge_matches(runs: list[dict]) -> dict:
    """Pool the refined candidates of several runs over the same queries. A larger search can lose a
    good candidate when its top-K lists fill up with entries later dropped, so the union is kept and
    the best 32x24 score wins. QA decisions are not carried over; crops redoes them."""
    first = runs[0]
    assert all(r["frames"] == first["frames"] for r in runs), "runs use different queries"
    merged = []
    for parts in zip(*(r["results"] for r in runs)):
        base = dict(parts[0])
        base.pop("qa", None)
        if base["kind"] == "match":
            pool, seen = [], set()
            for part in parts:
                for c in ([part["best"]] if part["best"] else []) + part.get("alternatives", []):
                    if crop_key(c) not in seen:
                        seen.add(crop_key(c))
                        pool.append(c)
            pool.sort(key=lambda c: -c["score32"])
            base["best"], base["alternatives"] = (pool[0] if pool else None), pool
        merged.append(base)
    gained = sum(
        1
        for m, p in zip(merged, first["results"])
        if m["kind"] == "match" and m["best"] and (p["best"] is None or crop_key(m["best"]) != crop_key(p["best"]))
    )
    print(f"merged {len(runs)} runs: {gained} uniques take a candidate from a later run")
    return {**first, "results": merged}


def make_crops(
    matches: dict,
    records: dict,
    crops: Path,
    workers: int = 24,
    matches_path: Path | None = None,
    fallback: dict | None = None,
) -> None:
    """Render every chosen surface; a crop that fails QA (saturation plateau, missing data) is replaced
    by that frame's next refined alternative, and the decision is written back into matches.json.
    `fallback` (an earlier, QA-checked matches.json for the same queries) supplies a last candidate
    for frames whose every alternative fails."""
    crops.mkdir(parents=True, exist_ok=True)
    previous = {r["unique"]: r["best"] for r in fallback["results"]} if fallback else {}
    # Ordered candidates per unique: the refined best first, then the other refined seeds.
    queue: dict[int, list[dict]] = {}
    for result in matches["results"]:
        best = result["best"]
        if best is None:
            if previous.get(result["unique"]) is None:
                continue
            best = result["best"] = previous[result["unique"]]
        options = [best] + [a for a in result.get("alternatives", []) if crop_key(a) != crop_key(best)]
        if result["kind"] != "match":
            options = [best]
        elif previous.get(result["unique"]) is not None:
            options.append(previous[result["unique"]])
        queue[result["unique"]] = options
        result.setdefault("qa", {"rejected": []})
    results = {r["unique"]: r for r in matches["results"]}
    pending = {u: 0 for u in queue}
    started, failures, rendered = time.time(), [], 0
    with futures.ProcessPoolExecutor(workers) as pool:
        while pending:
            jobs, owners = {}, {}
            for u, i in pending.items():
                cand = queue[u][i]
                path = crops / f"{crop_key(cand)}.jpg"
                owners.setdefault(path, []).append(u)
                if path not in jobs:
                    jobs[path] = pool.submit(_crop_job, (cand, records.get(cand["raster"]), path))
            next_pending = {}
            for path, task in jobs.items():
                try:
                    reason = qa_failure(task.result())
                except Exception as error:
                    reason = f"error {repr(error)[:120]}"
                for u in owners[path]:
                    i = pending[u]
                    if reason is None:
                        results[u]["best"] = queue[u][i]
                    else:
                        results[u]["qa"]["rejected"].append({"raster": queue[u][i]["raster"], "reason": reason})
                        if i + 1 < len(queue[u]):
                            next_pending[u] = i + 1
                        else:
                            failures.append((u, reason))  # keep the last candidate rather than nothing
                            results[u]["best"] = queue[u][i]
                rendered += 1
                if rendered % 100 == 0:
                    print(
                        f"\rcrops {rendered} ({rendered / (time.time() - started):.1f}/s), "
                        f"{len(next_pending)} replacements queued",
                        end="",
                        flush=True,
                    )
            pending = next_pending
    print()
    replaced = sum(1 for r in matches["results"] if r.get("qa", {}).get("rejected"))
    print(
        f"crops done: {rendered} rendered or reused, {replaced} uniques replaced, {len(failures)} without a clean candidate"
    )
    for failure in failures[:20]:
        print("no clean candidate:", *failure)
    if matches_path is not None:
        matches_path.write_text(json.dumps(matches), encoding="utf-8")


def preview(matches: dict, records: dict, video: Path, out: Path, frames: list[int]) -> None:
    """Original frame | matched surface, one row per requested frame (crops rendered on the fly)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    results = {r["unique"]: r for r in matches["results"]}
    tw, th = 320, 240
    sheet = Image.new("L", (2 * tw, th * len(frames)))
    for row, n in enumerate(frames):
        raw = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(video),
                "-vf",
                f"select=eq(n\\,{n}),scale={tw}:{th}",
                "-frames:v",
                "1",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "gray",
                "-",
            ],
            capture_output=True,
        ).stdout
        sheet.paste(Image.frombytes("L", (tw, th), raw), (0, row * th))
        result = results[matches["frames"][n]]
        image, _ = render_crop(result["best"], records.get(result["best"]["raster"]))
        sheet.paste(Image.fromarray(image).resize((tw, th), Image.BOX), (tw, row * th))
        b = result["best"]
        print(
            n,
            result["kind"],
            b["raster"],
            f"w={b['width_km']:.1f} th={b['theta_deg']:.0f} m={b['mirror']} ncc={b['score32']:.3f}",
        )
    sheet.save(out, quality=92)


# ---- Layout on one shared grid: moon view, right column (ORIGINAL, maps), footer ----
# Every text field has a fixed origin and baseline, a monospaced font and a fixed-width format, so a
# changing value never moves or resizes anything else from one frame to the next.

BASE = (1280, 960)
FOOTER_HEIGHT = 130
COLUMN = (26, 240)  # right column: margin from the right edge and width on the grid
COLUMN_GAP = 36  # vertical gap between the column's panels (room for each label)
NCC_BOX = (888, 12, 1244, 92)  # footer NCC graph (left, top, right, bottom on the grid; x right-anchored)
BACKGROUND = "#0c1018"
YELLOW = "#f5d76e"
TEAL = "#65d6c4"
GREY = "#919aa9"
RULE = "#657187"
FONTS_ROOT = Path("assets/fonts")
CREDITS = {
    "tc": "IMAGE  JAXA/SELENE  ·  USGS ASTROGEOLOGY ARD (CC0)",
    "wac": "IMAGE  NASA/GSFC/ARIZONA STATE UNIVERSITY  ·  LROC WAC",
}


def fmt_lat(lat: float) -> str:
    return f"{abs(lat):05.2f}°{'N' if lat >= 0 else 'S'}"


def fmt_lon(lon: float) -> str:
    return f"{abs(lon):06.2f}°{'E' if lon >= 0 else 'W'}"


def describe(result: dict, record: dict | None, gazetteer, subsolar: dict[str, list[float]]) -> dict:
    """All caption fields for one matched surface, formatted to fixed widths. The Sun elevation is computed at
    the matched place from the observation's sub-solar point (see sun.py), not taken from the STAC item."""
    from .sun import camera_line, local_sun

    best = result["best"]
    name, kind, edge, diameter = gazetteer.nearest(best["lon"], best["lat"])
    if best["source"] == "tc":
        camera = camera_line(*local_sun(best["lat"], best["lon"], subsolar[best["raster"]]))
        image = f"{best['raster']}  {record['gsd']:4.1f} m/px"
    else:
        camera = "LRO WIDE ANGLE CAMERA  GLOBAL MOSAIC"
        image = "LROC WAC GLOBAL MORPHOLOGY  100 m/px"
    where = "inside" if edge == 0 else f"{edge:5.1f} km from edge"
    view = (
        f"w {best['width_km']:5.1f} km  rot {best['theta_deg'] % 360:3.0f}°  {'mirror' if best['mirror'] else '      '}"
    )
    score = f"NCC {best['score32']:.3f}" if result["kind"] == "match" else "NCC   n/a"
    return {
        "coords": f"{fmt_lat(best['lat'])} {fmt_lon(best['lon'])}",
        "feature": f"{name}  ·  {kind}",
        "feature_detail": f"{where}  ·  Ø {diameter:6.1f} km",
        "camera": camera,
        "image": image,
        "view": view,
        "score": score,
        "credit": CREDITS[best["source"]],
        "local": f"LOCAL  {local_span_km(best):6.1f} km across",
    }


class MoonMaps:
    """Global equirectangular WAC map that keeps every footprint shown so far, plus a north-up
    local map at true scale with the current footprint and any earlier ones nearby."""

    def __init__(self, global_size: tuple[int, int], local_size: tuple[int, int], tiles: Path, s: float):
        self.gw, self.gh = global_size
        self.local_size = local_size
        self.s = s
        meta = json.loads((tiles / "wac.json").read_text(encoding="utf-8"))
        wac = np.fromfile(tiles / "wac.u8", np.uint8).reshape(meta["height"], meta["width"])
        base = Image.fromarray(wac).resize(global_size, Image.Resampling.BOX).convert("RGB")
        self.base = Image.eval(base, lambda v: int(v * 0.55))
        d = ImageDraw.Draw(self.base)
        for lon in range(-150, 180, 30):
            x = (lon + 180) / 360 * self.gw
            d.line((x, 0, x, self.gh), fill="#2b3341", width=1)
        for lat in range(-60, 90, 30):
            y = (90 - lat) / 180 * self.gh
            d.line((0, y, self.gw, y), fill="#2b3341", width=1)
        self.history = Image.new("RGBA", global_size, (0, 0, 0, 0))
        self.visited = 0
        self.past: list[np.ndarray] = []  # footprints (4x2 lon/lat) in display order
        self.past_centre = np.zeros((0, 2))

    def _global_xy(self, corners: np.ndarray, centre_lon: float):
        lon = centre_lon + (corners[:, 0] - centre_lon + 180) % 360 - 180  # unwrap at the seam
        return np.stack([(lon + 180) / 360 * self.gw, (90 - corners[:, 1]) / 180 * self.gh], axis=1)

    def add(self, corners: np.ndarray, lon: float, lat: float) -> None:
        """Commit a footprint to the history (call once per newly shown surface)."""
        d = ImageDraw.Draw(self.history)
        xy = self._global_xy(corners, lon)
        r = max(1.5, 1.0 * self.s)
        (x0, y0), (x1, y1) = xy.min(axis=0), xy.max(axis=0)
        if x1 - x0 < 2 * r and y1 - y0 < 2 * r:
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            d.rectangle((cx - r, cy - r, cx + r, cy + r), fill=(101, 214, 196, 150))
        else:
            d.polygon([tuple(p) for p in xy], fill=(101, 214, 196, 70), outline=(101, 214, 196, 200))
        self.past.append(corners)
        self.past_centre = np.vstack([self.past_centre, [lon, lat]])
        self.visited += 1

    def global_image(self, corners: np.ndarray, lon: float) -> Image.Image:
        image = self.base.copy()
        image.paste(self.history, (0, 0), self.history)
        d = ImageDraw.Draw(image)
        xy = self._global_xy(corners, lon)
        (x0, y0), (x1, y1) = xy.min(axis=0), xy.max(axis=0)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        half = max((x1 - x0) / 2, (y1 - y0) / 2, 5 * self.s)
        w = max(1, round(1.5 * self.s))
        d.rectangle((cx - half, cy - half * 0.75, cx + half, cy + half * 0.75), outline=YELLOW, width=w)
        tick = 6 * self.s
        for a, b in (
            ((cx - half - tick, cy), (cx - half, cy)),
            ((cx + half, cy), (cx + half + tick, cy)),
            ((cx, cy - half * 0.75 - tick), (cx, cy - half * 0.75)),
            ((cx, cy + half * 0.75), (cx, cy + half * 0.75 + tick)),
        ):
            d.line((a, b), fill=YELLOW, width=w)
        return image

    def local_image(self, background: Image.Image, corners: np.ndarray, lon: float, lat: float, span: float):
        from .geometry import ortho_forward

        image = background.convert("RGB")
        if image.size != self.local_size:
            image = image.resize(self.local_size, Image.Resampling.LANCZOS)
        lw, lh = self.local_size

        def project(c):
            x, y = ortho_forward(c[:, 0], c[:, 1], lon, lat)
            return [((xx / span + 0.5) * lw, (0.5 - yy / (0.75 * span)) * lh) for xx, yy in zip(x, y)]

        d = ImageDraw.Draw(image, "RGBA")
        if len(self.past):
            # Earlier footprints within reach of this view (great-circle distance below the span).
            plon, plat = np.radians(self.past_centre[:, 0]), np.radians(self.past_centre[:, 1])
            cos = np.sin(math.radians(lat)) * np.sin(plat) + np.cos(math.radians(lat)) * np.cos(plat) * np.cos(
                plon - math.radians(lon)
            )
            near = np.flatnonzero(1737.4 * np.arccos(np.clip(cos, -1, 1)) < span)[-300:]
            for i in near:
                d.polygon(project(self.past[i]), outline=(101, 214, 196, 210), width=max(1, round(self.s)))
        d.polygon(project(corners), outline=YELLOW, width=max(2, round(2 * self.s)))
        return image


class NccGraph:
    """Per-frame NCC of the frames played so far, on a time axis spanning the whole video; nothing ahead of the
    current frame is drawn. Each pixel column shows the min..max of its frames and a trailing one-second mean."""

    LOW, HIGH = 0.6, 1.0  # the ~1 % of scores below LOW sit on the bottom edge
    GRID = (0.7, 0.8, 0.9)
    SMOOTH = 30  # frames in the trailing mean

    def __init__(self, size: tuple[int, int], scores: list[float | None], s: float):
        w, h = size
        n = len(scores)
        self.size, self.s, self.scores = size, s, scores
        self.pad = round(3 * s)
        self.lw = max(1, round(1.5 * s))
        self.column = np.minimum(np.arange(n) * w // n, w - 1)
        self.first = np.searchsorted(self.column, np.arange(w))  # first frame of each column
        valid = np.array([v is not None for v in scores])
        values = np.array([v if v is not None else np.nan for v in scores], dtype=float)
        cs = np.concatenate([[0.0], np.cumsum(np.nan_to_num(values))])
        cv = np.concatenate([[0], np.cumsum(valid)])
        lo = np.maximum(np.arange(n) + 1 - self.SMOOTH, 0)
        self.trail = (cs[1:] - cs[lo]) / np.maximum(cv[1:] - cv[lo], 1)
        self.trail[~valid] = np.nan
        self.running_mean = cs[1:] / np.maximum(cv[1:], 1)
        self.valid_so_far = cv[1:]
        self.values = values
        self.base = Image.new("RGB", size, "#080b11")
        d = ImageDraw.Draw(self.base)
        for v in self.GRID:
            d.line((0, self.y(v), w, self.y(v)), fill="#1c2330", width=1)
        # Every complete column; image() shows the columns before the current one and draws that one itself.
        self.full = self.base.copy()
        d = ImageDraw.Draw(self.full)
        self.point: list[tuple[float, float] | None] = []
        for c in range(w):
            self.point.append(self._column(d, c, n - 1))

    def y(self, v: float) -> float:
        v = min(max(v, self.LOW), self.HIGH)
        return self.pad + (self.HIGH - v) / (self.HIGH - self.LOW) * (self.size[1] - 2 * self.pad)

    def _column(self, d, c: int, last: int) -> tuple[float, float] | None:
        """Draw column c from the frames up to `last`; returns its point on the mean line (None breaks the line)."""
        end = min(last, (self.first[c + 1] - 1) if c + 1 < len(self.first) else len(self.scores) - 1)
        v = self.values[self.first[c] : end + 1]
        v = v[~np.isnan(v)]
        if not len(v):
            return None
        d.line((c, self.y(v.max()), c, self.y(v.min())), fill="#24524b", width=1)
        t = self.trail[self.first[c] : end + 1]
        t = t[~np.isnan(t)]
        point = (c, self.y(t[-1]))
        previous = self.point[c - 1] if 0 < c <= len(self.point) else None
        if previous is not None:
            d.line((previous, point), fill=TEAL, width=self.lw)
        return point

    def image(self, index: int) -> Image.Image:
        c = int(self.column[index])
        image = self.base.copy()
        image.paste(self.full.crop((0, 0, c, self.size[1])), (0, 0))
        d = ImageDraw.Draw(image)
        point = self._column(d, c, index)
        if point is not None and self.scores[index] is not None:
            r = 2.5 * self.s
            d.ellipse((point[0] - r, point[1] - r, point[0] + r, point[1] + r), fill=YELLOW)
        return image

    def mean_label(self, index: int) -> str:
        return f"mean {self.running_mean[index]:.3f}" if self.valid_so_far[index] else "mean   -  "


class MoonLayout:
    """The 1280x960 grid scaled by height. The moon view and left-hand fields keep their offset from the left edge and
    the right column and right-hand fields from the right edge, so a wider frame only adds margin around the view."""

    def __init__(self, size: tuple[int, int]):
        from badapple_media import ui_font

        self.size = size
        self.s = size[1] / BASE[1]
        if size[0] < round(BASE[0] * self.s):
            raise ValueError("layout needs an aspect ratio of at least 4:3")
        self.footer_h = round(FOOTER_HEIGHT * self.s)
        self.footer_top = size[1] - self.footer_h
        self.mono = {
            k: ui_font(max(1, round(v * self.s)), FONTS_ROOT, monospace=True)
            for k, v in {"code": 36, "name": 22, "block": 17, "small": 14, "axis": 11}.items()
        }
        self.sans = ui_font(round(18 * self.s), FONTS_ROOT)
        self.sans_small = ui_font(round(11 * self.s), FONTS_ROOT)
        cx, cw = COLUMN
        # Right column: ORIGINAL (4:3), global map (2:1), local map (4:3), each under its label.
        x = self.R(BASE[0] - cx) - self.X(cw)
        y_global = 92 + cw * 3 / 4 + COLUMN_GAP
        y_local = y_global + cw / 2 + COLUMN_GAP
        y_bottom = BASE[1] - FOOTER_HEIGHT - 26
        self.inset_origin, self.inset_size = (x, self.Y(92)), (self.X(cw), self.X(cw * 3 / 4))
        self.global_origin, self.global_size = (x, self.Y(y_global)), (self.X(cw), self.X(cw / 2))
        self.local_origin, self.local_size = (x, self.Y(y_local)), (self.X(cw), self.X(cw * 3 / 4))
        # Moon view: the largest 4:3 box left of the column, above the footer.
        left, right, top, bottom = self.X(26), x - self.X(24), self.Y(92), self.Y(y_bottom)
        view_h = min(round((right - left) * 3 / 4), bottom - top)
        self.view_size = (round(view_h * 4 / 3), view_h)
        self.view_origin = (left + (right - left - self.view_size[0]) // 2, top)
        # NCC graph in the footer (footer coordinates), right-anchored above the score and frame counter.
        self.ncc_origin = (self.R(NCC_BOX[0]), self.Y(NCC_BOX[1]))
        self.ncc_size = (self.R(NCC_BOX[2]) - self.ncc_origin[0], self.Y(NCC_BOX[3]) - self.ncc_origin[1])
        self._footers: dict[int, Image.Image] = {}

    def X(self, x: float) -> int:  # left-anchored grid x
        return round(x * self.s)

    def R(self, x: float) -> int:  # right-anchored grid x
        return round(self.size[0] - (BASE[0] - x) * self.s)

    def Y(self, y: float) -> int:
        return round(y * self.s)

    def text(self, draw, text, x, baseline, kind, color, anchor="ls", max_px=None, right=False):
        font = self.mono[kind]
        if max_px is not None and draw.textlength(text, font=font) > max_px:
            while text and draw.textlength(text + "...", font=font) > max_px:
                text = text[:-1]
            text += "..."
        draw.text((self.R(x) if right else self.X(x), self.Y(baseline)), text, font=font, fill=color, anchor=anchor)

    def footer(self, key: int, f: dict) -> Image.Image:
        """Per-surface footer (cached); only the frame counter and the NCC graph are drawn per frame."""
        if key not in self._footers:
            if len(self._footers) > 256:
                self._footers.pop(next(iter(self._footers)))
            image = Image.new("RGB", (self.size[0], self.footer_h), BACKGROUND)
            d = ImageDraw.Draw(image)
            self.text(d, f["coords"], 36, 45, "code", YELLOW)
            self.text(d, f["feature"], 420, 24, "block", TEAL, max_px=self.R(NCC_BOX[0] - 40) - self.X(420))
            self.text(d, f["feature_detail"], 420, 48, "small", GREY)
            self.text(d, f["camera"], 36, 82, "name", "#ffffff", max_px=self.R(NCC_BOX[0] - 40) - self.X(36))
            self.text(d, f["image"], 36, 111, "small", GREY, max_px=self.X(470))
            self.text(d, "|", 530, 111, "small", RULE)
            self.text(d, f["view"], 556, 111, "small", GREY)
            self.text(d, "|", 862, 111, "small", RULE)
            self.text(d, f["score"], 888, 111, "small", GREY)
            self._footers[key] = image
        return self._footers[key]

    def boxed(self, canvas, draw, image, origin, size, label, label_right=None):
        x, y = origin
        b = max(2, round(2 * self.s))
        draw.rectangle((x - b, y - b, x + size[0] + b - 1, y + size[1] + b - 1), fill="#6b7280")
        canvas.paste(image, origin)
        draw.text((x, y - self.Y(8)), label, font=self.sans, fill="#b0b7c3", anchor="ls")
        if label_right:
            draw.text((x + size[0], y - self.Y(8)), label_right, font=self.mono["axis"], fill=GREY, anchor="rs")

    def compose(
        self,
        moon,
        original,
        global_map,
        local_map,
        f: dict,
        key: int,
        index: int,
        count: int,
        visited: int,
        ncc: NccGraph,
    ) -> Image.Image:
        canvas = Image.new("RGB", self.size, "black")
        canvas.paste(moon, self.view_origin)
        d = ImageDraw.Draw(canvas)
        d.text((self.X(26), self.Y(18)), "BAD APPLE  /  CLOSEST MOON SURFACE", font=self.sans, fill="#9ca3af")
        d.text((self.X(26), self.Y(44)), f["credit"], font=self.sans_small, fill="#6b7280")
        self.boxed(canvas, d, original, self.inset_origin, self.inset_size, "ORIGINAL")
        self.boxed(canvas, d, global_map, self.global_origin, self.global_size, "MAP", f"VISITED {visited:04d}")
        self.boxed(canvas, d, local_map, self.local_origin, self.local_size, "LOCAL", f["local"][7:])
        footer = self.footer(key, f).copy()
        footer.paste(ncc.image(index), self.ncc_origin)
        fd = ImageDraw.Draw(footer)
        for v in ncc.GRID:
            fd.text(
                (self.ncc_origin[0] - self.X(6), self.ncc_origin[1] + ncc.y(v)),
                f"{v:.1f}",
                font=self.mono["axis"],
                fill="#5b6576",
                anchor="rm",
            )
        self.text(fd, ncc.mean_label(index), 990, 111, "small", GREY)
        digits = len(str(count))
        self.text(fd, f"frame {index + 1:0{digits}d}/{count}", 1244, 111, "small", GREY, anchor="rs", right=True)
        canvas.paste(footer, (0, self.footer_top))
        return canvas


CAMERA_BAND = (100, 196)  # footer rows (at 1920 high) holding only the camera line; the lines above/below stay clear


def camera_band(layout: MoonLayout, camera: str) -> np.ndarray:
    """The camera line drawn exactly as MoonLayout.footer draws it, cut to the band left of the NCC graph."""
    image = Image.new("RGB", (layout.size[0], layout.footer_h), BACKGROUND)
    layout.text(
        ImageDraw.Draw(image), camera, 36, 82, "name", "#ffffff", max_px=layout.R(NCC_BOX[0] - 40) - layout.X(36)
    )
    top, bottom = (round(v * layout.s / 2) for v in CAMERA_BAND)
    return np.asarray(image)[top:bottom, : layout.R(NCC_BOX[0] - 40)]


def refooter(video: Path, frames_path: Path, out: Path, subsolar: dict[str, list[float]], bitrate: str = "60M") -> None:
    """Rewrite only the footer's camera line of a finished video (for when the crops cache is gone).
    The rest of each frame passes through a 16-bit RGB round trip and one more AV1 generation."""
    from .sun import camera_line, local_sun

    log = json.loads(frames_path.read_text(encoding="utf-8"))
    probe = (
        subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate",
                "-of",
                "csv=p=0",
                str(video),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        .stdout.strip()
        .split(",")
    )
    size, rate = (int(probe[0]), int(probe[1])), probe[2]
    layout = MoonLayout(size)
    top = layout.footer_top + round(CAMERA_BAND[0] * layout.s / 2)
    bands: dict[str, np.ndarray] = {}

    def band(text: str) -> np.ndarray:
        if text not in bands:
            bands[text] = camera_band(layout, text).astype(np.uint16) * 257
        return bands[text]

    new = []
    for entry in log:
        camera = entry["caption"]["camera"]
        if entry["raster"] in subsolar:
            camera = camera_line(*local_sun(entry["lat"], entry["lon"], subsolar[entry["raster"]]))
        new.append(camera)
    w, h = size
    frame_bytes = w * h * 6
    source = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(video), "-f", "rawvideo", "-pix_fmt", "rgb48le", "-"],
        stdout=subprocess.PIPE,
        bufsize=frame_bytes,
    )
    sink = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb48le",
            "-s",
            f"{w}x{h}",
            "-r",
            rate,
            "-i",
            "-",
            "-i",
            str(video),
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-c:v",
            "av1_nvenc",
            "-preset",
            "p7",
            "-tune",
            "hq",
            "-multipass",
            "fullres",
            "-rc",
            "vbr",
            "-b:v",
            bitrate,
            "-maxrate",
            "90M",
            "-bufsize",
            "120M",
            "-pix_fmt",
            "p010le",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            str(out),
        ],
        stdin=subprocess.PIPE,
    )
    worst = 0.0
    started = time.time()
    for n, entry in enumerate(log):
        raw = source.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            raise RuntimeError(f"video ended at frame {n} of {len(log)}")
        frame = np.frombuffer(raw, np.uint16).reshape(h, w, 3).copy()
        old = band(entry["caption"]["camera"])
        region = frame[top : top + old.shape[0], : old.shape[1]]
        # The decoded band must be the old caption drawn the same way (only compression noise apart).
        error = float(np.abs(region.astype(np.int32) - old).mean()) / 257
        worst = max(worst, error)
        if error > 1.5:  # the right caption measures ~0.65 after AV1; a caption a few characters off ~3
            raise RuntimeError(f"frame {n}: camera band does not match its old caption (mean error {error:.1f})")
        region[:] = band(new[n])
        sink.stdin.write(frame.tobytes())
        if n % 300 == 0:
            print(
                f"\rframe {n}/{len(log)} ({n / max(time.time() - started, 1e-3):.1f} fps, band error {worst:.2f})",
                end="",
                flush=True,
            )
    print()
    sink.stdin.close()
    if sink.wait() != 0:
        raise RuntimeError("encoder failed")
    source.kill()
    for entry, camera in zip(log, new):
        entry["caption"]["camera"] = camera
    frames_path.with_suffix(".sunfix.json").write_text(json.dumps(log, ensure_ascii=False), encoding="utf-8")
    print(
        f"wrote {out}; worst band error before patching {worst:.2f}/255; "
        f"{sum(a != e['caption']['camera'] for a, e in zip(new, json.loads(frames_path.read_text(encoding='utf-8'))))} "
        f"captions changed"
    )


def encode(
    matches: dict,
    records: dict,
    crops: Path,
    video: Path,
    out_dir: Path,
    encoder: str = "av1_nvenc",
    limit: int | None = None,
    size: tuple[int, int] = (2560, 1920),
    tiles: Path = Path("cache/moon/tiles"),
) -> None:
    from .names import Gazetteer
    from .sun import load_subsolar

    out_dir.mkdir(parents=True, exist_ok=True)
    missing = [r["unique"] for r in matches["results"] if r["best"] is None]
    if missing:
        raise ValueError(
            f"{len(missing)} uniques have no accepted crop (first: {missing[:5]}); re-run crops with --fallback"
        )
    gazetteer = Gazetteer()
    subsolar = load_subsolar({r["best"]["raster"] for r in matches["results"] if r["best"]["source"] == "tc"}, records)
    layout = MoonLayout(size)
    maps = MoonMaps(layout.global_size, layout.local_size, tiles, layout.s)
    rate = matches["rate"]
    frames = matches["frames"][:limit] if limit else matches["frames"]
    results = {r["unique"]: r for r in matches["results"]}
    ncc = NccGraph(
        layout.ncc_size,
        [results[u]["best"]["score32"] if results[u]["kind"] == "match" else None for u in matches["frames"]],
        layout.s,
    )
    iw, ih = layout.inset_size
    source = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(video),
            "-vf",
            f"scale={iw}:{ih}:flags=lanczos",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        stdout=subprocess.PIPE,
    )
    codec = {
        "libx264": ["-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p"],
        "h264_nvenc": ["-c:v", "h264_nvenc", "-preset", "p7", "-rc", "constqp", "-qp", "18", "-pix_fmt", "yuv420p"],
        "av1_nvenc": [
            "-c:v",
            "av1_nvenc",
            "-preset",
            "p7",
            "-tune",
            "hq",
            "-multipass",
            "fullres",
            "-rc",
            "vbr",
            "-b:v",
            "45M",
            "-maxrate",
            "70M",
            "-bufsize",
            "90M",
            "-pix_fmt",
            "p010le",
        ],
    }[encoder]
    out = out_dir / "badapple-moon.mp4"
    sink = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{size[0]}x{size[1]}",
            "-r",
            rate,
            "-i",
            "-",
            "-i",
            str(video),
            "-map",
            "0:v",
            "-map",
            "1:a",
            *codec,
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            "-movflags",
            "+faststart",
            str(out),
        ],
        stdin=subprocess.PIPE,
    )
    fields: dict[int, dict] = {}
    log = []
    started = time.time()
    frame_bytes = iw * ih * 3
    poster_at = min(600, len(frames) - 1)
    previous = None
    shown = None  # (unique, moon, global map, local map) of the surface on screen
    for n, u in enumerate(frames):
        original = source.stdout.read(frame_bytes)
        if len(original) < frame_bytes:
            raise RuntimeError(f"source video ended at frame {n} of {len(frames)}: check --video")
        result = results[u]
        best = result["best"]
        record = records.get(best["raster"])
        if u not in fields:
            fields[u] = describe(result, record, gazetteer, subsolar)
        if shown is None or shown[0] != u:
            if previous is not None:  # the surface that just left the screen becomes history
                maps.add(*previous)
            corners = footprint(best, record)
            previous = (corners, best["lon"], best["lat"])
            name = crops / f"{crop_key(best)}.jpg"
            moon = Image.open(name).convert("RGB")
            if moon.size != layout.view_size:
                moon = moon.resize(layout.view_size, Image.Resampling.LANCZOS)
            shown = (
                u,
                moon,
                maps.global_image(corners, best["lon"]),
                maps.local_image(
                    Image.open(name.with_suffix(".map.jpg")), corners, best["lon"], best["lat"], local_span_km(best)
                ),
            )
        frame = layout.compose(
            shown[1],
            Image.frombytes("RGB", (iw, ih), original),
            shown[2],
            shown[3],
            fields[u],
            u,
            n,
            len(frames),
            maps.visited,
            ncc,
        )
        sink.stdin.write(frame.tobytes())
        if n == poster_at:
            frame.save(out.with_suffix(".poster.png"))
        f = fields[u]
        log.append(
            {
                "frame": n,
                "unique": u,
                "kind": result["kind"],
                "raster": best["raster"],
                "lat": round(best["lat"], 5),
                "lon": round(best["lon"], 5),
                "width_km": round(best["width_km"], 3),
                "theta_deg": round(best["theta_deg"], 2),
                "mirror": best["mirror"],
                "ncc": None if result["kind"] != "match" else round(best["score32"], 4),
                "caption": {
                    k: f[k] for k in ("coords", "feature", "feature_detail", "camera", "image", "view", "score")
                },
            }
        )
        if n % 300 == 0:
            print(f"\rframe {n}/{len(frames)} ({n / max(time.time() - started, 1e-3):.1f} fps)", end="", flush=True)
    print()
    sink.stdin.close()
    if sink.wait() != 0:
        raise RuntimeError("encoder failed")
    source.kill()
    (out_dir / "frames.json").write_text(json.dumps(log, ensure_ascii=False), encoding="utf-8")
    print("wrote", out)
