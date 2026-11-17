"""North-up OpenStreetMap context maps for every displayed view.

Base maps are rendered once per unique view by MapLibre GL JS in headless
Chromium (tools/landwater-osm), from a Protomaps planet build read by HTTP range
requests through a local caching server. Footprint overlays (gold = current
photograph footprint, cyan = earlier nearby views) are drawn here per frame with
the same Web Mercator maths MapLibre uses, so one base map serves every frame
of a view and the history is always exact.
"""

from __future__ import annotations
import json, math, os, subprocess, tempfile, time, urllib.request
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .footprint_map import footprint
from .local_map import geometry, dense_boundary, GOLD
from .geostats import view_key

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/landwater-osm"
# Protomaps publishes dated planet builds and removes old ones after a while; the
# video used the 2026-11-09 build. Point BADAPPLE_PMTILES at a current build (or a
# local extract) to re-render; the tile cache directory is keyed by the build name.
PMTILES = os.environ.get("BADAPPLE_PMTILES", "https://build.protomaps.com/20261109.pmtiles")
CACHE = Path(os.environ.get("BADAPPLE_CACHE", "cache")) / "osm" / f"protomaps-{Path(PMTILES).stem}"


def chrome_binary():
    """Headless Chromium: BADAPPLE_CHROME, else the newest Playwright Chromium."""
    if os.environ.get("BADAPPLE_CHROME"):
        return Path(os.environ["BADAPPLE_CHROME"])
    root = Path(
        os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
        or Path(os.environ.get("LOCALAPPDATA", Path.home())) / "ms-playwright"
    )
    found = sorted(root.glob("chromium-*/chrome-win*/chrome.exe"), key=lambda p: int(p.parts[-3].split("-")[1]))
    if not found:
        raise RuntimeError("Chromium not found: run `npx playwright install chromium` or set BADAPPLE_CHROME")
    return found[-1]


EARTH = 40075016.68557849
NAVY = (8, 15, 23)
CYAN = (67, 174, 188)


def mercator(lon, lat, zoom):
    world = 512 * 2**zoom
    lat = np.clip(lat, -85.05112878, 85.05112878)
    x = (np.asarray(lon) + 180) / 360 * world
    y = (1 - np.log(np.tan(np.pi / 4 + np.deg2rad(lat) / 2)) / np.pi) / 2 * world
    return x, y


LABEL_SCALE = 1.6  # labels 1.6x larger relative to the panel (render at panel/1.6 CSS px)


def css_size(size):
    return round(size[0] / LABEL_SCALE), round(size[1] / LABEL_SCALE)


def view_spec(points, size):
    """Centre/zoom for a north-up map spanning the LOCAL-map width at the view centre."""
    _, lon, lat, span = geometry(points, size)
    lat_c = float(np.clip(lat, -84, 84))
    zoom = math.log2(EARTH * math.cos(math.radians(lat_c)) * css_size(size)[0] / (512 * span * 1000))
    return dict(center=[lon, lat_c], zoom=zoom, span_km=span)


def build_views(shown, size):
    views = {}
    order = []
    for i, r in enumerate(shown):
        points = footprint(r)
        key = view_key(points)
        if key not in views:
            spec = view_spec(points, size)
            views[key] = dict(
                name=f"view-{len(order):05d}.png", first_frame=i, corners=np.asarray(points).tolist(), **spec
            )
            order.append(key)
    return [views[k] for k in order], [views[view_key(footprint(r))]["name"] for r in shown]


def render(views, out, size, shards=6, port=8130, cache=CACHE, pmtiles=PMTILES, timeout=4 * 3600):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    todo = [v for v in views if not (out / v["name"]).is_file()]
    if not todo:
        return dict(rendered=0, cached=len(views))
    work = out.parent / "osm-views.json"
    work.write_text(
        json.dumps(dict(views=[dict(name=v["name"], center=v["center"], zoom=v["zoom"]) for v in todo])),
        encoding="utf-8",
    )
    logs = out.parent / "osm-server.log"
    server = subprocess.Popen(
        [
            "node",
            str(TOOL / "server.mjs"),
            "--port",
            str(port),
            "--cache",
            str(cache),
            "--pmtiles",
            pmtiles,
            "--views",
            str(work),
            "--out",
            str(out),
        ],
        stdout=open(logs, "w"),
        stderr=subprocess.STDOUT,
    )
    browsers = []
    start = time.perf_counter()
    try:
        for _ in range(100):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/status", timeout=1)
                break
            except Exception:
                time.sleep(0.1)
        shards = max(1, min(shards, len(todo)))
        for k in range(shards):
            profile = tempfile.mkdtemp(prefix=f"osm-chrome-{k}-")
            w, h = css_size(size)
            ratio = 2 * size[0] / w  # canvas = 2x the panel, downsampled once
            url = f"http://127.0.0.1:{port}/render.html?shard={k}&of={shards}&w={w}&h={h}&ratio={ratio:.6f}"
            browsers.append(
                subprocess.Popen(
                    [
                        str(chrome_binary()),
                        "--headless=new",
                        "--use-angle=d3d11",
                        "--enable-gpu-rasterization",
                        "--ignore-gpu-blocklist",
                        "--no-first-run",
                        "--no-default-browser-check",
                        "--disable-background-timer-throttling",
                        "--disable-renderer-backgrounding",
                        "--disable-backgrounding-occluded-windows",
                        f"--user-data-dir={profile}",
                        f"--window-size={size[0] + 40},{size[1] + 40}",
                        url,
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            )
        while True:
            status = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/status", timeout=10).read())
            if len(status["shardsDone"]) == shards:
                break
            if status["errors"]:
                raise RuntimeError(f"OSM renderer reported errors; see {logs}")
            if time.perf_counter() - start > timeout:
                raise TimeoutError("OSM rendering timed out")
            print(
                f"osm maps: {status['results']}/{len(todo)}; remote ranges {status['remoteRanges']} ({status['remoteBytes'] / 2**20:.0f} MiB); "
                f"tile cache hits {status['tileHits']}; {time.perf_counter() - start:.0f}s",
                flush=True,
            )
            time.sleep(5)
        missing = [v["name"] for v in todo if not (out / v["name"]).is_file()]
        if missing:
            raise RuntimeError(f"{len(missing)} maps missing after rendering")
        return dict(
            rendered=len(todo), cached=len(views) - len(todo), seconds=time.perf_counter() - start, server=status
        )
    finally:
        for b in browsers:
            if b.poll() is None:
                b.kill()
        if server.poll() is None:
            server.kill()


class OsmPanels:
    """Per-frame panel: base map + current (gold) and earlier nearby (cyan) footprints."""

    def __init__(self, views, names, maps_dir, size):
        self.views = {v["name"]: v for v in views}
        self.names = names
        self.dir = Path(maps_dir)
        self.size = size
        self.scale = size[0] / css_size(size)[0]
        self.cache = {}
        import threading

        self.lock = threading.Lock()
        self.dense = {v["name"]: dense_boundary(np.asarray(v["corners"])) for v in views}
        c = np.array([v["center"] for v in views])
        lon, lat = np.deg2rad(c).T
        self.units = np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
        self.order = [v["name"] for v in views]
        self.index = {n: k for k, n in enumerate(self.order)}
        self.radius = np.array([max(1e-6, v["span_km"]) for v in views])

    def base(self, name):
        with self.lock:
            hit = self.cache.get(name)
        if hit is not None:
            return hit
        with Image.open(self.dir / name) as im:
            image = im.convert("RGB").resize(self.size, Image.Resampling.LANCZOS)
        with self.lock:
            if len(self.cache) > 64:
                self.cache.clear()
            self.cache[name] = image
        return image

    def project(self, view, polygon):
        zoom = view["zoom"]
        cx, cy = mercator(*view["center"], zoom)
        p = np.asarray(polygon, float).copy()
        p[:, 0] = view["center"][0] + ((p[:, 0] - view["center"][0] + 180) % 360 - 180)
        x, y = mercator(p[:, 0], p[:, 1], zoom)
        return np.column_stack([(x - cx) * self.scale + self.size[0] / 2, (y - cy) * self.scale + self.size[1] / 2])

    def render(self, frame, seen_names):
        name = self.names[frame]
        view = self.views[name]
        image = self.base(name).copy()
        draw = ImageDraw.Draw(image)
        k = self.index[name]
        dot = (
            self.units[np.array([self.index[n] for n in seen_names], int)] @ self.units[k]
            if seen_names
            else np.array([])
        )
        reach = view["span_km"] * 1.2 + 50
        for n, d in zip(seen_names, dot):
            if n == name:
                continue
            if 6371.01 * math.acos(max(-1, min(1, d))) > reach:
                continue
            xy = self.project(view, self.dense[n])
            if np.isfinite(xy).all():
                draw.line([*map(tuple, xy), tuple(xy[0])], fill=(44, 118, 130), width=2)
        xy = [*map(tuple, self.project(view, self.dense[name]))]
        xy.append(xy[0])
        draw.line(xy, fill=NAVY, width=7, joint="curve")
        draw.line(xy, fill=GOLD, width=3, joint="curve")
        return np.asarray(image)
