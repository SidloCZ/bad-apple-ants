"""Country / continent assignment and statistics for every displayed view.

Method (see also docs/satellite.md, land & water edition):

* Boundaries: Natural Earth 1:10m Admin-0 *map units* (public domain, de facto
  "ground truth" point of view). Countries are counted by ``ADMIN`` so that
  overseas parts (French Guiana, Svalbard, Azores...) keep their own continent
  while still counting for their country.
* A view is the actual displayed photograph quadrilateral (``footprint``).
  Land views: the geodesic area of the quad's intersection with each map unit;
  the largest share is the primary country, all shares are kept.
* Views with < 2 % Natural Earth land: the nearest map unit within 200 nautical
  miles (370.4 km, the EEZ breadth), measured in an azimuthal-equidistant
  projection centred on the view; otherwise "Open ocean" with the Natural Earth
  marine-area name containing the view centre.
* Continent: the map unit's ``CONTINENT``; Natural Earth's "Seven seas (open
  ocean)" remote islands use their UN M49 region (REGION_UN). "Open ocean" is
  only for views farther than 200 nmi from every country.
  Russia is one map unit (CONTINENT=Europe) and is split at the Urals (60 E).
* Places: views whose quads overlap with IoU > 0.5 are joined (union-find), so
  near-identical consecutive views count as one place.
"""

from __future__ import annotations
import argparse, json, math, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import shapely
from shapely.geometry import Polygon, shape, box
from shapely.strtree import STRtree
from pyproj import Geod, CRS, Transformer

ROOT = Path(__file__).resolve().parents[1]
NE = ROOT / "assets/coastline/natural-earth"
GEOD = Geod(ellps="WGS84")
EEZ_KM = 370.4
LAND_SHARE_MIN = 0.02
OPEN_OCEAN = "Open ocean"
CONTINENT_ORDER = ["Asia", "North America", "Europe", "Africa", "South America", "Oceania", "Antarctica", OPEN_OCEAN]

_UNITS = _TREE = _MARINE = _MARINE_TREE = _DISPUTED = _DISPUTED_TREE = None


def _load():
    global _UNITS, _TREE, _MARINE, _MARINE_TREE, _DISPUTED, _DISPUTED_TREE
    if _UNITS is not None:
        return

    def read(name):
        data = json.loads((NE / name).read_text(encoding="utf-8"))
        return [(shapely.make_valid(shape(f["geometry"])), f["properties"]) for f in data["features"] if f["geometry"]]

    _UNITS = read("ne_10m_admin_0_map_units.geojson")
    _TREE = STRtree([g for g, _ in _UNITS])
    _MARINE = read("ne_10m_geography_marine_polys.geojson")
    _MARINE_TREE = STRtree([g for g, _ in _MARINE])
    _DISPUTED = read("ne_10m_admin_0_disputed_areas.geojson")
    _DISPUTED_TREE = STRtree([g for g, _ in _DISPUTED])


URALS_LONGITUDE = 60.0


def continent_of(props, lon=None):
    """Natural Earth CONTINENT; Russia (one map unit, CONTINENT=Europe) is split at the
    Urals, approximated by 60 E, so Siberian and Far-East views count as Asia."""
    c = props["CONTINENT"]
    if props.get("ADMIN") == "Russia" and lon is not None and not (-170 < lon < URALS_LONGITUDE):
        return "Asia"
    if c.startswith("Seven seas"):
        # Remote islands (Azores, Maldives, Mauritius...): UN M49 region, Americas split by subregion.
        region, sub = props.get("REGION_UN"), props.get("SUBREGION") or ""
        if region == "Americas":
            return "South America" if sub in ("South America", "Seven seas (open ocean)") else "North America"
        return region if region in ("Africa", "Asia", "Europe", "Oceania", "Antarctica") else OPEN_OCEAN
    return c


def unwrapped(corners):
    p = np.asarray(corners, float).copy()
    p[:, 0] = np.rad2deg(np.unwrap(np.deg2rad(p[:, 0])))
    return p


def quad_pieces(corners):
    """Quad in lon/lat, split into copies that fall inside [-180, 180]."""
    p = unwrapped(corners)
    q = Polygon(p)
    pieces = []
    for shift in (-360, 0, 360):
        part = shapely.affinity.translate(q, shift, 0).intersection(box(-180, -90, 180, 90))
        if not part.is_empty:
            pieces.append(part)
    return q, pieces


def geodesic_area(geometry):
    if geometry.is_empty:
        return 0.0
    return abs(GEOD.geometry_area_perimeter(shapely.segmentize(geometry, 0.05))[0])


def centre(corners):
    p = np.asarray(corners, float)
    r = np.deg2rad(p[:, 0])
    return float(np.rad2deg(np.arctan2(np.sin(r).mean(), np.cos(r).mean()))), float(p[:, 1].mean())


def assign(corners):
    _load()
    quad, pieces = quad_pieces(corners)
    total = sum(geodesic_area(p) for p in pieces)
    shares = defaultdict(float)
    units = {}
    lon, lat = centre(corners)
    for piece in pieces:
        for k in _TREE.query(piece, predicate="intersects"):
            geom, props = _UNITS[k]
            area = geodesic_area(geom.intersection(piece))
            if area > 0:
                key = (props["ADMIN"], props["GEOUNIT"], continent_of(props, lon))
                shares[key] += area
                units[key] = props
    land = sum(shares.values())
    result = dict(
        centre_lonlat=[lon, lat],
        view_area_km2=total / 1e6,
        natural_earth_land_share=min(1.0, land / total) if total else 0.0,
    )
    if total and land / total >= LAND_SHARE_MIN:
        ranked = sorted(shares.items(), key=lambda kv: -kv[1])
        (admin, geounit, continent), _ = ranked[0]
        by_country = defaultdict(float)
        for (a, _, _), area in ranked:
            by_country[a] += area / land
        result.update(
            method="land-overlap",
            country=admin,
            territory=geounit if geounit != admin else None,
            continent=continent,
            iso_a3=units[ranked[0][0]].get("ADM0_A3"),
            water=False,
            countries=[
                dict(country=a, share=round(s, 4)) for a, s in sorted(by_country.items(), key=lambda kv: -kv[1])
            ],
        )
        disputed = []
        for piece in pieces:
            for k in _DISPUTED_TREE.query(piece, predicate="intersects"):
                geom, props = _DISPUTED[k]
                area = geodesic_area(geom.intersection(piece))
                if area > 0.5 * land:
                    disputed.append(dict(name=props.get("BRK_NAME") or props.get("NAME"), note=props.get("NOTE_BRK")))
        if disputed:
            result["disputed_area"] = disputed[0]
        return result
    # Mostly water: nearest map unit within the 200 nmi EEZ breadth.
    aeqd = CRS.from_proj4(f"+proj=aeqd +lat_0={lat:.9f} +lon_0={lon:.9f} +datum=WGS84 +units=m +no_defs")
    forward = Transformer.from_crs("EPSG:4326", aeqd, always_xy=True)
    reach = EEZ_KM / 111.0 + 1.0
    lon_reach = min(180, reach / max(0.05, math.cos(math.radians(lat))))
    window = box(lon - lon_reach, max(-90, lat - reach), lon + lon_reach, min(90, lat + reach))
    windows = [shapely.affinity.translate(window, s, 0).intersection(box(-180, -90, 180, 90)) for s in (-360, 0, 360)]
    best = None
    projected_quad = shapely.transform(quad, lambda xy: np.column_stack(forward.transform(xy[:, 0], xy[:, 1])))
    for win in windows:
        if win.is_empty:
            continue
        for k in _TREE.query(win, predicate="intersects"):
            geom, props = _UNITS[k]
            clipped = geom.intersection(win)
            if clipped.is_empty:
                continue
            projected = shapely.transform(clipped, lambda xy: np.column_stack(forward.transform(xy[:, 0], xy[:, 1])))
            distance = projected.distance(projected_quad) / 1000
            if best is None or distance < best[0]:
                best = (distance, props)
    if best and best[0] <= EEZ_KM:
        props = best[1]
        result.update(
            method="nearest-within-200nmi",
            country=props["ADMIN"],
            territory=props["GEOUNIT"] if props["GEOUNIT"] != props["ADMIN"] else None,
            continent=continent_of(props, lon),
            iso_a3=props.get("ADM0_A3"),
            water=True,
            distance_to_land_km=round(best[0], 2),
            countries=[dict(country=props["ADMIN"], share=1.0)],
        )
        return result
    sea = None
    point = shapely.Point(lon, lat)
    for k in _MARINE_TREE.query(point, predicate="intersects"):
        sea = _MARINE[k][1].get("name_en") or _MARINE[k][1].get("name")
        break
    result.update(
        method="open-ocean",
        country=OPEN_OCEAN,
        territory=None,
        continent=OPEN_OCEAN,
        iso_a3=None,
        water=True,
        sea=sea,
        distance_to_land_km=round(best[0], 2) if best else None,
        countries=[],
    )
    return result


def view_key(corners):
    return tuple(sorted(tuple(round(v, 9) for v in c) for c in np.asarray(corners).tolist()))


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def places(views, threshold=0.5):
    """Cluster views whose quads overlap with IoU > threshold."""
    polys = []
    for v in views:
        p = unwrapped(v["corners_lonlat"])
        lat = p[:, 1].mean()
        scale = math.cos(math.radians(lat))
        polys.append(Polygon(np.column_stack([p[:, 0] * scale, p[:, 1]])))
    # Seam: duplicate shifted copies are not needed for IoU of near-identical quads,
    # but wrap longitudes so quads straddling +-180 compare in one frame.
    tree = STRtree(polys)
    uf = UnionFind(len(polys))
    for i, a in enumerate(polys):
        for j in tree.query(a, predicate="intersects"):
            if j <= i:
                continue
            b = polys[j]
            inter = a.intersection(b).area
            if inter and inter / (a.area + b.area - inter) > threshold:
                uf.union(i, j)
    return [uf.find(i) for i in range(len(polys))]


def _assign_job(item):
    index, corners = item
    return index, assign(corners)


def build(records, matched_frames):
    """records: displayed records in video order; matched_frames: count of matched (non-credit) frames."""
    from .footprint_map import footprint

    start = time.perf_counter()
    order = []
    views = {}
    frames_of = defaultdict(list)
    for i in range(matched_frames):
        corners = footprint(records[i])
        key = view_key(corners)
        if key not in views:
            views[key] = dict(view=len(order), corners_lonlat=np.asarray(corners).tolist())
            order.append(key)
        frames_of[key].append(i)
    jobs = [(views[k]["view"], views[k]["corners_lonlat"]) for k in order]
    # Serial on purpose (~80 s): process pools failed intermittently on Windows (WinError 6).
    for n, (index, result) in enumerate(map(_assign_job, jobs), 1):
        views[order[index]].update(result)
        if n % 500 == 0 or n == len(jobs):
            print(f"geostats: {n}/{len(jobs)} views; {time.perf_counter() - start:.0f}s", flush=True)
    view_list = [views[k] for k in order]
    clusters = places(view_list)
    for v, key, c in zip(view_list, order, clusters):
        f = frames_of[key]
        r = records[f[0]]
        v.update(
            place=c,
            frames=f,
            frame_count=len(f),
            width_km=r["width_km"],
            rotation_clockwise=r["rotation_clockwise"],
            mirror_horizontal=bool(r.get("mirror_horizontal")),
            nearest_settlement=r["place"],
            miou_mean=float(np.mean([records[i].get("native_miou", records[i].get("iou", 0)) for i in f])),
            photo_dates=sorted({s["datetime"][:10] for s in r["satellite"]["scenes"]}),
            imagery="LANDSAT 8/9" if r.get("landsat_replacement") else "SENTINEL-2",
        )
    remap = {c: k for k, c in enumerate(dict.fromkeys(clusters))}
    for v in view_list:
        v["place"] = remap[v["place"]]
    per_frame = [None] * matched_frames
    for v in view_list:
        for i in v["frames"]:
            per_frame[i] = v["view"]
    stats = summarize(view_list, matched_frames)
    stats.update(elapsed_seconds=time.perf_counter() - start)
    return dict(views=view_list, frame_view=per_frame, stats=stats)


EARTH_SURFACE_KM2 = 510_072_000.0


def union_area_km2(views):
    """Geodesic area of the union of the views' quadrilaterals: every point of the
    Earth counted once, however many views show it (antimeridian-safe)."""
    from shapely.ops import unary_union

    pieces = [part for v in views for part in quad_pieces(v["corners_lonlat"])[1]]
    return geodesic_area(unary_union(pieces)) / 1e6 if pieces else 0.0


def summarize(views, matched_frames):
    frames_c = Counter()
    views_c = Counter()
    places_c = defaultdict(set)
    area_c = Counter()
    frames_k = Counter()
    views_k = Counter()
    places_k = defaultdict(set)
    continent_of_country = {}
    for v in views:
        frames_c[v["continent"]] += v["frame_count"]
        views_c[v["continent"]] += 1
        places_c[v["continent"]].add(v["place"])
        area_c[v["continent"]] += v["view_area_km2"]  # sum over views (overlaps counted repeatedly)
        frames_k[v["country"]] += v["frame_count"]
        views_k[v["country"]] += 1
        places_k[v["country"]].add(v["place"])
        continent_of_country.setdefault(v["country"], Counter())[v["continent"]] += v["frame_count"]
    continents = [
        dict(
            continent=c,
            frames=frames_c[c],
            views=views_c[c],
            places=len(places_c[c]),
            area_union_km2=union_area_km2([v for v in views if v["continent"] == c]),
            area_sum_km2=area_c[c],
        )
        for c in sorted(frames_c, key=lambda c: -frames_c[c])
    ]
    countries = [
        dict(
            country=k,
            frames=frames_k[k],
            views=views_k[k],
            places=len(places_k[k]),
            continent=continent_of_country[k].most_common(1)[0][0],
        )
        for k in sorted(frames_k, key=lambda k: (-frames_k[k], k))
    ]
    real = [c for c in countries if c["country"] != OPEN_OCEAN]
    widths = np.array([v["width_km"] for v in views])
    return dict(
        matched_frames=matched_frames,
        views=len(views),
        places=len({v["place"] for v in views}),
        countries=len(real),
        continents=len([c for c in continents if c["continent"] != OPEN_OCEAN]),
        open_ocean_frames=frames_c.get(OPEN_OCEAN, 0),
        water_assigned_views=sum(v["water"] for v in views),
        mirrored_views=sum(v["mirror_horizontal"] for v in views),
        width_km_min=float(widths.min()),
        width_km_median=float(np.median(widths)),
        width_km_max=float(widths.max()),
        area_union_km2=(union := union_area_km2(views)),
        area_union_share_of_earth=union / EARTH_SURFACE_KM2,
        view_area_sum_km2=float(
            sum(v["view_area_km2"] for v in views)
        ),  # overlaps counted repeatedly; not an area of the Earth
        by_continent=continents,
        by_country=countries,
        method_counts=dict(Counter(v["method"] for v in views)),
    )


if __name__ == "__main__":
    from .pipeline import read_records
    from .flash_reduction import SOURCE, CREDIT_START

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = read_records(SOURCE / "frames.jsonl")
    data = build(records, CREDIT_START)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    s = data["stats"]
    print(json.dumps({k: v for k, v in s.items() if k not in ("by_country",)}, ensure_ascii=False, indent=1))
    print("top countries:", [(c["country"], c["frames"], c["places"]) for c in s["by_country"][:15]])
