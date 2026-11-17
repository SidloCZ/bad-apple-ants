//! Lunar rasters for the search: Kaguya TC image overviews (local quadratic geometry) and the
//! LROC WAC global mosaic (equirectangular), each with a nodata-aware 2x box pyramid.
use anyhow::{Context, Result, ensure};
use rayon::prelude::*;
use serde::Deserialize;
use std::{f64::consts::PI, fs, path::Path};

pub const R_KM: f64 = 1737.4;
pub const GW: usize = 32;
pub const GH: usize = 24;
pub const DIM: usize = GW * GH;
/// A level serves a cell size s when its pixel is at most this much larger than s.
pub const LEVEL_SLACK: f64 = 1.2;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Level {
    pub offset: u64,
    pub width: i32,
    pub height: i32,
    pub kind: i32,
    pub pad: i32,
    pub q: [f32; 12],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Pose {
    pub level: u32,
    pub a: f32,
    pub b: f32,
    pub w: f32,
    pub theta: f32,
    pub flags: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Entry {
    pub score: f32,
    pub pose: Pose,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    Tc,
    Wac,
}

#[derive(Deserialize, Clone)]
pub struct TcRecord {
    pub id: String,
    pub width: usize,
    pub height: usize,
    pub full_width: usize,
    pub full_height: usize,
    pub lon0: f64,
    pub lat0: f64,
    pub forward: [[f64; 6]; 2],
    pub inverse: [[f64; 6]; 2],
    pub offset: u64,
    // Kept from the index for completeness; the search does not use them.
    #[allow(dead_code)]
    pub gsd: f64,
    #[allow(dead_code)]
    pub sun_elevation: Option<f64>,
    #[allow(dead_code)]
    pub sun_azimuth: Option<f64>,
}

#[derive(Deserialize)]
struct GlobalMeta {
    width: usize,
    height: usize,
}

pub struct Raster {
    pub name: String,
    pub source: Source,
    pub levels: Vec<u32>,
    pub res_km: Vec<f64>,
    pub tc: Option<TcRecord>,
    /// Tangent-plane bounds (TC): xmin, xmax, ymin, ymax in km.
    pub bounds: [f64; 4],
}

/// A contiguous run of rasters small enough for one GPU context.
#[derive(Clone, Debug)]
pub struct Shard {
    pub rasters: std::ops::Range<usize>,
    pub levels: std::ops::Range<usize>,
    pub bytes: std::ops::Range<usize>,
}

pub struct World {
    /// Pixels of the currently filled shard only (see fill); level offsets are global.
    pub data: Vec<u8>,
    pub data_base: usize,
    pub total_bytes: usize,
    maps: Vec<memmap2::Mmap>,
    /// Per TC raster: (directory index, byte offset of its base level in that tc_base.u8).
    sources: Vec<Option<(usize, u64)>>,
    wac_pyramid: Vec<u8>,
    pub levels: Vec<Level>,
    pub rasters: Vec<Raster>,
    /// level index -> raster index
    pub owner: Vec<u32>,
}

pub fn quad(q: &[f64; 6], x: f64, y: f64) -> f64 {
    q[0] + q[1] * x + q[2] * y + q[3] * x * x + q[4] * x * y + q[5] * y * y
}

fn downsample(src: &[u8], w: usize, h: usize) -> (Vec<u8>, usize, usize) {
    let (nw, nh) = (w / 2, h / 2);
    let mut out = vec![0u8; nw * nh];
    out.par_chunks_mut(nw.max(1))
        .enumerate()
        .for_each(|(y, row)| {
            for (x, o) in row.iter_mut().enumerate() {
                let i = 2 * y * w + 2 * x;
                let (a, b, c, d) = (src[i], src[i + 1], src[i + w], src[i + w + 1]);
                if a != 0 && b != 0 && c != 0 && d != 0 {
                    *o = ((a as u32 + b as u32 + c as u32 + d as u32 + 2) / 4).max(1) as u8;
                }
            }
        });
    (out, nw, nh)
}

fn downsample_into(src: &[u8], w: usize, h: usize, out: &mut [u8]) {
    let (nw, nh) = (w / 2, h / 2);
    for y in 0..nh {
        for x in 0..nw {
            let i = 2 * y * w + 2 * x;
            let (a, b, c, d) = (src[i], src[i + 1], src[i + w], src[i + w + 1]);
            out[y * nw + x] = if a != 0 && b != 0 && c != 0 && d != 0 {
                ((a as u32 + b as u32 + c as u32 + d as u32 + 2) / 4).max(1) as u8
            } else {
                0
            };
        }
    }
}

/// Pyramid shape of one TC overview: (width, height, km per pixel) per level.
fn tc_pyramid(r: &TcRecord, max_cell: f64) -> Vec<(usize, usize, f64)> {
    let jx = r.forward[0][1] * r.forward[1][2] - r.forward[0][2] * r.forward[1][1];
    let full_res = 1.0 / jx.abs().sqrt();
    let (mut w, mut h) = (r.width, r.height);
    let mut res =
        full_res * ((r.full_width as f64 / w as f64) * (r.full_height as f64 / h as f64)).sqrt();
    let mut out = Vec::new();
    loop {
        out.push((w, h, res));
        if res * 2.0 > max_cell * 1.5 || w < 8 || h < 8 {
            return out;
        }
        w /= 2;
        h /= 2;
        res *= 2.0;
    }
}

impl World {
    /// Loads every TC series directory and the WAC mosaic into one host buffer with global level ids.
    pub fn load(
        tiles: &Path,
        extra_tc: &[std::path::PathBuf],
        with_tc: bool,
        with_wac: bool,
    ) -> Result<Self> {
        let tc_dirs: Vec<std::path::PathBuf> = std::iter::once(tiles.to_path_buf())
            .chain(extra_tc.iter().cloned())
            .collect();
        let mut world = World {
            data: Vec::new(),
            data_base: 0,
            total_bytes: 0,
            maps: Vec::new(),
            sources: Vec::new(),
            wac_pyramid: Vec::new(),
            levels: Vec::new(),
            rasters: Vec::new(),
            owner: Vec::new(),
        };
        if with_tc {
            // Every TC series directory (mono first, then the extra series) holds tc_index.jsonl,
            // tc_base.u8 and optionally tc_exclude.json; an id seen in an earlier directory wins.
            let mut seen = std::collections::HashSet::new();
            let mut records: Vec<(usize, TcRecord)> = Vec::new();
            for (d, dir) in tc_dirs.iter().enumerate() {
                let index = fs::read_to_string(dir.join("tc_index.jsonl"))
                    .with_context(|| format!("{}", dir.display()))?;
                let excluded: std::collections::HashSet<String> =
                    match fs::read_to_string(dir.join("tc_exclude.json")) {
                        Ok(text) => serde_json::from_str(&text)?,
                        Err(_) => Default::default(),
                    };
                for line in index.lines().filter(|l| !l.trim().is_empty()) {
                    let r: TcRecord = serde_json::from_str(line)?;
                    if !excluded.contains(&r.id) && seen.insert(r.id.clone()) {
                        records.push((d, r));
                    }
                }
                let file = fs::File::open(dir.join("tc_base.u8"))?;
                world.maps.push(unsafe { memmap2::Mmap::map(&file)? });
            }
            let max_cell = super::ladder::TC_MAX_KM / GW as f64 * LEVEL_SLACK;
            let shapes: Vec<Vec<(usize, usize, f64)>> = records
                .par_iter()
                .map(|(_, r)| tc_pyramid(r, max_cell))
                .collect();
            let sizes: Vec<usize> = shapes
                .iter()
                .map(|s| s.iter().map(|&(w, h, _)| w * h).sum())
                .collect();
            let mut starts = Vec::with_capacity(sizes.len());
            let mut total = 0usize;
            for s in &sizes {
                starts.push(total);
                total += s;
            }
            world.total_bytes = total;
            for (((d, r), shape), start) in records.into_iter().zip(shapes).zip(starts) {
                let ri = world.rasters.len() as u32;
                world.sources.push(Some((d, r.offset)));
                let mut levels = Vec::new();
                let mut offset = start;
                for (k, &(w, h, _)) in shape.iter().enumerate() {
                    let sx = (r.width as f64 / r.full_width as f64) / f64::powi(2.0, k as i32);
                    let sy = (r.height as f64 / r.full_height as f64) / f64::powi(2.0, k as i32);
                    let mut q = [0f32; 12];
                    for t in 0..6 {
                        q[t] = (r.forward[0][t] * sx) as f32;
                        q[6 + t] = (r.forward[1][t] * sy) as f32;
                    }
                    levels.push(world.levels.len() as u32);
                    world.levels.push(Level {
                        offset: offset as u64,
                        width: w as i32,
                        height: h as i32,
                        kind: 1,
                        pad: 0,
                        q,
                    });
                    world.owner.push(ri);
                    offset += w * h;
                }
                let corners = [
                    (0.0, 0.0),
                    (r.full_width as f64, 0.0),
                    (0.0, r.full_height as f64),
                    (r.full_width as f64, r.full_height as f64),
                ];
                let mut bounds = [f64::MAX, f64::MIN, f64::MAX, f64::MIN];
                for (c, rr) in corners {
                    let (x, y) = (quad(&r.inverse[0], c, rr), quad(&r.inverse[1], c, rr));
                    bounds[0] = bounds[0].min(x);
                    bounds[1] = bounds[1].max(x);
                    bounds[2] = bounds[2].min(y);
                    bounds[3] = bounds[3].max(y);
                }
                world.rasters.push(Raster {
                    name: r.id.clone(),
                    source: Source::Tc,
                    levels,
                    res_km: shape.iter().map(|s| s.2).collect(),
                    tc: Some(r),
                    bounds,
                });
            }
        }
        if with_wac {
            let meta: GlobalMeta =
                serde_json::from_str(&fs::read_to_string(tiles.join("wac.json"))?)?;
            let mut img = fs::read(tiles.join("wac.u8"))?;
            ensure!(img.len() == meta.width * meta.height, "wac.u8 size");
            let (mut w, mut h) = (meta.width, meta.height);
            let mut res = 2.0 * PI * R_KM / w as f64;
            let max_cell = super::ladder::WAC_MAX_KM / GW as f64 * LEVEL_SLACK;
            let ri = world.rasters.len() as u32;
            let (mut levels, mut ress) = (Vec::new(), Vec::new());
            loop {
                levels.push(world.levels.len() as u32);
                ress.push(res);
                world.levels.push(Level {
                    offset: (world.total_bytes + world.wac_pyramid.len()) as u64,
                    width: w as i32,
                    height: h as i32,
                    kind: 0,
                    pad: 0,
                    q: [0.0; 12],
                });
                world.owner.push(ri);
                world.wac_pyramid.extend_from_slice(&img);
                if res * 2.0 > max_cell * 1.5 {
                    break;
                }
                let (n, nw, nh) = downsample(&img, w, h);
                img = n;
                w = nw;
                h = nh;
                res *= 2.0;
            }
            world.sources.push(None);
            world.total_bytes += world.wac_pyramid.len();
            world.rasters.push(Raster {
                name: "LROC_WAC_global_100m".into(),
                source: Source::Wac,
                levels,
                res_km: ress,
                tc: None,
                bounds: [0.0; 4],
            });
        }
        Ok(world)
    }

    /// Split the rasters (stored contiguously in raster order) into runs of at most `budget` bytes.
    pub fn shards(&self, budget: usize) -> Vec<Shard> {
        let span = |r: &Raster| {
            let first = &self.levels[r.levels[0] as usize];
            let last = &self.levels[*r.levels.last().unwrap() as usize];
            (
                first.offset as usize,
                last.offset as usize + (last.width * last.height) as usize,
            )
        };
        let mut out: Vec<Shard> = Vec::new();
        for (i, r) in self.rasters.iter().enumerate() {
            let (b0, b1) = span(r);
            let (l0, l1) = (r.levels[0] as usize, *r.levels.last().unwrap() as usize + 1);
            match out.last_mut() {
                Some(s) if b1 - s.bytes.start <= budget => {
                    s.rasters.end = i + 1;
                    s.levels.end = l1;
                    s.bytes.end = b1;
                }
                _ => out.push(Shard {
                    rasters: i..i + 1,
                    levels: l0..l1,
                    bytes: b0..b1,
                }),
            }
        }
        out
    }

    pub fn whole(&self) -> Shard {
        Shard {
            rasters: 0..self.rasters.len(),
            levels: 0..self.levels.len(),
            bytes: 0..self.total_bytes,
        }
    }

    /// Materialise the pixels of one shard (TC pyramids from the memory-mapped overviews, WAC from
    /// its prebuilt pyramid), replacing whatever shard was held before.
    pub fn fill(&mut self, shard: &Shard) {
        self.data = Vec::new();
        let mut data = vec![0u8; shard.bytes.len()];
        let base = shard.bytes.start;
        let ptr = data.as_mut_ptr() as usize;
        let world = &*self;
        world.rasters[shard.rasters.clone()]
            .par_iter()
            .enumerate()
            .for_each(|(i, r)| {
                let dims: Vec<(usize, usize)> = r
                    .levels
                    .iter()
                    .map(|&l| {
                        let l = &world.levels[l as usize];
                        (l.width as usize, l.height as usize)
                    })
                    .collect();
                let start = world.levels[r.levels[0] as usize].offset as usize - base;
                let size: usize = dims.iter().map(|&(w, h)| w * h).sum();
                // Each raster owns a disjoint byte range of `data`.
                let out = unsafe { std::slice::from_raw_parts_mut((ptr + start) as *mut u8, size) };
                match world.sources[shard.rasters.start + i] {
                    Some((d, offset)) => {
                        let n0 = dims[0].0 * dims[0].1;
                        out[..n0]
                            .copy_from_slice(&world.maps[d][offset as usize..offset as usize + n0]);
                        let mut at = 0;
                        for k in 1..dims.len() {
                            let (pw, ph) = dims[k - 1];
                            let (src, dst) = out[at..].split_at_mut(pw * ph);
                            downsample_into(src, pw, ph, &mut dst[..dims[k].0 * dims[k].1]);
                            at += pw * ph;
                        }
                    }
                    None => out.copy_from_slice(&world.wac_pyramid),
                }
            });
        self.data = data;
        self.data_base = base;
    }

    /// Pixels of one level; the level must belong to the filled shard.
    pub fn level_data(&self, l: &Level) -> &[u8] {
        &self.data[l.offset as usize - self.data_base..]
    }

    /// Coarsest level whose pixel is no larger than LEVEL_SLACK x the cell; finest level as fallback.
    pub fn pick_level(&self, raster: usize, cell_km: f64, strict: bool) -> Option<u32> {
        let r = &self.rasters[raster];
        let mut best = None;
        for (k, &res) in r.res_km.iter().enumerate() {
            if res <= LEVEL_SLACK * cell_km {
                best = Some(r.levels[k]);
            }
        }
        if best.is_none() && !strict {
            Some(r.levels[0])
        } else {
            best
        }
    }

    pub fn raster_of(&self, pose: &Pose) -> &Raster {
        &self.rasters[self.owner[pose.level as usize] as usize]
    }

    /// Patch-centre latitude/longitude in degrees.
    pub fn centre(&self, pose: &Pose) -> (f64, f64) {
        let r = self.raster_of(pose);
        match &r.tc {
            Some(tc) => ortho_inverse(
                pose.a as f64,
                pose.b as f64,
                tc.lon0.to_radians(),
                tc.lat0.to_radians(),
            ),
            None => (pose.a as f64, pose.b as f64),
        }
        .to_degrees_pair()
    }
}

trait Degrees {
    fn to_degrees_pair(self) -> (f64, f64);
}
impl Degrees for (f64, f64) {
    fn to_degrees_pair(self) -> (f64, f64) {
        let lon = (self.0.to_degrees() + 540.0).rem_euclid(360.0) - 180.0;
        (lon, self.1.to_degrees())
    }
}

/// Orthographic inverse: tangent km -> (lon, lat) radians.
pub fn ortho_inverse(x: f64, y: f64, lon0: f64, lat0: f64) -> (f64, f64) {
    let (x, y) = (x / R_KM, y / R_KM);
    let rho = (x * x + y * y).sqrt();
    if rho == 0.0 {
        return (lon0, lat0);
    }
    let c = rho.min(1.0).asin();
    let lat = (c.cos() * lat0.sin() + y * c.sin() * lat0.cos() / rho)
        .clamp(-1.0, 1.0)
        .asin();
    let lon = lon0 + (x * c.sin()).atan2(rho * c.cos() * lat0.cos() - y * c.sin() * lat0.sin());
    (lon, lat)
}

// ---- CPU reference of the GPU resampler (same f32 arithmetic order) ----

fn fetch(data: &[u8], l: &Level, x: f32, y: f32) -> Option<f32> {
    let (fx, fy) = (x - 0.5, y - 0.5);
    let (x0f, y0f) = (fx.floor(), fy.floor());
    let (tx, ty) = (fx - x0f, fy - y0f);
    let (mut x0, mut y0) = (x0f as i32, y0f as i32);
    let (x1, y1);
    if l.kind == 1 {
        if x0 < 0 || y0 < 0 || x0 + 1 >= l.width || y0 + 1 >= l.height {
            return None;
        }
        x1 = x0 + 1;
        y1 = y0 + 1;
    } else {
        x0 = x0.rem_euclid(l.width);
        x1 = (x0 + 1) % l.width;
        y0 = y0.clamp(0, l.height - 1);
        y1 = (y0 + 1).min(l.height - 1);
    }
    let base = data;
    let at = |yy: i32, xx: i32| base[yy as usize * l.width as usize + xx as usize] as f32;
    let (a, b, c, d) = (at(y0, x0), at(y0, x1), at(y1, x0), at(y1, x1));
    if a == 0.0 || b == 0.0 || c == 0.0 || d == 0.0 {
        return None;
    }
    Some((a * (1.0 - tx) + b * tx) * (1.0 - ty) + (c * (1.0 - tx) + d * tx) * ty)
}

pub struct Sampled {
    pub values: Vec<f32>,
    pub in_data: bool,
    pub textured: bool,
    #[allow(dead_code)]
    pub mean: f32,
    #[allow(dead_code)]
    pub sd: f32,
}

pub fn sample_cpu(world: &World, p: &Pose, gw: usize) -> Sampled {
    let l = world.levels[p.level as usize];
    let gh = gw * 3 / 4;
    let (c, s) = (p.theta.cos(), p.theta.sin());
    let (sp, cp) = (p.b.sin(), p.b.cos());
    let (half_w, half_h, step) = (0.5 * p.w, 0.375 * p.w, p.w / gw as f32);
    let r_km = R_KM as f32;
    let pi = std::f32::consts::PI;
    let point = |u: f32, v: f32| -> Option<f32> {
        let (dx, dy) = (u * c - v * s, u * s + v * c);
        let (col, row);
        if l.kind == 1 {
            let (x, y) = (p.a + dx, p.b + dy);
            let q = &l.q;
            col = q[0] + q[1] * x + q[2] * y + q[3] * x * x + q[4] * x * y + q[5] * y * y;
            row = q[6] + q[7] * x + q[8] * y + q[9] * x * x + q[10] * x * y + q[11] * y * y;
        } else {
            let (x, y) = (dx / r_km, dy / r_km);
            let rho = (x * x + y * y).sqrt();
            let (mut lat, mut lon) = (p.b, p.a);
            if rho > 0.0 {
                let cc = rho.min(1.0).asin();
                let (sc, cosc) = (cc.sin(), cc.cos());
                lat = (cosc * sp + y * sc * cp / rho).clamp(-1.0, 1.0).asin();
                lon = p.a + (x * sc).atan2(rho * cosc * cp - y * sc * sp);
            }
            col = (lon + pi) * (1.0 / (2.0 * pi)) * l.width as f32;
            row = (0.5 * pi - lat) * (1.0 / pi) * l.height as f32;
        }
        fetch(world.level_data(&l), &l, col, row)
    };
    let mut values = vec![0f32; gw * gh];
    let mut in_data = true;
    'outer: for j in 0..gh {
        for i in 0..gw {
            let u = -half_w + (i as f32 + 0.5) * step;
            let v = half_h - (j as f32 + 0.5) * step;
            let q = 0.25 * step;
            let mut sum = 0.0;
            for (du, dv) in [(-q, q), (q, q), (-q, -q), (q, -q)] {
                match point(u + du, v + dv) {
                    Some(t) => sum += t,
                    None => {
                        in_data = false;
                        break 'outer;
                    }
                }
            }
            values[j * gw + i] = 0.25 * sum;
        }
    }
    if !in_data {
        return Sampled {
            values: vec![0.0; gw * gh],
            in_data,
            textured: false,
            mean: 0.0,
            sd: 0.0,
        };
    }
    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;
    let var: f32 = values.iter().map(|v| (v - mean) * (v - mean)).sum();
    let sd = (var / n).sqrt();
    // Same rule as the GPU: more than 10 % of cells at the stretch ceiling is a saturation plateau.
    let saturated = values.iter().filter(|&&v| v > 254.5).count() as f32 / n;
    let textured = sd >= 3.0 && saturated <= 0.10;
    let inv = if textured { 1.0 / var.sqrt() } else { 0.0 };
    for v in values.iter_mut() {
        *v = (*v - mean) * inv;
    }
    Sampled {
        values,
        in_data,
        textured,
        mean,
        sd,
    }
}

/// Cheap conservative footprint test used before sending TC poses to the GPU.
pub fn footprint_ok(world: &World, p: &Pose) -> bool {
    let l = &world.levels[p.level as usize];
    let data = world.level_data(l);
    let (c, s) = (p.theta.cos() as f64, p.theta.sin() as f64);
    let (hw, hh) = (0.5 * p.w as f64, 0.375 * p.w as f64);
    let qc: [f64; 6] = std::array::from_fn(|t| l.q[t] as f64);
    let qr: [f64; 6] = std::array::from_fn(|t| l.q[6 + t] as f64);
    for fu in [-1.0, 0.0, 1.0] {
        for fv in [-1.0, 0.0, 1.0] {
            let (u, v) = (fu * hw, fv * hh);
            let (x, y) = (p.a as f64 + u * c - v * s, p.b as f64 + u * s + v * c);
            let (col, row) = (quad(&qc, x, y), quad(&qr, x, y));
            if col < 1.0 || row < 1.0 || col >= l.width as f64 - 1.0 || row >= l.height as f64 - 1.0
            {
                return false;
            }
            if data[row as usize * l.width as usize + col as usize] == 0 {
                return false;
            }
        }
    }
    true
}
