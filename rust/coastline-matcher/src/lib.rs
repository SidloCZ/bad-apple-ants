use rayon::prelude::*;
use rustfft::{Fft, FftPlanner, num_complex::Complex32};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

#[derive(Clone, Copy)]
pub struct Atlas<'a> {
    pub pixels: &'a [u8],
    pub rows: usize,
    pub cols: usize,
    pub resolution: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct View {
    pub longitude: f64,
    pub latitude: f64,
    pub width_km: f64,
    pub height_km: f64,
    pub rotation_clockwise: f64,
    pub mse: f64,
    pub fft_distance: f64,
    pub distance: f64,
    pub metric: String,
    pub iou: f64,
    pub footprint_lonlat: Vec<[f64; 2]>,
}

pub fn transform(lon: f64, lat: f64, width: f64, angle: f64, w: usize, h: usize) -> [f64; 6] {
    let (s, c) = angle.to_radians().sin_cos();
    let dx = width / w as f64;
    let ls = 111.32 * lat.to_radians().cos();
    [
        c * dx / ls,
        -s * dx / ls,
        lon + (-c * w as f64 / 2.0 + s * h as f64 / 2.0) * dx / ls,
        -s * dx / 111.32,
        -c * dx / 111.32,
        lat + (s * w as f64 / 2.0 + c * h as f64 / 2.0) * dx / 111.32,
    ]
}

fn point(t: &[f64; 6], x: f64, y: f64) -> [f64; 2] {
    [t[0] * x + t[1] * y + t[2], t[3] * x + t[4] * y + t[5]]
}

impl Atlas<'_> {
    pub fn sample(&self, candidate: [f64; 4], w: usize, h: usize) -> (Vec<f32>, Vec<u8>) {
        let t = transform(candidate[0], candidate[1], candidate[2], candidate[3], w, h);
        let mut values = vec![0.0; w * h];
        let mut valid = vec![0; w * h];
        for y in 0..h {
            for x in 0..w {
                let p = point(&t, x as f64 + 0.5, y as f64 + 0.5);
                let ix = (p[0] + 180.0) / self.resolution - 0.5;
                let iy = (90.0 - p[1]) / self.resolution - 0.5;
                if ix >= 0.0
                    && iy >= 0.0
                    && ix <= self.cols as f64 - 1.0
                    && iy <= self.rows as f64 - 1.0
                {
                    let value = self.pixels
                        [(iy + 0.5).floor() as usize * self.cols + (ix + 0.5).floor() as usize];
                    if value != 255 {
                        values[y * w + x] = value as f32;
                        valid[y * w + x] = 1;
                    }
                }
            }
        }
        (values, valid)
    }
}

fn fast_len(n: usize) -> usize {
    (n..)
        .find(|&v| {
            let mut k = v;
            for p in [2, 3, 5] {
                while k % p == 0 {
                    k /= p;
                }
            }
            k == 1
        })
        .unwrap()
}

pub struct Plan {
    nx: usize,
    ny: usize,
    row: Arc<dyn Fft<f32>>,
    col: Arc<dyn Fft<f32>>,
    irow: Arc<dyn Fft<f32>>,
    icol: Arc<dyn Fft<f32>>,
    scratch_len: usize,
}

impl Plan {
    fn new(w: usize, h: usize) -> Self {
        let nx = fast_len(4 * w - 1);
        let ny = fast_len(4 * h - 1);
        let mut p = FftPlanner::<f32>::new();
        let row = p.plan_fft_forward(nx);
        let col = p.plan_fft_forward(ny);
        let irow = p.plan_fft_inverse(nx);
        let icol = p.plan_fft_inverse(ny);
        let scratch_len = [&row, &col, &irow, &icol]
            .iter()
            .map(|f| f.get_inplace_scratch_len())
            .max()
            .unwrap();
        Self {
            nx,
            ny,
            row,
            col,
            irow,
            icol,
            scratch_len,
        }
    }

    fn fft(
        &self,
        data: &mut [Complex32],
        transposed: &mut [Complex32],
        scratch: &mut [Complex32],
        inverse: bool,
    ) {
        let (row, col) = if inverse {
            (&self.irow, &self.icol)
        } else {
            (&self.row, &self.col)
        };
        for r in data.chunks_exact_mut(self.nx) {
            row.process_with_scratch(r, scratch);
        }
        for y in 0..self.ny {
            for x in 0..self.nx {
                transposed[x * self.ny + y] = data[y * self.nx + x];
            }
        }
        for c in transposed.chunks_exact_mut(self.ny) {
            col.process_with_scratch(c, scratch);
        }
        for y in 0..self.ny {
            for x in 0..self.nx {
                data[y * self.nx + x] = transposed[x * self.ny + y];
            }
        }
    }
}

type PlanCache = Mutex<HashMap<(usize, usize), Arc<Plan>>>;
pub fn plan(w: usize, h: usize) -> Arc<Plan> {
    static PLANS: OnceLock<PlanCache> = OnceLock::new();
    let mut cache = PLANS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap();
    cache
        .entry((w, h))
        .or_insert_with(|| Arc::new(Plan::new(w, h)))
        .clone()
}

pub struct QueryFft {
    plan: Arc<Plan>,
    spectrum: Vec<Complex32>,
    buffer: Vec<Complex32>,
    transposed: Vec<Complex32>,
    scratch: Vec<Complex32>,
    query_mass: f64,
    query_squares: f64,
    w: usize,
    h: usize,
}

impl QueryFft {
    pub fn new(query: &[f32], w: usize, h: usize) -> Self {
        let plan = plan(w, h);
        let mut spectrum = vec![Complex32::default(); plan.nx * plan.ny];
        let mut transposed = spectrum.clone();
        let mut scratch = vec![Complex32::default(); plan.scratch_len];
        for y in 0..h {
            for x in 0..w {
                spectrum[y * plan.nx + x].re = query[(h - 1 - y) * w + w - 1 - x];
            }
        }
        plan.fft(&mut spectrum, &mut transposed, &mut scratch, false);
        let buffer = vec![Complex32::default(); spectrum.len()];
        Self {
            plan,
            spectrum,
            buffer,
            transposed,
            scratch,
            query_mass: query.iter().map(|&v| v as f64).sum(),
            query_squares: query.iter().map(|&v| (v * v) as f64).sum(),
            w,
            h,
        }
    }

    pub fn best(
        &mut self,
        scene: &[f32],
        valid: &[u8],
        sw: usize,
        sh: usize,
        iou: bool,
    ) -> Option<(usize, usize, f64)> {
        let mut sums = vec![0.0f64; (sw + 1) * (sh + 1)];
        let mut missing = vec![0u32; sums.len()];
        for y in 0..sh {
            let mut mass = 0.0;
            let mut bad = 0;
            for x in 0..sw {
                let v = scene[y * sw + x] as f64;
                mass += if iou { v } else { v * v };
                bad += u32::from(valid[y * sw + x] == 0);
                sums[(y + 1) * (sw + 1) + x + 1] = sums[y * (sw + 1) + x + 1] + mass;
                missing[(y + 1) * (sw + 1) + x + 1] = missing[y * (sw + 1) + x + 1] + bad;
            }
        }
        self.buffer.fill(Complex32::default());
        for y in 0..sh {
            for x in 0..sw {
                self.buffer[y * self.plan.nx + x].re = scene[y * sw + x];
            }
        }
        self.plan.fft(
            &mut self.buffer,
            &mut self.transposed,
            &mut self.scratch,
            false,
        );
        for (v, q) in self.buffer.iter_mut().zip(&self.spectrum) {
            *v *= q;
        }
        self.plan.fft(
            &mut self.buffer,
            &mut self.transposed,
            &mut self.scratch,
            true,
        );
        let norm = (self.plan.nx * self.plan.ny) as f64;
        let mut best = None;
        let mut best_score = f64::INFINITY;
        for y in 0..=sh - self.h {
            for x in 0..=sw - self.w {
                let a = y * (sw + 1) + x;
                let b = y * (sw + 1) + x + self.w;
                let c = (y + self.h) * (sw + 1) + x;
                let d = (y + self.h) * (sw + 1) + x + self.w;
                if missing[d] + missing[a] != missing[b] + missing[c] {
                    continue;
                }
                let mass = sums[d] + sums[a] - sums[b] - sums[c];
                let corr = (self.buffer[(y + self.h - 1) * self.plan.nx + x + self.w - 1].re
                    as f64
                    / norm)
                    .max(0.0);
                let score = if iou {
                    let union = (mass + self.query_mass - corr).max(0.0);
                    if union > 1e-5 {
                        (1.0 - corr / union).clamp(0.0, 1.0)
                    } else {
                        0.0
                    }
                } else {
                    ((mass + self.query_squares - 2.0 * corr) / (self.w * self.h) as f64).max(0.0)
                };
                if score < best_score {
                    best_score = score;
                    best = Some((x, y, score));
                }
            }
        }
        best
    }
}

pub fn refine(
    atlas: Atlas<'_>,
    query: &[f32],
    w: usize,
    h: usize,
    candidate: [f64; 4],
    rotations: &[f64],
    scales: &[f64],
    iou: bool,
) -> Option<View> {
    let mut fft = QueryFft::new(query, w, h);
    refine_fft(
        atlas, query, w, h, candidate, rotations, scales, iou, &mut fft,
    )
}

pub fn refine_fft(
    atlas: Atlas<'_>,
    query: &[f32],
    w: usize,
    h: usize,
    candidate: [f64; 4],
    rotations: &[f64],
    scales: &[f64],
    iou_metric: bool,
    fft: &mut QueryFft,
) -> Option<View> {
    let [lon, lat, width, angle] = candidate;
    let mut best: Option<View> = None;
    for &scale in scales {
        for &delta in rotations {
            let ew = width * scale;
            let a = (angle + delta).rem_euclid(360.0);
            let (scene, valid) = atlas.sample([lon, lat, ew * 3.0, a], w * 3, h * 3);
            let Some((x, y, score)) = fft.best(&scene, &valid, w * 3, h * 3, iou_metric) else {
                continue;
            };
            let t = transform(lon, lat, ew * 3.0, a, w * 3, h * 3);
            let center = point(&t, x as f64 + w as f64 / 2.0, y as f64 + h as f64 / 2.0);
            let (shifted, covered) = atlas.sample([center[0], center[1], ew, a], w, h);
            if covered.contains(&0) {
                continue;
            }
            let mut mse = 0.0;
            let mut intersection = 0.0;
            let mut union = 0.0;
            for (&s, &q) in shifted.iter().zip(query) {
                let (s, q) = (s as f64, q as f64);
                mse += (s - q) * (s - q);
                intersection += s * q;
                union += s + q - s * q;
            }
            mse /= (w * h) as f64;
            let iou = if union > 1e-5 {
                intersection / union
            } else {
                1.0
            };
            let distance = if iou_metric { 1.0 - iou } else { mse };
            if best.as_ref().is_some_and(|b| distance >= b.distance) {
                continue;
            }
            let t = transform(center[0], center[1], ew, a, w, h);
            let footprint_lonlat = vec![
                point(&t, 0.0, 0.0),
                point(&t, w as f64, 0.0),
                point(&t, w as f64, h as f64),
                point(&t, 0.0, h as f64),
            ];
            best = Some(View {
                longitude: center[0],
                latitude: center[1],
                width_km: ew,
                height_km: ew * h as f64 / w as f64,
                rotation_clockwise: a,
                mse,
                fft_distance: score,
                distance,
                metric: if iou_metric { "iou" } else { "mse" }.into(),
                iou,
                footprint_lonlat,
            });
        }
    }
    best
}

/// C ABI for small refinement batches. Output rows have 10 values.
///
/// # Safety
/// All pointers must refer to the stated contiguous buffers for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn coastline_refine_batch(
    pixels: *const u8,
    rows: usize,
    cols: usize,
    resolution: f64,
    query: *const f32,
    w: usize,
    h: usize,
    candidates: *const f64,
    count: usize,
    rotations: *const f64,
    nrot: usize,
    scales: *const f64,
    nscale: usize,
    output: *mut f64,
) -> i32 {
    if pixels.is_null()
        || query.is_null()
        || candidates.is_null()
        || rotations.is_null()
        || scales.is_null()
        || output.is_null()
        || rows == 0
        || cols == 0
        || w == 0
        || h == 0
        || nrot == 0
        || nscale == 0
        || resolution <= 0.0
    {
        return -1;
    }
    let result = std::panic::catch_unwind(|| {
        let atlas = Atlas {
            pixels: unsafe { std::slice::from_raw_parts(pixels, rows * cols) },
            rows,
            cols,
            resolution,
        };
        let query = unsafe { std::slice::from_raw_parts(query, w * h) };
        let candidates = unsafe { std::slice::from_raw_parts(candidates, count * 4) };
        let rotations = unsafe { std::slice::from_raw_parts(rotations, nrot) };
        let scales = unsafe { std::slice::from_raw_parts(scales, nscale) };
        let results: Vec<_> = candidates
            .par_chunks_exact(4)
            .map(|c| {
                refine(
                    atlas,
                    query,
                    w,
                    h,
                    [c[0], c[1], c[2], c[3]],
                    rotations,
                    scales,
                    true,
                )
            })
            .collect();
        let out = unsafe { std::slice::from_raw_parts_mut(output, count * 10) };
        for (row, result) in out.chunks_exact_mut(10).zip(results) {
            if let Some(v) = result {
                row.copy_from_slice(&[
                    v.longitude,
                    v.latitude,
                    v.width_km,
                    v.height_km,
                    v.rotation_clockwise,
                    v.mse,
                    v.fft_distance,
                    v.distance,
                    v.iou,
                    1.0,
                ]);
            } else {
                row.fill(f64::NAN);
                row[9] = 0.0;
            }
        }
    });
    if result.is_ok() { 0 } else { -2 }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fft_agrees_with_direct_iou_and_masks_unknown_pixels() {
        let (w, h, sw, sh) = (5, 4, 15, 12);
        let scene: Vec<f32> = (0..sw * sh)
            .map(|i| {
                if ((i * 73 + i * i * 11) % 17) < 8 {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        let query: Vec<f32> = (0..w * h)
            .map(|i| ((i * 37) % 101) as f32 / 100.0)
            .collect();
        let mut valid = vec![1; sw * sh];
        valid[3 * sw + 6] = 0;
        let mut fft = QueryFft::new(&query, w, h);
        for iou in [true, false] {
            let (x, y, score) = fft.best(&scene, &valid, sw, sh, iou).unwrap();
            let direct = |x: usize, y: usize| {
                let mut intersect = 0.0f64;
                let mut union = 0.0;
                let mut mse = 0.0;
                for dy in 0..h {
                    for dx in 0..w {
                        if valid[(y + dy) * sw + x + dx] == 0 {
                            return f64::INFINITY;
                        }
                        let (s, q) = (
                            scene[(y + dy) * sw + x + dx] as f64,
                            query[dy * w + dx] as f64,
                        );
                        intersect += s * q;
                        union += s + q - s * q;
                        mse += (s - q).powi(2);
                    }
                }
                if iou {
                    1.0 - intersect / union
                } else {
                    mse / (w * h) as f64
                }
            };
            let mut expected = f64::INFINITY;
            for y in 0..=sh - h {
                for x in 0..=sw - w {
                    expected = expected.min(direct(x, y));
                }
            }
            assert!((score - expected).abs() < 2e-6);
            assert!((direct(x, y) - expected).abs() < 2e-6);
        }
    }
    #[test]
    fn constants_and_missing_coverage() {
        for v in [0.0, 1.0] {
            let mut fft = QueryFft::new(&vec![v; 12], 4, 3);
            assert!(
                fft.best(&vec![v; 108], &vec![1; 108], 12, 9, true)
                    .unwrap()
                    .2
                    < 1e-5
            );
            assert!(
                fft.best(&vec![v; 108], &vec![0; 108], 12, 9, true)
                    .is_none()
            );
        }
    }
    #[test]
    fn geographic_transform_preserves_scale_center_and_angle() {
        for a in [0.0, 30.0, 90.0, 270.0] {
            let t = transform(139.7, 35.5, 48.0, a, 32, 24);
            let center = point(&t, 16.0, 12.0);
            assert!((center[0] - 139.7).abs() < 1e-10 && (center[1] - 35.5).abs() < 1e-10);
            let l = point(&t, 0.0, 12.0);
            let r = point(&t, 32.0, 12.0);
            let east = (r[0] - l[0]) * 111.32 * 35.5f64.to_radians().cos();
            let north = (r[1] - l[1]) * 111.32;
            assert!((east.hypot(north) - 48.0).abs() < 1e-8);
            assert!((north + 48.0 * a.to_radians().sin()).abs() < 1e-8);
        }
    }
}
