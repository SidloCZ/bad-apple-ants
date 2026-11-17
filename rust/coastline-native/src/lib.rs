use anyhow::{Result, ensure};
use memmap2::{Mmap, MmapOptions};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    ffi::{CStr, c_char},
    fs::File,
    path::Path,
};

#[derive(Serialize, Deserialize, Clone)]
pub struct TileSpec {
    pub file: String,
    pub sha256: String,
    pub bounds: [f64; 4],
}
#[derive(Serialize, Deserialize)]
pub struct Manifest {
    pub source_directory: String,
    pub tiles: Vec<TileSpec>,
}

pub fn transitions(packed: &[u8], result: &mut Vec<u32>) {
    result.clear();
    let mut state = false;
    for (i, &byte) in packed.iter().enumerate() {
        if byte == 0 || byte == 255 {
            let value = byte == 255;
            if state != value {
                result.push((i * 8) as u32);
                state = value;
            }
        } else {
            for bit in 0..8 {
                let value = byte & (128 >> bit) != 0;
                if state != value {
                    result.push((i * 8 + bit) as u32);
                    state = value;
                }
            }
        }
    }
}

pub struct Tile {
    map: Mmap,
    width: usize,
    height: usize,
    data_start: usize,
    bounds: [f64; 4],
}
impl Tile {
    fn open(path: &Path, bounds: [f64; 4]) -> Result<Self> {
        let file = File::open(path)?;
        let map = unsafe { MmapOptions::new().map(&file)? };
        ensure!(
            map.len() >= 16 && &map[..8] == b"LWMRLE01",
            "Invalid RLE tile"
        );
        let width = u32::from_le_bytes(map[8..12].try_into()?) as usize;
        let height = u32::from_le_bytes(map[12..16].try_into()?) as usize;
        let data_start = 16 + 8 * (height + 1);
        ensure!(map.len() >= data_start, "Truncated RLE index");
        let last = u64::from_le_bytes(map[16 + height * 8..24 + height * 8].try_into()?) as usize;
        ensure!(data_start + last * 4 == map.len(), "Truncated RLE payload");
        Ok(Self {
            map,
            width,
            height,
            data_start,
            bounds,
        })
    }
    fn value(&self, lon: f64, lat: f64) -> u8 {
        let [left, bottom, right, top] = self.bounds;
        // Match the affine inverse used by GDAL. Subtracting the origin first
        // can move an exact source-pixel boundary one column to the left.
        let sx = 1.0 / ((right - left) / self.width as f64);
        let sy = -1.0 / ((top - bottom) / self.height as f64);
        let x = (lon * sx - left * sx).floor() as usize;
        let y = (lat * sy - top * sy).floor() as usize;
        if x >= self.width || y >= self.height {
            return 255;
        }
        let start =
            u64::from_le_bytes(self.map[16 + y * 8..24 + y * 8].try_into().unwrap()) as usize;
        let end = u64::from_le_bytes(self.map[24 + y * 8..32 + y * 8].try_into().unwrap()) as usize;
        let (mut lo, mut hi) = (start, end);
        while lo < hi {
            let mid = (lo + hi) / 2;
            let offset = self.data_start + mid * 4;
            let transition =
                u32::from_le_bytes(self.map[offset..offset + 4].try_into().unwrap()) as usize;
            if transition <= x {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        ((lo - start) % 2) as u8
    }
}

pub struct NativeAtlas {
    tiles: Vec<Option<Tile>>,
}
impl NativeAtlas {
    pub fn open(path: &Path) -> Result<Self> {
        let manifest: Manifest = serde_json::from_reader(File::open(path)?)?;
        let mut tiles: Vec<Option<Tile>> = (0..72).map(|_| None).collect();
        for tile in manifest.tiles {
            let col = ((tile.bounds[0] + 180.0) / 30.0).round() as usize;
            let row = ((90.0 - tile.bounds[3]) / 30.0).round() as usize;
            ensure!(col < 12 && row < 6, "Invalid tile bounds");
            tiles[row * 12 + col] = Some(Tile::open(
                &path.parent().unwrap().join(format!("{}.rle", tile.file)),
                tile.bounds,
            )?);
        }
        ensure!(
            tiles.iter().all(|t| t.is_some()),
            "Incomplete native RLE globe"
        );
        Ok(Self { tiles })
    }
    pub fn value(&self, lon: f64, lat: f64) -> u8 {
        if !(-180.0..180.0).contains(&lon) || !(-90.0..=90.0).contains(&lat) {
            return 255;
        }
        let col = ((lon + 180.0) / 30.0).floor() as usize;
        let row = ((90.0 - lat) / 30.0).floor() as usize;
        if col >= 12 || row >= 6 {
            return 255;
        }
        self.tiles[row * 12 + col].as_ref().unwrap().value(lon, lat)
    }
}

/// # Safety
/// `path` must be a valid nul-terminated UTF-8 path. The returned handle must be
/// released once with `coastline_native_close`, after all sampling calls finish.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn coastline_native_open(path: *const c_char) -> *mut NativeAtlas {
    if path.is_null() {
        return std::ptr::null_mut();
    }
    let result = std::panic::catch_unwind(|| {
        let path = unsafe { CStr::from_ptr(path) }.to_str().ok()?;
        NativeAtlas::open(Path::new(path))
            .ok()
            .map(|a| Box::into_raw(Box::new(a)))
    });
    result.ok().flatten().unwrap_or(std::ptr::null_mut())
}

/// # Safety
/// `handle` must be a live handle from open and `output` must hold w*h bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn coastline_native_sample(
    handle: *const NativeAtlas,
    lon: f64,
    lat: f64,
    width: f64,
    angle: f64,
    w: usize,
    h: usize,
    output: *mut u8,
) -> i32 {
    if handle.is_null() || output.is_null() || w == 0 || h == 0 {
        return -1;
    }
    let result = std::panic::catch_unwind(|| {
        let atlas = unsafe { &*handle };
        let t = coastline_matcher::transform(lon, lat, width, angle, w, h);
        let output = unsafe { std::slice::from_raw_parts_mut(output, w * h) };
        output.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for (x, pixel) in row.iter_mut().enumerate() {
                let (x, y) = (x as f64 + 0.5, y as f64 + 0.5);
                *pixel = atlas.value(t[0] * x + t[1] * y + t[2], t[3] * x + t[4] * y + t[5]);
            }
        });
    });
    if result.is_ok() { 0 } else { -2 }
}

/// Sample arbitrary lon/lat pixel centres, for accurately projected context maps.
///
/// # Safety
/// `handle` must remain live, `coordinates` must hold 2*count f64 values in
/// lon/lat order, and `output` must hold count bytes without overlapping them.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn coastline_native_sample_points(
    handle: *const NativeAtlas,
    coordinates: *const f64,
    count: usize,
    output: *mut u8,
) -> i32 {
    if handle.is_null()
        || coordinates.is_null()
        || output.is_null()
        || count == 0
        || count > isize::MAX as usize / 16
    {
        return -1;
    }
    let result = std::panic::catch_unwind(|| {
        let atlas = unsafe { &*handle };
        let coordinates = unsafe { std::slice::from_raw_parts(coordinates, count * 2) };
        let output = unsafe { std::slice::from_raw_parts_mut(output, count) };
        output.par_iter_mut().enumerate().for_each(|(i, pixel)| {
            *pixel = atlas.value(coordinates[2 * i], coordinates[2 * i + 1]);
        });
    });
    if result.is_ok() { 0 } else { -2 }
}

const CELL_DEGREES: f64 = 0.3 / 3600.0;

fn wrapped(atlas: &NativeAtlas, lon: f64, lat: f64) -> u8 {
    atlas.value((lon + 180.0).rem_euclid(360.0) - 180.0, lat)
}

/// Land fraction of every display pixel of an affine view (pixel-edge transform
/// `lon = a*x + b*y + c`, `lat = d*x + e*y + f`).
///
/// `supersample >= 1`: mean of s x s original cells sampled at sub-pixel centres
/// (s = 1 reproduces nearest-neighbour pixel centres). `supersample == 0`:
/// Gaussian-weighted (`sigma` cells) field of the 4x4 nearest cell centres,
/// contoured at 0.5 with a one-display-pixel anti-aliasing ramp.
/// Returns 1 if any sample falls outside the acquired globe.
///
/// # Safety
/// `handle` must be live, `affine` must hold 6 f64 and `output` width*height f32.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn coastline_native_render_coverage(
    handle: *const NativeAtlas,
    affine: *const f64,
    width: usize,
    height: usize,
    supersample: usize,
    pixels_per_cell: f64,
    sigma: f64,
    output: *mut f32,
) -> i32 {
    if handle.is_null()
        || affine.is_null()
        || output.is_null()
        || width == 0
        || height == 0
        || supersample > 16
        || !(sigma > 0.0)
    {
        return -1;
    }
    let result = std::panic::catch_unwind(|| {
        let atlas = unsafe { &*handle };
        let t = unsafe { std::slice::from_raw_parts(affine, 6) };
        let (a, b, c, d, e, f) = (t[0], t[1], t[2], t[3], t[4], t[5]);
        let output = unsafe { std::slice::from_raw_parts_mut(output, width * height) };
        let missing = std::sync::atomic::AtomicBool::new(false);
        output
            .par_chunks_mut(width)
            .enumerate()
            .for_each(|(y, row)| {
                for (x, pixel) in row.iter_mut().enumerate() {
                    let (px, py) = (x as f64, y as f64);
                    if supersample == 0 {
                        // Gaussian-weighted field over the 4x4 nearest cell centres
                        // (`sigma` in cells), contoured at 0.5. Removes the stair-step
                        // of rotated 10 m cells; nothing finer than the grid is added.
                        let (cx, cy) = (px + 0.5, py + 0.5);
                        let lon = a * cx + b * cy + c;
                        let lat = d * cx + e * cy + f;
                        let u = (lon + 180.0) / CELL_DEGREES - 0.5;
                        let v = (90.0 - lat) / CELL_DEGREES - 0.5;
                        let (u0, v0) = (u.floor(), v.floor());
                        let (mut weight, mut field) = (0f64, 0f64);
                        for dv in -1..=2 {
                            for du in -1..=2 {
                                let (cu, cv) = (u0 + du as f64, v0 + dv as f64);
                                let distance2 = (u - cu).powi(2) + (v - cv).powi(2);
                                let w = (-distance2 / (2.0 * sigma * sigma)).exp();
                                let clon = (cu + 0.5) * CELL_DEGREES - 180.0;
                                let clat = (90.0 - (cv + 0.5) * CELL_DEGREES)
                                    .clamp(-90.0 + CELL_DEGREES / 2.0, 90.0 - CELL_DEGREES / 2.0);
                                let value = wrapped(atlas, clon, clat);
                                if value > 1 {
                                    missing.store(true, std::sync::atomic::Ordering::Relaxed);
                                }
                                weight += w;
                                field += w * value as f64;
                            }
                        }
                        // Edge slope of a Gaussian-blurred step is ~1/(sigma*sqrt(2pi)) per cell.
                        let slope = pixels_per_cell * sigma * (2.0 * std::f64::consts::PI).sqrt();
                        *pixel = (0.5 + (field / weight - 0.5) * slope).clamp(0.0, 1.0) as f32;
                    } else {
                        let s = supersample as f64;
                        let mut total = 0u32;
                        for j in 0..supersample {
                            for i in 0..supersample {
                                let (cx, cy) =
                                    (px + (i as f64 + 0.5) / s, py + (j as f64 + 0.5) / s);
                                let value =
                                    wrapped(atlas, a * cx + b * cy + c, d * cx + e * cy + f);
                                if value > 1 {
                                    missing.store(true, std::sync::atomic::Ordering::Relaxed);
                                }
                                total += value as u32;
                            }
                        }
                        *pixel = total as f32 / (supersample * supersample) as f32;
                    }
                }
            });
        missing.load(std::sync::atomic::Ordering::Relaxed)
    });
    match result {
        Ok(false) => 0,
        Ok(true) => 1,
        Err(_) => -2,
    }
}

/// # Safety
/// The handle must have been returned by open, and must not be used afterward.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn coastline_native_close(handle: *mut NativeAtlas) {
    if !handle.is_null() {
        drop(unsafe { Box::from_raw(handle) });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn run_transitions_preserve_every_bit_including_rows_starting_white() {
        let bytes = [0, 255, 128, 1, 85, 170, 255, 0];
        let mut runs = Vec::new();
        transitions(&bytes, &mut runs);
        for x in 0..bytes.len() * 8 {
            let actual = (runs.partition_point(|&r| r <= x as u32) % 2) as u8;
            assert_eq!(actual, (bytes[x / 8] >> (7 - x % 8)) & 1);
        }
        transitions(&[255, 255], &mut runs);
        assert_eq!(runs, vec![0]);
        transitions(&[0, 0], &mut runs);
        assert!(runs.is_empty());
    }
}
