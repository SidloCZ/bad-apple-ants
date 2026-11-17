use anyhow::{Context, Result, ensure};
use memmap2::MmapOptions;
use rayon::prelude::*;
use serde::Deserialize;
use std::{
    fs::File,
    io::Write,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};

#[derive(Deserialize)]
struct Tile {
    file: String,
    bounds: [f64; 4],
}
#[derive(Deserialize)]
struct Manifest {
    tiles: Vec<Tile>,
}
#[derive(Clone, Copy)]
struct Best {
    hash: u64,
    x: u32,
    y: u32,
}
#[derive(Clone)]
struct Center {
    priority: u64,
    region: i32,
    lon: f64,
    lat: f64,
}
fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}
fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}
fn u64_at(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap())
}

fn scan(bytes: &[u8], bounds: [f64; 4], cell: usize, seed: u64) -> Result<(Vec<Center>, u64)> {
    ensure!(
        bytes.len() >= 16 && &bytes[..8] == b"LWMRLE01",
        "Invalid native RLE header"
    );
    let w = u32_at(bytes, 8) as usize;
    let h = u32_at(bytes, 12) as usize;
    ensure!(w > 0 && h > 0 && cell > 0, "Invalid native dimensions");
    let data = 16 + 8 * (h + 1);
    ensure!(bytes.len() >= data, "Missing row offsets");
    ensure!(
        data + 4 * u64_at(bytes, 16 + 8 * h) as usize == bytes.len(),
        "Invalid RLE length"
    );
    let nx = w.div_ceil(cell);
    let ny = h.div_ceil(cell);
    let mut cells = vec![
        Best {
            hash: u64::MAX,
            x: 0,
            y: 0
        };
        nx * ny
    ];
    let col = ((bounds[0] + 180.) / 30.).round() as u64;
    let row = ((90. - bounds[3]) / 30.).round() as u64;
    let mut examined = 0u64;
    for y in 0..h {
        let lat = bounds[3] - (y as f64 + 0.5) * (bounds[3] - bounds[1]) / h as f64;
        if !(-80.0..=84.0).contains(&lat) {
            continue;
        }
        let start = u64_at(bytes, 16 + 8 * y) as usize;
        let end = u64_at(bytes, 24 + 8 * y) as usize;
        ensure!(
            start <= end && data + end * 4 <= bytes.len(),
            "Invalid native row range"
        );
        for entry in bytes[data + 4 * start..data + 4 * end].chunks_exact(4) {
            let x = u32::from_le_bytes(entry.try_into().unwrap()) as usize;
            ensure!(x < w, "Native transition outside tile");
            // A row beginning white stores 0 as its initial state, not a coast.
            if x == 0 {
                continue;
            }
            examined += 1;
            let gx = col * w as u64 + x as u64;
            let gy = row * h as u64 + y as u64;
            let hash = mix((gy << 23) ^ gx ^ seed);
            let best = &mut cells[(y / cell) * nx + x / cell];
            if hash < best.hash {
                *best = Best {
                    hash,
                    x: x as u32,
                    y: y as u32,
                };
            }
        }
    }
    let global_nx = 12 * nx;
    let centers = cells
        .into_iter()
        .enumerate()
        .filter_map(|(index, best)| {
            if best.hash == u64::MAX {
                return None;
            }
            let cell_x = col as usize * nx + index % nx;
            let cell_y = row as usize * ny + index / nx;
            Some(Center {
                priority: mix((cell_y * global_nx + cell_x) as u64 ^ seed.rotate_left(29)),
                region: (row * 12 + col) as i32,
                lon: bounds[0] + best.x as f64 * (bounds[2] - bounds[0]) / w as f64,
                lat: bounds[3] - (best.y as f64 + 0.5) * (bounds[3] - bounds[1]) / h as f64,
            })
        })
        .collect();
    Ok((centers, examined))
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().collect();
    ensure!(
        args.len() >= 3,
        "Usage: coastline-native-index MANIFEST OUTPUT [COUNT] [CELL_PIXELS]"
    );
    let manifest_path = PathBuf::from(&args[1]);
    let output = PathBuf::from(&args[2]);
    let count = args
        .get(3)
        .map(|v| v.to_string_lossy().parse())
        .transpose()?
        .unwrap_or(64000usize);
    let cell = args
        .get(4)
        .map(|v| v.to_string_lossy().parse())
        .transpose()?
        .unwrap_or(375usize);
    let seed = 75213u64;
    ensure!(count > 0 && cell > 0, "Invalid sampling options");
    std::fs::create_dir_all(&output)?;
    let manifest: Manifest = serde_json::from_reader(File::open(&manifest_path)?)?;
    ensure!(manifest.tiles.len() == 72, "Need all original native tiles");
    let start = Instant::now();
    let done = AtomicUsize::new(0);
    let base = manifest_path.parent().context("Manifest has no parent")?;
    let partial: Vec<_> = manifest
        .tiles
        .par_iter()
        .map(|tile| -> Result<_> {
            let file = File::open(base.join(format!("{}.rle", tile.file)))?;
            let map = unsafe { MmapOptions::new().map(&file)? };
            let result = scan(&map, tile.bounds, cell, seed)?;
            let finished = done.fetch_add(1, Ordering::Relaxed) + 1;
            if finished % 6 == 0 {
                eprintln!(
                    "native coastline index: {finished}/72 tiles, {:.1}s",
                    start.elapsed().as_secs_f64()
                );
            }
            Ok(result)
        })
        .collect::<Result<Vec<_>>>()?;
    let examined: u64 = partial.iter().map(|(_, n)| n).sum();
    let mut centers: Vec<_> = partial.into_iter().flat_map(|(v, _)| v).collect();
    let occupied = centers.len();
    centers.par_sort_unstable_by_key(|v| v.priority);
    centers.truncate(count);
    centers.par_sort_unstable_by(|a, b| {
        a.region
            .cmp(&b.region)
            .then(a.lat.total_cmp(&b.lat))
            .then(a.lon.total_cmp(&b.lon))
    });
    let mut positions = File::create(output.join("centers.f64"))?;
    let mut regions = File::create(output.join("regions.i32"))?;
    for c in &centers {
        positions.write_all(&c.lon.to_le_bytes())?;
        positions.write_all(&c.lat.to_le_bytes())?;
        regions.write_all(&c.region.to_le_bytes())?;
    }
    let report = serde_json::json!({"source_manifest":manifest_path,"source":"original 0.3 arcsecond lossless RLE, all eligible rows scanned",
        "coastal_transition_points_examined":examined,"occupied_spatial_cells":occupied,"count":centers.len(),"cell_source_pixels":cell,"seed":seed,
        "latitude_range":[-80,84],"elapsed_seconds":start.elapsed().as_secs_f64(),"search":"finite native boundary sampling; no downsampled global atlas"});
    serde_json::to_writer_pretty(File::create(output.join("centers.json"))?, &report)?;
    eprintln!("{report}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tile(rows: &[Vec<u32>], w: u32) -> Vec<u8> {
        let mut b = b"LWMRLE01".to_vec();
        b.extend(w.to_le_bytes());
        b.extend((rows.len() as u32).to_le_bytes());
        let mut offset = 0u64;
        b.extend(offset.to_le_bytes());
        for r in rows {
            offset += r.len() as u64;
            b.extend(offset.to_le_bytes());
        }
        for r in rows {
            for v in r {
                b.extend(v.to_le_bytes());
            }
        }
        b
    }
    #[test]
    fn all_white_row_starts_are_not_false_coasts() {
        let bytes = tile(&vec![vec![0]; 16], 16);
        let (points, n) = scan(&bytes, [0., 0., 30., 30.], 4, 4).unwrap();
        assert_eq!(n, 0);
        assert!(points.is_empty());
    }
    #[test]
    fn exact_native_transitions_seed_each_occupied_cell_deterministically() {
        let rows = vec![vec![0, 2, 9]; 16];
        let bytes = tile(&rows, 16);
        let (a, n) = scan(&bytes, [0., 0., 30., 30.], 4, 4).unwrap();
        let (b, _) = scan(&bytes, [0., 0., 30., 30.], 4, 4).unwrap();
        assert_eq!(n, 32);
        assert_eq!(a.len(), 8);
        assert_eq!(
            a.iter().map(|c| (c.lon, c.lat)).collect::<Vec<_>>(),
            b.iter().map(|c| (c.lon, c.lat)).collect::<Vec<_>>()
        );
        assert!(a.iter().all(|c| c.lon == 3.75 || c.lon == 16.875));
    }
}
