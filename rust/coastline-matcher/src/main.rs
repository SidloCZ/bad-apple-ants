use anyhow::{Context, Result, bail, ensure};
use coastline_matcher::{Atlas, QueryFft, View, refine_fft};
use memmap2::{Mmap, MmapOptions};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Instant,
};

#[derive(Deserialize)]
struct Config {
    atlas: PathBuf,
    resolution: f64,
    candidates: PathBuf,
    features: PathBuf,
    queries: PathBuf,
    query_features: PathBuf,
    output: PathBuf,
    metrics: PathBuf,
    per_scale: usize,
    rotations: Vec<f64>,
    scales: Vec<f64>,
    threads: usize,
}

struct Npy {
    map: Mmap,
    offset: usize,
    shape: Vec<usize>,
    dtype: String,
}
impl Npy {
    fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        // Input files are immutable while a matching run is active.
        let map = unsafe { MmapOptions::new().map(&file)? };
        ensure!(
            map.len() >= 10 && &map[..6] == b"\x93NUMPY",
            "Invalid NPY header"
        );
        let (start, len) = match map[6] {
            1 => (10, u16::from_le_bytes(map[8..10].try_into()?) as usize),
            2 | 3 => {
                ensure!(map.len() >= 12, "Short NPY header");
                (12, u32::from_le_bytes(map[8..12].try_into()?) as usize)
            }
            _ => bail!("Unsupported NPY version"),
        };
        ensure!(map.len() >= start + len, "Truncated NPY header");
        let header = std::str::from_utf8(&map[start..start + len])?;
        ensure!(
            !header.contains("'fortran_order': True"),
            "Fortran NPY layout unsupported"
        );
        let dtype = header
            .split("'descr': '")
            .nth(1)
            .context("NPY dtype absent")?
            .split('\'')
            .next()
            .unwrap()
            .to_owned();
        let shape = header
            .split("'shape': (")
            .nth(1)
            .context("NPY shape absent")?
            .split(')')
            .next()
            .unwrap()
            .split(',')
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim().parse())
            .collect::<Result<Vec<usize>, _>>()?;
        let bytes: usize = match dtype.as_str() {
            "<f4" => 4,
            "<f8" => 8,
            "|u1" => 1,
            _ => bail!("Unsupported NPY dtype {dtype}"),
        };
        let size = shape
            .iter()
            .try_fold(bytes, |a, b| a.checked_mul(*b))
            .context("NPY dimensions overflow")?;
        ensure!(map.len() == start + len + size, "NPY payload size mismatch");
        Ok(Self {
            map,
            offset: start + len,
            shape,
            dtype,
        })
    }
    fn bytes(&self) -> &[u8] {
        &self.map[self.offset..]
    }
    fn floats(&self) -> Result<Vec<f32>> {
        ensure!(self.dtype == "<f4", "Expected little-endian f32");
        Ok(self
            .bytes()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect())
    }
    fn doubles(&self) -> Result<Vec<f64>> {
        ensure!(self.dtype == "<f8", "Expected little-endian f64");
        Ok(self
            .bytes()
            .chunks_exact(8)
            .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
            .collect())
    }
}

#[derive(Clone, Serialize, Deserialize)]
struct Record {
    #[serde(flatten)]
    view: View,
    alternatives: Vec<View>,
    frame: usize,
}

fn normalize(data: &mut [f32], size: usize) {
    data.par_chunks_exact_mut(size).for_each(|row| {
        let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
        for v in row {
            *v /= norm;
        }
    });
}

fn diverse(pool: &[(f32, usize)], candidates: &[f64], count: usize) -> Vec<usize> {
    let mut selected: Vec<usize> = Vec::new();
    for &(_, index) in pool {
        let c = &candidates[index * 4..index * 4 + 4];
        if selected.iter().any(|&p| {
            let p = &candidates[p * 4..p * 4 + 4];
            let separation =
                ((c[0] - p[0]) * 111.32 * c[1].to_radians().cos()).hypot((c[1] - p[1]) * 111.32);
            let turn = ((c[3] - p[3] + 180.0).rem_euclid(360.0) - 180.0).abs();
            separation < c[2] * 0.2 && turn < 45.0
        }) {
            continue;
        }
        selected.push(index);
        if selected.len() == count {
            break;
        }
    }
    selected
}

fn search(
    config: &Config,
    queries: &[f32],
    mut features: Vec<f32>,
    candidates: &[f64],
    dimension: usize,
) -> Result<Vec<Vec<usize>>> {
    normalize(&mut features, dimension);
    let mut q = queries.to_vec();
    normalize(&mut q, dimension);
    let mut groups: HashMap<u64, Vec<usize>> = HashMap::new();
    for (i, c) in candidates.chunks_exact(4).enumerate() {
        groups.entry(c[2].to_bits()).or_default().push(i);
    }
    let mut widths: Vec<_> = groups.keys().copied().collect();
    widths.sort_by(|a, b| f64::from_bits(*a).total_cmp(&f64::from_bits(*b)));
    let mut answer = vec![Vec::new(); q.len() / dimension];
    for width in widths {
        let group = &groups[&width];
        let mut packed = vec![0.0; group.len() * dimension];
        packed
            .par_chunks_exact_mut(dimension)
            .zip(group.par_iter())
            .for_each(|(row, &index)| {
                row.copy_from_slice(&features[index * dimension..(index + 1) * dimension])
            });
        let batches: Vec<_> = q
            .par_chunks(64 * dimension)
            .map(|batch| {
                let nq = batch.len() / dimension;
                let nc = group.len();
                let mut products = vec![0.0; nq * nc];
                // Row-major Q (nq x d) times the transpose of packed F (nc x d).
                // Disjoint result storage and immutable, correctly sized inputs.
                unsafe {
                    matrixmultiply::sgemm(
                        nq,
                        dimension,
                        nc,
                        1.0,
                        batch.as_ptr(),
                        dimension as isize,
                        1,
                        packed.as_ptr(),
                        1,
                        dimension as isize,
                        0.0,
                        products.as_mut_ptr(),
                        nc as isize,
                        1,
                    );
                }
                products
                    .chunks_exact(nc)
                    .map(|row| {
                        let limit = (config.per_scale * 16).min(nc);
                        let mut best: Vec<(f32, usize)> = Vec::with_capacity(limit + 1);
                        let mut floor = f32::NEG_INFINITY;
                        for (j, &similarity) in row.iter().enumerate() {
                            if similarity < floor {
                                continue;
                            }
                            let position = best.partition_point(|&(score, index)| {
                                score > similarity || (score == similarity && index < group[j])
                            });
                            best.insert(position, (similarity, group[j]));
                            if best.len() > limit {
                                best.pop();
                            }
                            if best.len() == limit {
                                floor = best.last().unwrap().0;
                            }
                        }
                        diverse(&best, candidates, config.per_scale)
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        for (dest, indices) in answer.iter_mut().zip(batches.into_iter().flatten()) {
            dest.extend(indices);
        }
        eprintln!(
            "rust search: width {} km, {} windows",
            f64::from_bits(width),
            group.len()
        );
    }
    Ok(answer)
}

fn main() -> Result<()> {
    let config_path = std::env::args_os()
        .nth(1)
        .context("Usage: coastline-matcher CONFIG.json")?;
    let config: Config = serde_json::from_reader(File::open(config_path)?)?;
    ensure!(
        config.per_scale > 0
            && config.threads > 0
            && !config.rotations.is_empty()
            && !config.scales.is_empty(),
        "Invalid search settings"
    );
    rayon::ThreadPoolBuilder::new()
        .num_threads(config.threads)
        .build_global()?;
    let started = Instant::now();
    let atlas_npy = Npy::open(&config.atlas)?;
    ensure!(
        atlas_npy.shape.len() == 2 && atlas_npy.dtype == "|u1",
        "Atlas must be a 2D u8 NPY"
    );
    let atlas = Atlas {
        pixels: atlas_npy.bytes(),
        rows: atlas_npy.shape[0],
        cols: atlas_npy.shape[1],
        resolution: config.resolution,
    };
    let c = Npy::open(&config.candidates)?;
    ensure!(
        c.shape.len() == 2 && c.shape[1] == 4,
        "Candidate shape invalid"
    );
    let candidates = c.doubles()?;
    drop(c);
    let q = Npy::open(&config.queries)?;
    ensure!(q.shape.len() == 3, "Query shape invalid");
    let (n, h, w) = (q.shape[0], q.shape[1], q.shape[2]);
    let queries = q.floats()?;
    drop(q);
    let f = Npy::open(&config.features)?;
    ensure!(
        f.shape.len() == 2 && f.shape[0] * 4 == candidates.len(),
        "Feature shape invalid"
    );
    let dimension = f.shape[1];
    let features = f.floats()?;
    drop(f);
    let qf = Npy::open(&config.query_features)?;
    ensure!(
        qf.shape == vec![n, dimension],
        "Query feature shape invalid"
    );
    let qfeatures = qf.floats()?;
    drop(qf);
    let load_seconds = started.elapsed().as_secs_f64();
    let nearest = search(&config, &qfeatures, features, &candidates, dimension)?;
    let search_seconds = started.elapsed().as_secs_f64() - load_seconds;
    let mut completed = HashSet::new();
    if config.output.exists() {
        // An interrupted run can leave a partial last line: cut the file back to the last
        // complete record so new records are not appended onto the fragment.
        let text = std::fs::read(&config.output)?;
        let keep = text.iter().rposition(|&b| b == b'\n').map_or(0, |i| i + 1);
        if keep < text.len() {
            File::options()
                .write(true)
                .open(&config.output)?
                .set_len(keep as u64)?;
        }
        for line in BufReader::new(&text[..keep]).lines() {
            if let Ok(record) = serde_json::from_str::<Record>(&line?) {
                completed.insert(record.frame);
            }
        }
    }
    let sink = Mutex::new(
        File::options()
            .create(true)
            .append(true)
            .open(&config.output)?,
    );
    let done = AtomicUsize::new(completed.len());
    let refine_started = Instant::now();
    (0..n)
        .into_par_iter()
        .filter(|i| !completed.contains(i))
        .try_for_each(|i| -> Result<()> {
            let query = &queries[i * w * h..(i + 1) * w * h];
            let mut fft = QueryFft::new(query, w, h);
            let mut views: Vec<_> = nearest[i]
                .iter()
                .filter_map(|&index| {
                    let c = &candidates[index * 4..index * 4 + 4];
                    refine_fft(
                        atlas,
                        query,
                        w,
                        h,
                        [c[0], c[1], c[2], c[3]],
                        &config.rotations,
                        &config.scales,
                        true,
                        &mut fft,
                    )
                })
                .collect();
            ensure!(!views.is_empty(), "No covered view for query {i}");
            views.sort_by(|a, b| a.distance.total_cmp(&b.distance));
            let view = views.remove(0);
            let record = Record {
                view,
                alternatives: views,
                frame: i,
            };
            {
                // One write per record, so a crash never leaves more than one partial line.
                let mut line = serde_json::to_vec(&record)?;
                line.push(b'\n');
                let mut sink = sink.lock().unwrap();
                sink.write_all(&line)?;
                sink.flush()?;
            }
            let count = done.fetch_add(1, Ordering::Relaxed) + 1;
            if count % 25 == 0 || count == n {
                eprintln!(
                    "rust refine: {count}/{n}, {:.1}s",
                    refine_started.elapsed().as_secs_f64()
                );
            }
            Ok(())
        })?;
    let metrics = serde_json::json!({"backend":"Rust + Rayon + RustFFT + matrixmultiply","threads":config.threads,
        "unique_queries":n,"dictionary_candidates":candidates.len()/4,"per_scale":config.per_scale,
        "rotations":config.rotations,"scales":config.scales,"load_seconds":load_seconds,
        "descriptor_search_seconds":search_seconds,"refinement_seconds":refine_started.elapsed().as_secs_f64(),
        "elapsed_seconds":started.elapsed().as_secs_f64(),"resumed_queries":completed.len()});
    serde_json::to_writer_pretty(File::create(&config.metrics)?, &metrics)?;
    eprintln!("{metrics}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn matrix_search_covers_each_scale_and_matches_known_cosine_neighbors() {
        let config = Config {
            atlas: PathBuf::new(),
            resolution: 1.0,
            candidates: PathBuf::new(),
            features: PathBuf::new(),
            queries: PathBuf::new(),
            query_features: PathBuf::new(),
            output: PathBuf::new(),
            metrics: PathBuf::new(),
            per_scale: 1,
            rotations: vec![0.0],
            scales: vec![1.0],
            threads: 2,
        };
        let candidates = vec![
            0.0, 0.0, 6.0, 0.0, 1.0, 0.0, 6.0, 0.0, 0.0, 0.0, 768.0, 0.0, 1.0, 0.0, 768.0, 0.0,
        ];
        let features = vec![1.0, 0.0, 0.0, 2.0, 0.0, 3.0, 4.0, 0.0];
        let found = search(&config, &[1.0, 0.0, 0.0, 1.0], features, &candidates, 2).unwrap();
        assert_eq!(found, vec![vec![0, 3], vec![1, 2]]);
    }
}
