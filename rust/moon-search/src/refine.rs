//! Coarse-to-fine refinement of the top-K list of every query (position, scale, rotation),
//! then per-frame selection over both mirrors. Uniform frames take the dark/bright lists.
use crate::{
    frames::{self, Queries},
    gpu::Gpu,
    search::entries_from_bytes,
    world::{Entry, Pose, R_KM, Source, World},
};
use anyhow::{Context, Result, ensure};
use serde::Serialize;
use std::{collections::HashSet, f32::consts::PI, fs, path::Path};

const SEEDS: usize = 12;
const ITERATIONS: usize = 6;

#[derive(Serialize, Clone)]
pub struct Candidate {
    pub raster: String,
    pub source: &'static str,
    pub mirror: bool,
    pub coarse: f32,
    pub score32: f32,
    pub score64: f32,
    pub level: u32,
    pub a: f32,
    pub b: f32,
    pub width_km: f32,
    pub theta_deg: f32,
    pub lon: f64,
    pub lat: f64,
    pub level_res_km: f64,
}

#[derive(Serialize)]
pub struct UniqueResult {
    pub unique: usize,
    pub kind: &'static str,
    pub best: Option<Candidate>,
    pub alternatives: Vec<Candidate>,
}

#[derive(Serialize)]
pub struct Matches {
    pub input: String,
    pub rate: String,
    pub frames: Vec<u32>,
    pub results: Vec<UniqueResult>,
}

fn same_place(world: &World, a: &Pose, b: &Pose, radius_frac: f32) -> bool {
    if world.owner[a.level as usize] != world.owner[b.level as usize] {
        return false;
    }
    let r = radius_frac * a.w.max(b.w);
    let d = match world.raster_of(a).source {
        Source::Tc => ((a.a - b.a).powi(2) + (a.b - b.b).powi(2)).sqrt(),
        Source::Wac => {
            let dl = (a.a - b.a + PI).rem_euclid(2.0 * PI) - PI;
            R_KM as f32 * ((dl * a.b.cos()).powi(2) + (a.b - b.b).powi(2)).sqrt()
        }
    };
    d < r
}

fn seeds(world: &World, list: &[Entry], radius_frac: f32, max: usize) -> Vec<Entry> {
    let mut out: Vec<Entry> = Vec::new();
    for e in list.iter().filter(|e| e.score.is_finite()) {
        if out.len() >= max {
            break;
        }
        if !out
            .iter()
            .any(|o| same_place(world, &o.pose, &e.pose, radius_frac))
        {
            out.push(*e);
        }
    }
    out
}

/// Width range a refined pose may drift into: one ladder step past the searched range, but never so
/// small that the 32x24 cells would have to upsample the finest level.
fn width_bounds(world: &World, p: &Pose) -> (f32, f32) {
    let r = world.raster_of(p);
    let finest = (r.res_km[0] * 32.0 / crate::world::LEVEL_SLACK) as f32;
    match r.source {
        Source::Tc => ((2.4 / 2f32.sqrt()).max(finest), 27.2 * 2f32.sqrt()),
        Source::Wac => ((38.4 / 2f32.sqrt()).max(finest), 434.4),
    }
}

fn moved(world: &World, p: &Pose, dx: f32, dy: f32, dlog: f32, dtheta: f32, gw: usize) -> Pose {
    let mut q = *p;
    match world.raster_of(p).source {
        Source::Tc => {
            q.a += dx;
            q.b += dy;
        }
        Source::Wac => {
            q.b = (q.b + dy / R_KM as f32).clamp(-1.45, 1.45);
            q.a = (q.a + dx / (R_KM as f32 * q.b.cos()) + PI).rem_euclid(2.0 * PI) - PI;
        }
    }
    let (lo, hi) = width_bounds(world, p);
    q.w = (q.w * dlog.exp()).clamp(lo, hi);
    q.theta = (q.theta + dtheta).rem_euclid(2.0 * PI);
    let raster = world.owner[p.level as usize] as usize;
    q.level = world
        .pick_level(raster, (q.w / gw as f32) as f64, false)
        .unwrap();
    q
}

fn describe(world: &World, e: &Entry, mirror: bool, coarse: f32, s64: f32) -> Candidate {
    let r = world.raster_of(&e.pose);
    let (lon, lat) = world.centre(&e.pose);
    let k = r.levels.iter().position(|&l| l == e.pose.level).unwrap();
    Candidate {
        raster: r.name.clone(),
        source: match r.source {
            Source::Tc => "tc",
            Source::Wac => "wac",
        },
        mirror,
        coarse,
        score32: e.score,
        score64: s64,
        level: e.pose.level,
        a: e.pose.a,
        b: e.pose.b,
        width_km: e.pose.w,
        theta_deg: e.pose.theta.to_degrees(),
        lon,
        lat,
        level_res_km: r.res_km[k],
    }
}

/// Pattern search of each seed's pose; all seeds must lie in the shard `gpu` holds. Returns 64x48 scores.
fn refine_seeds(world: &World, gpu: &Gpu, cur: &mut [(i32, Entry, f32)]) -> Result<Vec<f32>> {
    {
        let poses: Vec<Pose> = cur.iter().map(|c| c.1.pose).collect();
        let qidx: Vec<i32> = cur.iter().map(|c| c.0).collect();
        let s = gpu.score(&poses, &qidx, 32)?;
        for (c, s) in cur.iter_mut().zip(s) {
            c.1.score = s;
        }
    }
    let mut step = (1.0f32 / 32.0, (2f32).ln() / 4.0, PI / 8.0); // (position as fraction of width, log scale, rotation)
    for it in 0..ITERATIONS {
        let mut poses = Vec::with_capacity(cur.len() * 81);
        let mut qidx = Vec::with_capacity(cur.len() * 81);
        for (j, e, _) in cur.iter() {
            for ix in -1..=1 {
                for iy in -1..=1 {
                    for is in -1..=1 {
                        for ir in -1..=1 {
                            let d = step.0 * e.pose.w;
                            poses.push(moved(
                                world,
                                &e.pose,
                                ix as f32 * d,
                                iy as f32 * d,
                                is as f32 * step.1,
                                ir as f32 * step.2,
                                32,
                            ));
                            qidx.push(*j);
                        }
                    }
                }
            }
        }
        let scores = gpu.score(&poses, &qidx, 32)?;
        let mut improved = 0;
        for (n, c) in cur.iter_mut().enumerate() {
            let block = &scores[n * 81..(n + 1) * 81];
            let (bi, bs) = block
                .iter()
                .enumerate()
                .fold((0, f32::MIN), |m, (i, &s)| if s > m.1 { (i, s) } else { m });
            if bs > c.1.score {
                c.1 = Entry {
                    score: bs,
                    pose: poses[n * 81 + bi],
                };
                improved += 1;
            }
        }
        println!(
            "  refine iteration {}: {} of {} seeds improved",
            it + 1,
            improved,
            cur.len()
        );
        step = (step.0 * 0.5, step.1 * 0.5, step.2 * 0.5);
    }
    // 64x48 score of each refined seed (same pose, finer level where the data allows).
    let p64: Vec<Pose> = cur
        .iter()
        .map(|c| moved(world, &c.1.pose, 0.0, 0.0, 0.0, 0.0, 64))
        .collect();
    gpu.score(&p64, &cur.iter().map(|c| c.0).collect::<Vec<_>>(), 64)
}

/// `drop`: rasters whose list entries are ignored (a series judged unfit after the search, so the
/// lists need not be recomputed without it).
pub fn run(
    world: &mut World,
    queries: &Queries,
    search: &Path,
    out: &Path,
    shard_budget: usize,
    drop: &HashSet<String>,
) -> Result<()> {
    let nq = queries.textured.len() * 2;
    let k = Gpu::topk_size()?;
    let mut lists = entries_from_bytes(&fs::read(search.join("lists.bin")).context("lists.bin")?);
    ensure!(
        lists.len() == (nq + 2) * k,
        "lists.bin does not match the queries"
    );
    if !drop.is_empty() {
        let mut dropped = 0usize;
        for e in lists
            .iter_mut()
            .filter(|e| e.score.is_finite() && drop.contains(&world.raster_of(&e.pose).name))
        {
            e.score = f32::NAN;
            dropped += 1;
        }
        let empty = (0..nq)
            .filter(|j| {
                lists[j * k..(j + 1) * k]
                    .iter()
                    .all(|e| !e.score.is_finite())
            })
            .count();
        println!(
            "dropped {dropped} list entries from {} excluded rasters; {empty} query columns left empty",
            drop.len()
        );
    }
    // Seeds per GEMM column.
    let mut cur: Vec<(i32, Entry, f32)> = Vec::new(); // (column, entry, coarse score)
    for j in 0..nq {
        for e in seeds(world, &lists[j * k..(j + 1) * k], 0.25, SEEDS) {
            cur.push((j as i32, e, e.score));
        }
    }
    // Refine shard by shard: a seed never leaves its raster, so it stays in its shard.
    let mut s64 = vec![f32::NAN; cur.len()];
    for (si, shard) in world.shards(shard_budget).iter().enumerate() {
        let idx: Vec<usize> = (0..cur.len())
            .filter(|&i| shard.levels.contains(&(cur[i].1.pose.level as usize)))
            .collect();
        if idx.is_empty() {
            continue;
        }
        world.fill(shard);
        let world: &World = &*world;
        println!("refine shard {}: {} seeds", si + 1, idx.len());
        let gpu = Gpu::new(world, shard, &queries.q32, &queries.q64, nq, 1024)?;
        let mut part: Vec<(i32, Entry, f32)> = idx.iter().map(|&i| cur[i]).collect();
        let scores = refine_seeds(world, &gpu, &mut part)?;
        for ((&i, c), s) in idx.iter().zip(part).zip(scores) {
            cur[i] = c;
            s64[i] = s;
        }
    }

    let world: &World = &*world;
    let mut per_unique: Vec<Vec<Candidate>> = vec![Vec::new(); queries.set.uniques.len()];
    for ((j, e, coarse), s) in cur.iter().zip(s64) {
        let u = queries.textured[*j as usize / 2] as usize;
        per_unique[u].push(describe(world, e, j % 2 == 1, *coarse, s));
    }
    let dark = seeds(world, &lists[nq * k..(nq + 1) * k], 0.5, k);
    let bright = seeds(world, &lists[(nq + 1) * k..(nq + 2) * k], 0.5, k);
    let (mut nd, mut nb) = (0usize, 0usize);
    let mut results = Vec::new();
    for (u, unique) in queries.set.uniques.iter().enumerate() {
        let mut alts = std::mem::take(&mut per_unique[u]);
        alts.sort_by(|a, b| b.score32.total_cmp(&a.score32));
        if unique.sd >= frames::UNIFORM_SD {
            results.push(UniqueResult {
                unique: u,
                kind: "match",
                best: alts.first().cloned(),
                alternatives: alts,
            });
        } else {
            let is_dark = unique.mean < 128.0;
            let (list, n) = if is_dark {
                (&dark, &mut nd)
            } else {
                (&bright, &mut nb)
            };
            let best = list
                .get(*n % list.len().max(1))
                .map(|e| describe(world, e, false, e.score, f32::NAN));
            *n += 1;
            results.push(UniqueResult {
                unique: u,
                kind: if is_dark { "dark" } else { "bright" },
                best,
                alternatives: Vec::new(),
            });
        }
    }
    let matches = Matches {
        input: queries.set.input.clone(),
        rate: queries.set.rate.clone(),
        frames: queries.set.frames.clone(),
        results,
    };
    if let Some(dir) = out.parent() {
        fs::create_dir_all(dir)?;
    }
    fs::write(out, serde_json::to_vec(&matches)?)?;
    let scores: Vec<f32> = matches
        .results
        .iter()
        .filter_map(|r| {
            r.best
                .as_ref()
                .filter(|_| r.kind == "match")
                .map(|b| b.score32)
        })
        .collect();
    let mean = scores.iter().sum::<f32>() / scores.len().max(1) as f32;
    println!(
        "wrote {} ({} uniques, mean refined NCC {:.4})",
        out.display(),
        matches.results.len(),
        mean
    );
    Ok(())
}
