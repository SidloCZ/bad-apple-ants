//! CPU (Rayon) reference vs GPU: resampled patches, and exact top-K on a small pose subset.
use crate::{
    frames::Queries,
    gpu::Gpu,
    ladder,
    world::{DIM, Pose, World, sample_cpu},
};
use anyhow::{Result, ensure};
use rayon::prelude::*;

pub fn run(world: &World, queries: &Queries, jobs_per_source: usize, rotations: u32) -> Result<()> {
    let nq = queries.textured.len() * 2;
    let all = ladder::jobs(world, ladder::TC_STEPS);
    // A few TC images (all scales) and a few WAC latitude bands of the coarsest scales.
    let mut picked: Vec<_> = all
        .iter()
        .filter(|j| world.rasters[j.raster as usize].tc.is_some())
        .take(jobs_per_source)
        .cloned()
        .collect();
    picked.extend(
        all.iter()
            .filter(|j| world.rasters[j.raster as usize].tc.is_none() && j.step >= 22)
            .step_by(7)
            .take(jobs_per_source / 3),
    );
    let poses: Vec<Pose> = picked
        .iter()
        .flat_map(|j| ladder::poses(world, j, rotations))
        .collect();
    println!(
        "verify: {} jobs, {} poses, {} queries",
        picked.len(),
        poses.len(),
        nq
    );
    let gpu = Gpu::new(world, &world.whole(), &queries.q32, &queries.q64, nq, 16384)?;

    // 1) Resampled patches.
    let probe: Vec<Pose> = poses
        .iter()
        .step_by((poses.len() / 4000).max(1))
        .cloned()
        .collect();
    let (g, gv, _) = gpu.sample(&probe, 32)?;
    let mut flag_mismatch = 0;
    let mut max_diff = 0f32;
    for (i, p) in probe.iter().enumerate() {
        let c = sample_cpu(world, p, 32);
        let gflags = gv[i];
        if (gflags & 1 != 0) != c.in_data || (gflags & 2 != 0) != c.textured {
            flag_mismatch += 1;
            continue;
        }
        if c.textured {
            for k in 0..DIM {
                max_diff = max_diff.max((g[i * DIM + k] - c.values[k]).abs());
            }
        }
    }
    println!(
        "patches: {} probes, {} flag mismatches, max |gpu-cpu| of normalised values {:.2e}",
        probe.len(),
        flag_mismatch,
        max_diff
    );

    // 2) Exact top-1 per query on the subset.
    for part in poses.chunks(16384) {
        gpu.search(part)?;
    }
    let lists = gpu.results()?;
    let cpu: Vec<Option<Vec<f32>>> = poses
        .par_iter()
        .map(|p| {
            let s = sample_cpu(world, p, 32);
            s.textured.then_some(s.values)
        })
        .collect();
    let k = gpu.topk;
    let stats: Vec<(f32, f32, bool)> = (0..nq)
        .into_par_iter()
        .map(|j| {
            let q = &queries.q32[j * DIM..(j + 1) * DIM];
            let mut best = f32::MIN;
            for v in cpu.iter().flatten() {
                let s: f32 = v.iter().zip(q).map(|(a, b)| a * b).sum();
                best = best.max(s);
            }
            let g = lists[j * k];
            // The GPU's chosen pose, rescored on the CPU, must be (near-)optimal too.
            let rescored = {
                let s = sample_cpu(world, &g.pose, 32);
                if s.textured {
                    s.values.iter().zip(q).map(|(a, b)| a * b).sum()
                } else {
                    f32::MIN
                }
            };
            (best, g.score, (best - rescored).abs() <= 2e-3)
        })
        .collect();
    let max_score_diff = stats.iter().map(|s| (s.0 - s.1).abs()).fold(0f32, f32::max);
    let agree = stats.iter().filter(|s| s.2).count();
    println!(
        "top-1: {}/{} queries agree (GPU pose within 2e-3 of the CPU optimum), max |cpu best - gpu fp16 score| {:.2e}",
        agree, nq, max_score_diff
    );
    ensure!(
        flag_mismatch * 100 <= probe.len(),
        "too many validity mismatches"
    );
    ensure!(
        agree * 1000 >= nq * 995,
        "GPU top-1 disagrees with the CPU reference"
    );
    println!("verify OK");
    Ok(())
}
