//! Exhaustive coarse search: every pose of every job against every textured query (x2 mirrors).
use crate::{
    frames,
    gpu::Gpu,
    ladder::{self, Job},
    world::{Entry, Pose, World},
};
use anyhow::{Context, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::Path,
    sync::mpsc,
    time::{Duration, Instant},
};

#[derive(Serialize, Deserialize)]
pub struct Checkpoint {
    pub next_job: usize,
    pub jobs: usize,
    pub poses: u64,
    pub cols: usize,
    pub topk: usize,
    pub rotations: u32,
    pub seconds: f64,
    /// FNV-1a of the queries, the job list and the raster names; a checkpoint only resumes
    /// a run with the same inputs (older checkpoints without it are not resumed).
    #[serde(default)]
    pub fingerprint: String,
}

fn fingerprint(queries: &frames::Queries, jobs: &[Job], world: &World, rotations: u32) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |bytes: &[u8]| {
        for &b in bytes {
            h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    eat(&rotations.to_le_bytes());
    for v in queries.q32.iter().chain(&queries.q64) {
        eat(&v.to_le_bytes());
    }
    for j in jobs {
        eat(&j.raster.to_le_bytes());
        eat(&j.step.to_le_bytes());
        eat(&j.part.to_le_bytes());
    }
    for r in &world.rasters {
        eat(r.name.as_bytes());
        eat(&[0]);
    }
    format!("{h:016x}")
}

pub fn entries_to_bytes(e: &[Entry]) -> Vec<u8> {
    let bytes = unsafe { std::slice::from_raw_parts(e.as_ptr() as *const u8, size_of_val(e)) };
    bytes.to_vec()
}

pub fn entries_from_bytes(b: &[u8]) -> Vec<Entry> {
    let n = b.len() / size_of::<Entry>();
    let mut out = vec![Entry::default(); n];
    unsafe {
        std::ptr::copy_nonoverlapping(
            b.as_ptr(),
            out.as_mut_ptr() as *mut u8,
            n * size_of::<Entry>(),
        )
    };
    out
}

fn save(out: &Path, gpu: &Gpu, ck: &Checkpoint) -> Result<()> {
    let lists = gpu.results()?;
    fs::write(out.join("lists.tmp"), entries_to_bytes(&lists))?;
    fs::rename(out.join("lists.tmp"), out.join("lists.bin"))?;
    fs::write(out.join("checkpoint.tmp"), serde_json::to_vec_pretty(ck)?)?;
    fs::rename(out.join("checkpoint.tmp"), out.join("checkpoint.json"))?;
    Ok(())
}

pub fn run(
    world: &mut World,
    queries: &frames::Queries,
    out: &Path,
    chunk: usize,
    rotations: u32,
    tc_steps: std::ops::RangeInclusive<u32>,
    limit_jobs: Option<usize>,
    shard_budget: usize,
) -> Result<()> {
    fs::create_dir_all(out)?;
    let nq = queries.textured.len() * 2;
    let mut all_jobs = ladder::jobs(world, tc_steps);
    if let Some(n) = limit_jobs {
        all_jobs.truncate(n);
    }
    let shards = world.shards(shard_budget);
    println!(
        "{} shard(s) of at most {:.1} GB",
        shards.len(),
        shard_budget as f64 / 1e9
    );
    let topk = Gpu::topk_size()?;
    let mut ck = Checkpoint {
        next_job: 0,
        jobs: all_jobs.len(),
        poses: 0,
        cols: nq + 2,
        topk,
        rotations,
        seconds: 0.0,
        fingerprint: fingerprint(queries, &all_jobs, world, rotations),
    };
    // The running top-K lists travel from shard to shard (and through checkpoints).
    let mut lists: Option<Vec<Entry>> = None;
    if let Ok(text) = fs::read(out.join("checkpoint.json")) {
        let old: Checkpoint = serde_json::from_slice(&text)?;
        if old.fingerprint == ck.fingerprint
            && old.cols == ck.cols
            && old.jobs == ck.jobs
            && old.topk == ck.topk
        {
            lists = Some(entries_from_bytes(
                &fs::read(out.join("lists.bin")).context("lists.bin")?,
            ));
            println!(
                "resuming at job {}/{} ({} poses done)",
                old.next_job, old.jobs, old.poses
            );
            ck = old;
        } else {
            println!(
                "checkpoint.json belongs to different inputs; starting this search from the beginning"
            );
        }
    }
    let started = Instant::now();
    let base_seconds = ck.seconds;
    for (si, shard) in shards.iter().enumerate() {
        // Jobs are generated raster by raster, so each shard owns a contiguous job range.
        let j0 = all_jobs.partition_point(|j| (j.raster as usize) < shard.rasters.start);
        let j1 = all_jobs.partition_point(|j| (j.raster as usize) < shard.rasters.end);
        if j1 <= ck.next_job {
            continue;
        }
        world.fill(shard);
        let world: &World = &*world;
        let gpu = Gpu::new(world, shard, &queries.q32, &queries.q64, nq, chunk)?;
        if let Some(l) = &lists {
            gpu.set_results(l)?;
        }
        println!(
            "shard {}/{}: rasters {:?}, {:.2} GB, jobs {}..{}",
            si + 1,
            shards.len(),
            shard.rasters,
            shard.bytes.len() as f64 / 1e9,
            j0.max(ck.next_job),
            j1
        );
        // Generator thread: batches of jobs -> poses, a few batches ahead of the GPU.
        const BATCH: usize = 192;
        let (tx, rx) = mpsc::sync_channel::<(usize, Vec<Pose>)>(3);
        let jobs: &[Job] = &all_jobs[..j1];
        let start_job = j0.max(ck.next_job);
        std::thread::scope(|scope| -> Result<()> {
            scope.spawn(move || {
                let mut j = start_job;
                while j < jobs.len() {
                    let end = (j + BATCH).min(jobs.len());
                    let poses: Vec<Pose> = jobs[j..end]
                        .par_iter()
                        .flat_map_iter(|job| ladder::poses(world, job, rotations))
                        .collect();
                    if tx.send((end, poses)).is_err() {
                        return;
                    }
                    j = end;
                }
            });
            let mut last_save = Instant::now();
            let mut last_print = Instant::now();
            for (end, poses) in rx {
                for part in poses.chunks(chunk) {
                    gpu.search(part)?;
                }
                ck.poses += poses.len() as u64;
                ck.next_job = end;
                ck.seconds = base_seconds + started.elapsed().as_secs_f64();
                if last_print.elapsed() > Duration::from_secs(30) || end == ck.jobs {
                    let rate = ck.poses as f64 / ck.seconds.max(1e-3);
                    println!(
                        "job {}/{}  poses {:.3e}  {:.2e} poses/s  {:.0} s",
                        end, ck.jobs, ck.poses as f64, rate, ck.seconds
                    );
                    last_print = Instant::now();
                }
                if last_save.elapsed() > Duration::from_secs(300) {
                    save(out, &gpu, &ck)?;
                    last_save = Instant::now();
                }
            }
            Ok(())
        })?;
        save(out, &gpu, &ck)?;
        lists = Some(gpu.results()?);
    }
    println!("search done: {} poses in {:.0} s", ck.poses, ck.seconds);
    Ok(())
}
