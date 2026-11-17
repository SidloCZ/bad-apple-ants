mod frames;
mod gpu;
mod ladder;
mod refine;
mod search;
mod verify;
mod world;

use anyhow::{Context, Result, bail};
use std::{collections::HashMap, path::PathBuf, time::Instant};

struct Args {
    command: String,
    values: HashMap<String, String>,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut it = std::env::args().skip(1);
        let command = it
            .next()
            .context("usage: moon-search <frames|search|refine|verify> [--key value]...")?;
        let mut values = HashMap::new();
        while let Some(k) = it.next() {
            let Some(key) = k.strip_prefix("--") else {
                bail!("unexpected argument {k}")
            };
            values.insert(
                key.to_string(),
                it.next()
                    .with_context(|| format!("--{key} needs a value"))?,
            );
        }
        Ok(Args { command, values })
    }
    fn path(&self, key: &str, default: &str) -> PathBuf {
        PathBuf::from(self.values.get(key).map(String::as_str).unwrap_or(default))
    }
    fn num<T: std::str::FromStr>(&self, key: &str, default: T) -> Result<T> {
        match self.values.get(key) {
            Some(v) => v
                .parse()
                .map_err(|_| anyhow::anyhow!("--{key}: bad value {v}")),
            None => Ok(default),
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse()?;
    let tiles = args.path("tiles", "cache/moon/tiles");
    let queries_dir = args.path("queries", "cache/moon/queries");
    let search_dir = args.path("search", "cache/moon/search");
    let rotations: u32 = args.num("rotations", 8)?;
    // Extra TC series directories (comma separated), e.g. cache/moon/tiles-stereo-0,cache/moon/tiles-sp-0
    let extra_tc: Vec<PathBuf> = args
        .values
        .get("extra-tiles")
        .map(|v| {
            v.split(',')
                .filter(|p| !p.is_empty())
                .map(PathBuf::from)
                .collect()
        })
        .unwrap_or_default();
    let shard_budget = (args.num("shard-gb", 14.0f64)? * 1e9) as usize;
    // TC ladder steps to search, "lo-hi" (e.g. a later run of only the finest step).
    let tc_steps = match args.values.get("tc-steps") {
        Some(v) => {
            let (lo, hi) = v.split_once('-').context("--tc-steps lo-hi")?;
            lo.parse()?..=hi.parse()?
        }
        None => ladder::TC_STEPS,
    };
    let t = Instant::now();
    let load_world = || -> Result<world::World> {
        let w = world::World::load(
            &tiles,
            &extra_tc,
            args.num("tc", 1)? == 1,
            args.num("wac", 1)? == 1,
        )?;
        println!(
            "world: {} rasters, {} levels, {:.2} GB in {:.1} s",
            w.rasters.len(),
            w.levels.len(),
            w.total_bytes as f64 / 1e9,
            t.elapsed().as_secs_f64()
        );
        Ok(w)
    };
    match args.command.as_str() {
        "frames" => frames::extract(&args.path("input", "assets/badapple.mp4"), &queries_dir)?,
        "count" => {
            let mut world = load_world()?;
            world.fill(&world.whole());
            let jobs = ladder::jobs(&world, tc_steps);
            use rayon::prelude::*;
            let sample: Vec<_> = jobs.iter().step_by(args.num("every", 50)?).collect();
            let n: usize = sample
                .par_iter()
                .map(|j| ladder::poses(&world, j, rotations).len())
                .sum();
            println!(
                "{} jobs; ~{:.3e} poses (sampled every {})",
                jobs.len(),
                n as f64 * jobs.len() as f64 / sample.len() as f64,
                args.num("every", 50)?
            );
        }
        "search" => {
            let mut world = load_world()?;
            let queries = frames::load(&queries_dir)?;
            let limit = args
                .values
                .get("limit-jobs")
                .map(|v| v.parse())
                .transpose()?;
            search::run(
                &mut world,
                &queries,
                &search_dir,
                args.num("chunk", 65536)?,
                rotations,
                tc_steps,
                limit,
                shard_budget,
            )?;
        }
        "refine" => {
            let mut world = load_world()?;
            let queries = frames::load(&queries_dir)?;
            // Series directories whose rasters stay loaded (list indices depend on them) but are never chosen.
            let mut drop = std::collections::HashSet::new();
            for dir in args
                .values
                .get("drop-tiles")
                .map(|v| v.split(',').filter(|p| !p.is_empty()).collect::<Vec<_>>())
                .unwrap_or_default()
            {
                let index = std::fs::read_to_string(PathBuf::from(dir).join("tc_index.jsonl"))
                    .with_context(|| dir.to_string())?;
                for line in index.lines().filter(|l| !l.trim().is_empty()) {
                    drop.insert(
                        serde_json::from_str::<serde_json::Value>(line)?["id"]
                            .as_str()
                            .context("id")?
                            .to_string(),
                    );
                }
            }
            refine::run(
                &mut world,
                &queries,
                &search_dir,
                &args.path("out", "output/moon/matches.json"),
                shard_budget,
                &drop,
            )?;
        }
        "verify" => {
            let mut world = load_world()?;
            world.fill(&world.whole());
            let queries = frames::load(&queries_dir)?;
            verify::run(&world, &queries, args.num("jobs", 24)?, rotations)?;
        }
        other => bail!("unknown command {other}"),
    }
    Ok(())
}
