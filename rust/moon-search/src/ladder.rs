//! Patch-width ladder (sqrt(2) steps from 0.15 km), jobs, and exhaustive pose grids.
use crate::world::{Pose, R_KM, Source, World, footprint_ok};
use std::f64::consts::PI;

pub const TC_STEPS: std::ops::RangeInclusive<u32> = 8..=15; // 2.4 .. 27.2 km; `--tc-steps` searches a sub-range (steps finer than an image's base are skipped)
pub const WAC_STEPS: std::ops::RangeInclusive<u32> = 16..=23; // 38.4 .. 434.4 km
pub const TC_MAX_KM: f64 = 27.2;
pub const WAC_MAX_KM: f64 = 434.5;
pub const WAC_MAX_LAT_DEG: f64 = 80.0;
/// Grid stride = width / STRIDE_DIV (2 cells of the 32-wide comparison grid).
pub const STRIDE_DIV: f64 = 16.0;
const WAC_ROWS_PER_JOB: i64 = 24;

pub fn width_km(step: u32) -> f64 {
    0.15 * 2f64.powf(step as f64 / 2.0)
}

#[derive(Clone, Copy, Debug)]
pub struct Job {
    pub raster: u32,
    pub step: u32,
    pub part: i64,
}

pub fn jobs(world: &World, tc_steps: std::ops::RangeInclusive<u32>) -> Vec<Job> {
    let mut out = Vec::new();
    for (ri, r) in world.rasters.iter().enumerate() {
        match r.source {
            Source::Tc => {
                for step in tc_steps.clone() {
                    out.push(Job {
                        raster: ri as u32,
                        step,
                        part: 0,
                    });
                }
            }
            Source::Wac => {
                for step in WAC_STEPS {
                    let d = width_km(step) / STRIDE_DIV;
                    let rows = (WAC_MAX_LAT_DEG.to_radians() * R_KM / d).floor() as i64;
                    let mut start = -rows;
                    while start <= rows {
                        out.push(Job {
                            raster: ri as u32,
                            step,
                            part: start,
                        });
                        start += WAC_ROWS_PER_JOB;
                    }
                }
            }
        }
    }
    out
}

pub fn poses(world: &World, job: &Job, rotations: u32) -> Vec<Pose> {
    let w = width_km(job.step);
    let d = w / STRIDE_DIV;
    let Some(level) = world.pick_level(job.raster as usize, w / 32.0, true) else {
        return Vec::new();
    };
    let r = &world.rasters[job.raster as usize];
    let mut out = Vec::new();
    match r.source {
        Source::Tc => {
            let [x0, x1, y0, y1] = r.bounds;
            let (nx, ny) = (
                ((x1 - x0) / d).floor() as i64,
                ((y1 - y0) / d).floor() as i64,
            );
            for iy in 0..=ny {
                for ix in 0..=nx {
                    let (cx, cy) = (x0 + ix as f64 * d, y0 + iy as f64 * d);
                    for k in 0..rotations {
                        let pose = Pose {
                            level,
                            a: cx as f32,
                            b: cy as f32,
                            w: w as f32,
                            theta: (k as f64 * 2.0 * PI / rotations as f64) as f32,
                            flags: (k == 0 && ix % 4 == 0 && iy % 4 == 0) as u32,
                        };
                        if footprint_ok(world, &pose) {
                            out.push(pose);
                        }
                    }
                }
            }
        }
        Source::Wac => {
            let rows = (WAC_MAX_LAT_DEG.to_radians() * R_KM / d).floor() as i64;
            for i in job.part..(job.part + WAC_ROWS_PER_JOB).min(rows + 1) {
                let lat = i as f64 * d / R_KM;
                let n_lon = ((2.0 * PI * R_KM * lat.cos()) / d).floor().max(1.0) as i64;
                let dl = 2.0 * PI / n_lon as f64;
                for j in 0..n_lon {
                    let lon = -PI + (j as f64 + 0.5) * dl;
                    for k in 0..rotations {
                        out.push(Pose {
                            level,
                            a: lon as f32,
                            b: lat as f32,
                            w: w as f32,
                            theta: (k as f64 * 2.0 * PI / rotations as f64) as f32,
                            flags: (k == 0 && i.rem_euclid(4) == 0 && j % 4 == 0) as u32,
                        });
                    }
                }
            }
        }
    }
    out
}
