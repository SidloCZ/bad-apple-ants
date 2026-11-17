//! FFI to cuda/moon/moon.cu (built by cuda/moon/build.ps1).
use crate::world::{Entry, Level, Pose, Shard, World};
use anyhow::{Context, Result, bail, ensure};
use libloading::Library;
use std::{
    ffi::{CStr, c_char, c_void},
    fs,
    path::{Path, PathBuf},
};

type Create = unsafe extern "C" fn(
    *const u8,
    u64,
    *const Level,
    i32,
    u32,
    *const f32,
    *const f32,
    i32,
    i32,
) -> *mut c_void;
type Search = unsafe extern "C" fn(*mut c_void, *const Pose, i32) -> i32;
type Results = unsafe extern "C" fn(*mut c_void, *mut Entry) -> i32;
type SetResults = unsafe extern "C" fn(*mut c_void, *const Entry) -> i32;
type Sample =
    unsafe extern "C" fn(*mut c_void, *const Pose, i32, i32, *mut f32, *mut u8, *mut f32) -> i32;
type Score = unsafe extern "C" fn(*mut c_void, *const Pose, *const i32, i32, i32, *mut f32) -> i32;
type Size = unsafe extern "C" fn() -> i32;

const SCRATCH: usize = 131072;

pub struct Gpu {
    lib: Library,
    ctx: *mut c_void,
    pub cols: usize,
    pub topk: usize,
}

fn dll() -> Result<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../cuda/moon/build/current.json");
    let text = fs::read_to_string(&root)
        .with_context(|| format!("{}; run cuda/moon/build.ps1", root.display()))?;
    let json: serde_json::Value = serde_json::from_str(text.trim_start_matches('\u{feff}'))?;
    Ok(PathBuf::from(json["library"].as_str().context("library")?))
}

impl Gpu {
    pub fn topk_size() -> Result<usize> {
        unsafe {
            let lib = Library::new(dll()?).context("load moon CUDA DLL")?;
            Ok(lib.get::<Size>(b"mb_topk\0")?() as usize)
        }
    }
    /// A context holding one shard of the world (see World::shards); poses keep global level ids.
    pub fn new(
        world: &World,
        shard: &Shard,
        q32: &[f32],
        q64: &[f32],
        nq: usize,
        chunk: usize,
    ) -> Result<Self> {
        let levels: Vec<Level> = world.levels[shard.levels.clone()]
            .iter()
            .map(|l| Level {
                offset: l.offset - shard.bytes.start as u64,
                ..*l
            })
            .collect();
        ensure!(
            q32.len() == nq * 768 && q64.len() == nq * 3072,
            "query buffers do not hold {nq} queries"
        );
        ensure!(
            world.data_base <= shard.bytes.start
                && shard.bytes.end <= world.data_base + world.data.len(),
            "shard pixels are not loaded (World::fill)"
        );
        let data =
            &world.data[shard.bytes.start - world.data_base..shard.bytes.end - world.data_base];
        unsafe {
            let lib = Library::new(dll()?)
                .context("load moon CUDA DLL (CUDA bin/x64 must be on PATH)")?;
            let size = |name: &[u8]| -> Result<i32> { Ok(lib.get::<Size>(name)?()) };
            ensure!(
                size(b"mb_pose_size\0")? as usize == size_of::<Pose>(),
                "Pose layout"
            );
            ensure!(
                size(b"mb_level_size\0")? as usize == size_of::<Level>(),
                "Level layout"
            );
            ensure!(size(b"mb_dims\0")? == 768, "dims");
            let topk = size(b"mb_topk\0")? as usize;
            let create = lib.get::<Create>(b"mb_create\0")?;
            let ctx = create(
                data.as_ptr(),
                data.len() as u64,
                levels.as_ptr(),
                levels.len() as i32,
                shard.levels.start as u32,
                q32.as_ptr(),
                q64.as_ptr(),
                nq as i32,
                chunk as i32,
            );
            let gpu = Gpu {
                lib,
                ctx,
                cols: nq + 2,
                topk,
            };
            if ctx.is_null() {
                gpu.fail()?;
            }
            Ok(gpu)
        }
    }
    fn fail(&self) -> Result<()> {
        unsafe {
            let err = self
                .lib
                .get::<unsafe extern "C" fn() -> *const c_char>(b"mb_error\0")?;
            bail!("CUDA: {}", CStr::from_ptr(err()).to_string_lossy())
        }
    }
    fn ok(&self, code: i32) -> Result<()> {
        if code != 0 { self.fail() } else { Ok(()) }
    }
    pub fn search(&self, poses: &[Pose]) -> Result<()> {
        unsafe {
            self.ok(self.lib.get::<Search>(b"mb_search\0")?(
                self.ctx,
                poses.as_ptr(),
                poses.len() as i32,
            ))
        }
    }
    pub fn results(&self) -> Result<Vec<Entry>> {
        let mut out = vec![Entry::default(); self.cols * self.topk];
        unsafe {
            self.ok(self.lib.get::<Results>(b"mb_results\0")?(
                self.ctx,
                out.as_mut_ptr(),
            ))?
        };
        Ok(out)
    }
    pub fn set_results(&self, lists: &[Entry]) -> Result<()> {
        ensure!(lists.len() == self.cols * self.topk, "checkpoint shape");
        unsafe {
            self.ok(self.lib.get::<SetResults>(b"mb_set_results\0")?(
                self.ctx,
                lists.as_ptr(),
            ))
        }
    }
    /// Normalised patches (n x gw*gw*3/4), flags (bit0 in data, bit1 textured), (mean, sd) stats.
    pub fn sample(&self, poses: &[Pose], gw: usize) -> Result<(Vec<f32>, Vec<u8>, Vec<f32>)> {
        let dim = gw * gw * 3 / 4;
        let (mut out, mut valid, mut stats) = (
            vec![0f32; poses.len() * dim],
            vec![0u8; poses.len()],
            vec![0f32; poses.len() * 2],
        );
        let f = unsafe { self.lib.get::<Sample>(b"mb_sample\0")? };
        for (k, part) in poses.chunks(SCRATCH).enumerate() {
            let o = k * SCRATCH;
            unsafe {
                self.ok(f(
                    self.ctx,
                    part.as_ptr(),
                    part.len() as i32,
                    gw as i32,
                    out[o * dim..].as_mut_ptr(),
                    valid[o..].as_mut_ptr(),
                    stats[o * 2..].as_mut_ptr(),
                ))?;
            }
        }
        Ok((out, valid, stats))
    }
    /// fp32 NCC of each pose against GEMM column qidx[i]; -2 for invalid or flat poses.
    pub fn score(&self, poses: &[Pose], qidx: &[i32], gw: usize) -> Result<Vec<f32>> {
        ensure!(qidx.len() == poses.len(), "one query index per pose");
        let mut out = vec![0f32; poses.len()];
        let f = unsafe { self.lib.get::<Score>(b"mb_score\0")? };
        for (k, (part, qs)) in poses.chunks(SCRATCH).zip(qidx.chunks(SCRATCH)).enumerate() {
            unsafe {
                self.ok(f(
                    self.ctx,
                    part.as_ptr(),
                    qs.as_ptr(),
                    part.len() as i32,
                    gw as i32,
                    out[k * SCRATCH..].as_mut_ptr(),
                ))?;
            }
        }
        Ok(out)
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        unsafe {
            if let Ok(f) = self
                .lib
                .get::<unsafe extern "C" fn(*mut c_void)>(b"mb_destroy\0")
            {
                f(self.ctx);
            }
        }
    }
}
