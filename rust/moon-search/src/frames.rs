//! Video frames -> exact area-averaged 32x24 / 64x48 queries, with exact-duplicate removal.
use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs,
    hash::{DefaultHasher, Hash, Hasher},
    io::{BufWriter, Read, Write},
    path::Path,
    process::{Command, Stdio},
};

#[derive(Serialize, Deserialize, Clone)]
pub struct Unique {
    pub first_frame: usize,
    pub frames: usize,
    pub mean: f32,
    pub sd: f32,
}

#[derive(Serialize, Deserialize)]
pub struct FrameSet {
    pub input: String,
    pub width: usize,
    pub height: usize,
    pub rate: String,
    pub frames: Vec<u32>,
    pub uniques: Vec<Unique>,
}

/// Exact box average of a w x h u8 image onto ow x oh cells (fractional pixel coverage).
pub fn area(src: &[u8], w: usize, h: usize, ow: usize, oh: usize) -> Vec<f32> {
    fn weights(n: usize, on: usize) -> Vec<Vec<(usize, f32)>> {
        (0..on)
            .map(|o| {
                let (a, b) = (
                    o as f64 * n as f64 / on as f64,
                    (o + 1) as f64 * n as f64 / on as f64,
                );
                let mut v = Vec::new();
                let mut i = a.floor() as usize;
                while (i as f64) < b && i < n {
                    let cover = (b.min(i as f64 + 1.0) - a.max(i as f64)) / (b - a);
                    if cover > 0.0 {
                        v.push((i, cover as f32));
                    }
                    i += 1;
                }
                v
            })
            .collect()
    }
    let (wx, wy) = (weights(w, ow), weights(h, oh));
    let mut rows = vec![0f32; h * ow];
    for y in 0..h {
        for (o, ws) in wx.iter().enumerate() {
            rows[y * ow + o] = ws.iter().map(|&(i, c)| src[y * w + i] as f32 * c).sum();
        }
    }
    let mut out = vec![0f32; ow * oh];
    for (o, ws) in wy.iter().enumerate() {
        for x in 0..ow {
            out[o * ow + x] = ws.iter().map(|&(i, c)| rows[i * ow + x] * c).sum();
        }
    }
    out
}

fn probe(input: &Path) -> Result<(usize, usize, String)> {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate",
            "-of",
            "csv=p=0",
        ])
        .arg(input)
        .output()
        .context("ffprobe")?;
    let text = String::from_utf8_lossy(&out.stdout);
    let parts: Vec<&str> = text.trim().split(',').collect();
    ensure!(parts.len() >= 3, "ffprobe output: {text}");
    Ok((parts[0].parse()?, parts[1].parse()?, parts[2].to_string()))
}

fn write_f32(path: &Path, values: &[f32]) -> Result<()> {
    let mut f = BufWriter::new(fs::File::create(path)?);
    for v in values {
        f.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

pub fn read_f32(path: &Path) -> Result<Vec<f32>> {
    let bytes = fs::read(path).with_context(|| path.display().to_string())?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

pub fn extract(input: &Path, out: &Path) -> Result<()> {
    fs::create_dir_all(out)?;
    let (w, h, rate) = probe(input)?;
    let mut child = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(input)
        .args(["-f", "rawvideo", "-pix_fmt", "gray", "-"])
        .stdout(Stdio::piped())
        .spawn()
        .context("ffmpeg")?;
    let mut stdout = child.stdout.take().unwrap();
    let mut frame = vec![0u8; w * h];
    let (mut q32, mut q64) = (Vec::new(), Vec::new());
    let mut seen: HashMap<u64, u32> = HashMap::new();
    let mut set = FrameSet {
        input: input.display().to_string(),
        width: w,
        height: h,
        rate,
        frames: Vec::new(),
        uniques: Vec::new(),
    };
    loop {
        match stdout.read_exact(&mut frame) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => bail!(e),
        }
        let mut hasher = DefaultHasher::new();
        frame.hash(&mut hasher);
        let key = hasher.finish();
        let index = set.frames.len();
        let u = *seen.entry(key).or_insert_with(|| {
            let small = area(&frame, w, h, 32, 24);
            let large = area(&frame, w, h, 64, 48);
            let mean = large.iter().sum::<f32>() / large.len() as f32;
            let sd = (large.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>()
                / large.len() as f32)
                .sqrt();
            q32.extend(small.iter().map(|v| v / 255.0));
            q64.extend(large.iter().map(|v| v / 255.0));
            set.uniques.push(Unique {
                first_frame: index,
                frames: 0,
                mean,
                sd,
            });
            (set.uniques.len() - 1) as u32
        });
        set.uniques[u as usize].frames += 1;
        set.frames.push(u);
    }
    ensure!(child.wait()?.success(), "ffmpeg failed");
    write_f32(&out.join("q32.f32"), &q32)?;
    write_f32(&out.join("q64.f32"), &q64)?;
    fs::write(out.join("frames.json"), serde_json::to_vec(&set)?)?;
    println!(
        "{} frames, {} unique ({}x{} @ {})",
        set.frames.len(),
        set.uniques.len(),
        w,
        h,
        set.rate
    );
    Ok(())
}

/// Uniform frames (blank black/white) cannot be matched by NCC.
pub const UNIFORM_SD: f32 = 2.0;

pub struct Queries {
    pub set: FrameSet,
    /// GEMM column j = 2 * k + mirror, for textured unique k
    pub textured: Vec<u32>,
    pub q32: Vec<f32>,
    pub q64: Vec<f32>,
}

fn normalise(v: &mut [f32]) {
    let n = v.len() as f32;
    let mean = v.iter().sum::<f32>() / n;
    let norm = v
        .iter()
        .map(|x| (x - mean) * (x - mean))
        .sum::<f32>()
        .sqrt();
    for x in v.iter_mut() {
        *x = (*x - mean) / norm.max(1e-12);
    }
}

fn mirrored(v: &[f32], gw: usize) -> Vec<f32> {
    let mut out = v.to_vec();
    for row in out.chunks_mut(gw) {
        row.reverse();
    }
    out
}

pub fn load(dir: &Path) -> Result<Queries> {
    let set: FrameSet = serde_json::from_slice(&fs::read(dir.join("frames.json"))?)?;
    let (raw32, raw64) = (
        read_f32(&dir.join("q32.f32"))?,
        read_f32(&dir.join("q64.f32"))?,
    );
    let textured: Vec<u32> = (0..set.uniques.len() as u32)
        .filter(|&u| set.uniques[u as usize].sd >= UNIFORM_SD)
        .collect();
    let (mut q32, mut q64) = (Vec::new(), Vec::new());
    for &u in &textured {
        let a = &raw32[u as usize * 768..(u as usize + 1) * 768];
        let b = &raw64[u as usize * 3072..(u as usize + 1) * 3072];
        for m in 0..2 {
            let mut x = if m == 0 { a.to_vec() } else { mirrored(a, 32) };
            let mut y = if m == 0 { b.to_vec() } else { mirrored(b, 64) };
            normalise(&mut x);
            normalise(&mut y);
            q32.extend(x);
            q64.extend(y);
        }
    }
    Ok(Queries {
        set,
        textured,
        q32,
        q64,
    })
}
