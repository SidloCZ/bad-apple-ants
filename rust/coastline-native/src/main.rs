use anyhow::{Context, Result, ensure};
use coastline_native::{Manifest, TileSpec, transitions};
use libloading::Library;
use rayon::prelude::*;
use std::{
    ffi::{CString, c_char, c_void},
    fs::File,
    io::{BufWriter, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};

type Open = unsafe extern "C" fn(*const c_char, *const c_char) -> *mut c_void;
type Close = unsafe extern "C" fn(*mut c_void);
type Read = unsafe extern "C" fn(*mut c_void, u32, *mut c_void, isize) -> isize;
type Get = unsafe extern "C" fn(*mut c_void, u32, ...) -> i32;
struct TiffApi {
    _lib: Library,
    open: Open,
    close: Close,
    read: Read,
    get: Get,
}
impl TiffApi {
    fn load(path: &Path) -> Result<Self> {
        let lib = unsafe { Library::new(path) }?;
        unsafe {
            Ok(Self {
                open: *lib.get(b"TIFFOpen\0")?,
                close: *lib.get(b"TIFFClose\0")?,
                read: *lib.get(b"TIFFReadEncodedStrip\0")?,
                get: *lib.get(b"TIFFGetFieldDefaulted\0")?,
                _lib: lib,
            })
        }
    }
}
struct Handle<'a> {
    raw: *mut c_void,
    api: &'a TiffApi,
}
impl Drop for Handle<'_> {
    fn drop(&mut self) {
        unsafe { (self.api.close)(self.raw) }
    }
}

fn build(tile: &TileSpec, source: &Path, output: &Path, api: &TiffApi) -> Result<()> {
    let dest = output.join(format!("{}.rle", tile.file));
    if dest.is_file() {
        return Ok(());
    }
    let name = CString::new(
        source
            .join(&tile.file)
            .to_str()
            .context("Non UTF-8 TIFF path")?,
    )?;
    let raw = unsafe { (api.open)(name.as_ptr(), c"r".as_ptr()) };
    ensure!(!raw.is_null(), "Cannot open native TIFF");
    let handle = Handle { raw, api };
    let (mut width, mut height, mut rows) = (0u32, 0u32, 0u32);
    let (mut bits, mut compression, mut photometric, mut fill_order) = (0u16, 0u16, 0u16, 0u16);
    unsafe {
        ensure!(
            (api.get)(raw, 256, &mut width) == 1 && (api.get)(raw, 257, &mut height) == 1,
            "Missing TIFF dimensions"
        );
        (api.get)(raw, 278, &mut rows);
        (api.get)(raw, 258, &mut bits);
        (api.get)(raw, 259, &mut compression);
        (api.get)(raw, 262, &mut photometric);
        (api.get)(raw, 266, &mut fill_order);
    }
    ensure!(
        rows == 1
            && bits == 1
            && compression == 4
            && photometric == 1
            && fill_order == 1
            && width % 8 == 0,
        "Expected one-row, 1-bit, MSB-first CCITT4 original mask"
    );
    let partial = dest.with_extension("rle.partial");
    let mut file = File::create(&partial)?;
    let data_start = 16 + 8 * (height as u64 + 1);
    file.seek(SeekFrom::Start(data_start))?;
    let mut writer = BufWriter::with_capacity(4 * 1024 * 1024, file);
    let mut offsets = Vec::with_capacity(height as usize + 1);
    offsets.push(0u64);
    let mut packed = vec![0u8; width as usize / 8];
    let mut runs = Vec::new();
    let mut encoded = Vec::new();
    for row in 0..height {
        let got =
            unsafe { (api.read)(raw, row, packed.as_mut_ptr().cast(), packed.len() as isize) };
        ensure!(
            got == packed.len() as isize,
            "Incomplete TIFF row {} in {}",
            row,
            tile.file
        );
        transitions(&packed, &mut runs);
        encoded.clear();
        for &run in &runs {
            encoded.extend_from_slice(&run.to_le_bytes());
        }
        writer.write_all(&encoded)?;
        offsets.push(offsets.last().unwrap() + runs.len() as u64);
    }
    writer.flush()?;
    let mut file = writer.into_inner()?;
    file.seek(SeekFrom::Start(0))?;
    let mut header = BufWriter::with_capacity(4 * 1024 * 1024, file);
    header.write_all(b"LWMRLE01")?;
    header.write_all(&width.to_le_bytes())?;
    header.write_all(&height.to_le_bytes())?;
    for offset in offsets {
        header.write_all(&offset.to_le_bytes())?;
    }
    header.flush()?;
    drop(header);
    drop(handle);
    std::fs::rename(partial, dest)?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    ensure!(
        args.len() == 3,
        "Usage: coastline-native-build ATLAS.json OUTPUT_DIRECTORY LIBTIFF.dll"
    );
    let manifest: Manifest = serde_json::from_reader(File::open(&args[0])?)?;
    let output = PathBuf::from(&args[1]);
    std::fs::create_dir_all(&output)?;
    let api = TiffApi::load(Path::new(&args[2]))?;
    rayon::ThreadPoolBuilder::new()
        .num_threads(24)
        .build_global()?;
    let done = AtomicUsize::new(0);
    let start = Instant::now();
    manifest
        .tiles
        .par_iter()
        .try_for_each(|tile| -> Result<()> {
            build(tile, Path::new(&manifest.source_directory), &output, &api)?;
            let count = done.fetch_add(1, Ordering::Relaxed) + 1;
            eprintln!(
                "native RLE: {count}/{} {}, {:.1}s",
                manifest.tiles.len(),
                tile.file,
                start.elapsed().as_secs_f64()
            );
            Ok(())
        })?;
    serde_json::to_writer_pretty(File::create(output.join("manifest.json"))?, &manifest)?;
    eprintln!("native RLE complete: {:.1}s", start.elapsed().as_secs_f64());
    Ok(())
}
