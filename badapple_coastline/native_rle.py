"""Exact original 10 m pixels from a lossless run-length cache built in Rust."""

import ctypes
import json
import os
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "cache" / "coastline-native-rle" / "manifest.json"


def build_cache(atlas: Path, output: Path):
    import rasterio

    metadata = json.loads((atlas / "atlas.json").read_text(encoding="utf-8"))
    if not metadata.get("global_coverage"):
        raise ValueError("Native RLE preparation requires the verified global source tiles")
    identity = {(tile["file"], tile["sha256"]) for tile in metadata["tiles"]}
    output.mkdir(parents=True, exist_ok=True)
    source_manifest = output / "source-manifest.json"
    for previous in [source_manifest, output / "manifest.json"]:
        if previous.is_file():
            old = json.loads(previous.read_text(encoding="utf-8"))
            if identity != {(tile["file"], tile["sha256"]) for tile in old["tiles"]}:
                raise ValueError("Native RLE cache belongs to different source data; choose a new output directory")
    source_manifest.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    binary = (
        ROOT
        / "rust"
        / "coastline-native"
        / "target"
        / "release"
        / ("coastline-native-build.exe" if os.name == "nt" else "coastline-native-build")
    )
    if not binary.is_file():
        raise FileNotFoundError(
            "Build Rust native cache first: cargo build --release --locked --manifest-path rust/coastline-native/Cargo.toml"
        )
    env = os.environ.copy()
    if os.name == "nt":
        libraries = Path(rasterio.__file__).resolve().parent.parent / "rasterio.libs"
        tiff = next(libraries.glob("tiff-*.dll"))
        env["PATH"] = str(libraries) + os.pathsep + env.get("PATH", "")
    else:
        from ctypes.util import find_library

        tiff = find_library("tiff")
        if not tiff:
            raise FileNotFoundError("libtiff is required for lossless native mask conversion")
    subprocess.run([str(binary), str(atlas / "atlas.json"), str(output), str(tiff)], env=env, check=True)
    return {"tiles": len(metadata["tiles"]), "manifest": str((output / "manifest.json").resolve()), "lossless": True}


@lru_cache(maxsize=2)
def load(path, modified_ns):
    metadata = json.loads(Path(path).read_text(encoding="utf-8"))
    library_path = (
        ROOT
        / "rust"
        / "coastline-native"
        / "target"
        / "release"
        / ("coastline_native.dll" if os.name == "nt" else "libcoastline_native.so")
    )
    if not library_path.is_file():
        return None
    library = ctypes.CDLL(str(library_path))
    library.coastline_native_open.argtypes = [ctypes.c_char_p]
    library.coastline_native_open.restype = ctypes.c_void_p
    library.coastline_native_sample.argtypes = (
        [ctypes.c_void_p] + [ctypes.c_double] * 4 + [ctypes.c_size_t] * 2 + [ctypes.POINTER(ctypes.c_uint8)]
    )
    library.coastline_native_sample.restype = ctypes.c_int
    handle = library.coastline_native_open(str(path).encode("utf-8"))
    if not handle:
        raise ValueError("Invalid or incomplete native RLE cache")
    identities = {(tile["file"], tile["sha256"]) for tile in metadata["tiles"]}
    return library, handle, identities


def sample(metadata, lon, lat, width, angle, size):
    if not MANIFEST.is_file():
        return None
    loaded = load(str(MANIFEST), MANIFEST.stat().st_mtime_ns)
    if loaded is None:
        return None
    library, handle, identities = loaded
    if identities != {(tile["file"], tile["sha256"]) for tile in metadata["tiles"]}:
        return None
    output = np.empty((size[1], size[0]), np.uint8)
    status = library.coastline_native_sample(
        handle, lon, lat, width, angle, *size, output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    )
    if status:
        raise RuntimeError(f"Rust native pixel sampling failed: {status}")
    if (output == 255).any():
        raise ValueError("Native mask window includes unacquired source coverage")
    return output
