import json, time
from pathlib import Path
import numpy as np
from badapple_coastline.cuda_backend import CudaAtlas
from badapple_coastline.pipeline import read_records
from badapple_coastline.native_rle import sample as cpu_sample

rng = np.random.default_rng(75213)
bank = rng.integers(0, 17, (1024, 192), dtype=np.int8)
queries = np.vstack([bank[[17, 333, 721, 999]], np.zeros((1, 192), np.int8)])
valid = np.ones(1024, np.uint8)
valid[333] = 0
with CudaAtlas(load_globe=False) as cuda:
    cuda.queries(queries)
    cuda.start_region(4)
    cuda.rank(bank[:512], valid[:512], 0)
    cuda.rank(bank[512:], valid[512:], 512)
    scores, indices = cuda.region_results()
    b = bank.astype(np.float64)
    b -= b.mean(axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    q = queries[:4].astype(np.float64)
    q -= q.mean(axis=1, keepdims=True)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    expected = q @ b.T
    expected[:, valid == 0] = -np.inf
    order = np.argsort(-expected, axis=1, kind="stable")[:, :4]
    assert np.array_equal(indices[:4], order), (indices, order)
    assert np.max(np.abs(scores[:4] - np.take_along_axis(expected, order, axis=1))) < 2e-6
    assert (indices[4] == -1).all()
print("CUDA INT8 descriptor ranking matches float64 reference", flush=True)

metadata = json.loads(Path("cache/coastline-atlas-fine/atlas.json").read_text())
records = read_records(Path("output/coastline-wide-rust-landsat/frames.jsonl"))
geometry = np.array(
    [[r[k] for k in ["longitude", "latitude", "width_km", "rotation_clockwise"]] for r in records], np.float64
)
with CudaAtlas() as cuda:
    chosen = geometry[np.arange(0, len(geometry), 107)]
    start = time.perf_counter()
    gpu = cuda.sample(chosen)
    gpu_seconds = time.perf_counter() - start
    start = time.perf_counter()
    cpu = np.stack([cpu_sample(metadata, *r, (260, 195)) for r in chosen])
    cpu_seconds = time.perf_counter() - start
    differences = int(np.count_nonzero(gpu != cpu))
    assert differences == 0, f"{differences} native pixels differ"
    full = geometry[np.arange(0, len(geometry), 503)]
    start = time.perf_counter()
    gpu_full = cuda.sample(full, (1444, 1080))
    full_gpu_seconds = time.perf_counter() - start
    start = time.perf_counter()
    cpu_full = np.stack([cpu_sample(metadata, *r, (1444, 1080)) for r in full])
    full_cpu_seconds = time.perf_counter() - start
    full_differences = int(np.count_nonzero(gpu_full != cpu_full))
    assert full_differences == 0, f"{full_differences} full-resolution pixels differ"
    test = geometry[::3][:2048]
    start = time.perf_counter()
    features, flags = cuda.features(test)
    feature_seconds = time.perf_counter() - start
    assert flags.all()
    features_test, _ = cuda.features(chosen[:4], aspect=0)
    supersampled = cuda.sample(chosen[:4], (256, 192))
    expected = supersampled.reshape(4, 48, 4, 64, 4).sum(axis=(2, 4)).reshape(4, -1)
    assert np.array_equal(features_test, expected)
    report = dict(
        native_comparison_pixels=int(cpu.size),
        native_differences=differences,
        gpu_seconds=gpu_seconds,
        cpu_seconds=cpu_seconds,
        full_comparison_pixels=int(cpu_full.size),
        full_differences=full_differences,
        full_gpu_seconds=full_gpu_seconds,
        full_cpu_seconds=full_cpu_seconds,
        feature_windows=len(test),
        feature_native_samples=int(len(test) * 64 * 48 * 16),
        feature_seconds=feature_seconds,
        estimated_24576000_window_feature_seconds=feature_seconds / len(test) * 24576000,
        library=cuda.build,
    )
Path("output/coastline-cuda-native-validation.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
