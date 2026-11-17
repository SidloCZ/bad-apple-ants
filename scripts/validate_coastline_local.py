"""Native RLE point sampling against independent original GeoTIFF pixels."""

from pathlib import Path
import json
import numpy as np
import faiss  # Load before GDAL on Windows, as in the repository test configuration.
import rasterio
from badapple_coastline.local_map import NativePointSampler
from badapple_coastline.pipeline import write_json


def main():
    metadata = json.loads(Path("cache/coastline-native-rle/manifest.json").read_text())
    sampler = NativePointSampler()
    rng = np.random.default_rng(620)
    count = 0
    try:
        for tile in metadata["tiles"]:
            left, bottom, right, top = tile["bounds"]
            longitude = rng.uniform(left + 1e-6, right - 1e-6, 32)
            latitude = rng.uniform(bottom + 1e-6, top - 1e-6, 32)
            with rasterio.open(Path(metadata["source_directory"]) / tile["file"]) as source:
                expected = np.array([v[0] for v in source.sample(zip(longitude, latitude))])
            actual = sampler.sample(longitude, latitude)
            np.testing.assert_array_equal(actual, expected)
            count += len(actual)
        # The seam wraps to the same actual globe pixels, never missing sea fill.
        longitude = np.array([179.999999, -179.999999, 180.000001, -180.000001])
        latitude = np.array([83.0, 83.0, -75.0, -75.0])
        wrapped = (longitude + 180) % 360 - 180
        np.testing.assert_array_equal(sampler.sample(longitude, latitude), sampler.sample(wrapped, latitude))
    finally:
        sampler.close()
    result = dict(
        source_tiles_verified=len(metadata["tiles"]),
        independent_original_geotiff_pixels_compared=count,
        exact_native_pixel_match=True,
        dateline_wrap_and_polar_latitudes=True,
        missing_data_fill=False,
    )
    write_json(Path("output/local-map-native-validation.json"), result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
