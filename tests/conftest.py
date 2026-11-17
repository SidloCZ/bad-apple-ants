import importlib.util
import sys

# Windows native DLL initialization is order-sensitive when GDAL and FAISS coexist:
# load the optional FAISS backend before any test initializes GDAL through rasterio.
if sys.platform == "win32" and importlib.util.find_spec("faiss") is not None:
    import faiss  # noqa: F401
