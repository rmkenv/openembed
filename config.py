"""
config.py — Central configuration for NAIP Similarity Search
All tuneable constants live here; override via environment variables or .env file.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR  = Path(__file__).parent
_default_cache = ROOT_DIR / "cache"
try:
    _default_cache.mkdir(exist_ok=True)
    # Verify it's actually writable
    (_default_cache / ".write_test").touch()
    (_default_cache / ".write_test").unlink()
    CACHE_DIR = _default_cache
except OSError:
    # Streamlit Cloud mounts the repo read-only; use /tmp instead
    CACHE_DIR = Path("/tmp/naip_sim_cache")
    CACHE_DIR.mkdir(exist_ok=True)

# ── Planetary Computer ────────────────────────────────────────────────────────
PC_STAC_URL    = "https://planetarycomputer.microsoft.com/api/stac/v1"
NAIP_COLLECTION = "naip"
MAX_SCENES      = 5          # max scenes returned from STAC search
OVERVIEW_LEVELS = [2, 1, 0]  # try coarsest first, fallback to full-res

# ── Embedding model ───────────────────────────────────────────────────────────
EMBED_DIM   = 2048           # ResNet-50 penultimate layer
BATCH_SIZE  = 32
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Chipping ──────────────────────────────────────────────────────────────────
DEFAULT_CHIP_SIZE   = 224    # pixels
DEFAULT_STRIDE_FRAC = 0.5
MAX_CHIPS           = 2000   # safety cap — warn user if exceeded

# ── FAISS ─────────────────────────────────────────────────────────────────────
DEFAULT_TOP_K  = 8
FAISS_INDEX_SUFFIX = "_faiss.index"
EMBED_NPY_SUFFIX   = "_embeddings.npy"
META_PKL_SUFFIX    = "_meta.pkl"

# ── UI ────────────────────────────────────────────────────────────────────────
APP_TITLE   = "NAIP · Embeddings Similarity Search"
APP_ICON    = "🛰️"
ESRI_TILES  = (
    "https://server.arcgisonline.com/ArcGIS/rest/services"
    "/World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
ESRI_ATTR   = "Esri, Maxar, Earthstar Geographics"

# Default AOI — DC/PG County area
DEFAULT_BBOX = dict(west=-77.05, south=38.88, east=-76.98, north=38.93)
NAIP_YEARS   = [2023, 2022, 2021, 2020, 2019, 2018, 2017]
