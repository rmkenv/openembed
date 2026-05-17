"""
utils/imagery.py — NAIP fetch, chipping, and geo-projection helpers
"""
import hashlib
import pickle
from pathlib import Path

import numpy as np
import pystac_client
import planetary_computer
import rioxarray
from shapely.geometry import box

import sys
_ROOT = str(Path(__file__).parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from config import (
    PC_STAC_URL, NAIP_COLLECTION, MAX_SCENES,
    OVERVIEW_LEVELS, DEFAULT_CHIP_SIZE, MAX_CHIPS, CACHE_DIR,
)


# ── STAC search ───────────────────────────────────────────────────────────────

def get_catalog():
    return pystac_client.Client.open(
        PC_STAC_URL,
        modifier=planetary_computer.sign_inplace,
    )


def search_naip_scenes(bbox: list[float], year: int) -> list:
    """Return up to MAX_SCENES STAC items for bbox + year."""
    catalog = get_catalog()
    search = catalog.search(
        collections=[NAIP_COLLECTION],
        bbox=bbox,
        datetime=f"{year}-01-01/{year}-12-31",
        max_items=MAX_SCENES,
    )
    return list(search.items())


def load_naip_scene(item, overview_level: int | None = None) -> tuple:
    """
    Load a NAIP COG as a rioxarray DataArray.
    Tries overview levels from coarsest to finest until data is readable.
    Returns (DataArray, effective_overview_level).
    """
    href = item.assets["image"].href
    levels = [overview_level] if overview_level is not None else OVERVIEW_LEVELS

    for lvl in levels:
        try:
            ds = rioxarray.open_rasterio(href, overview_level=lvl)
            # Sanity check — must have spatial extent
            if ds.shape[1] > 0 and ds.shape[2] > 0:
                return ds, lvl
        except Exception:
            continue

    # Last resort: full resolution
    ds = rioxarray.open_rasterio(href)
    return ds, None


# ── Chipping ──────────────────────────────────────────────────────────────────

def normalize_rgb(arr: np.ndarray) -> np.ndarray:
    """Per-image percentile stretch: (2nd, 98th) → [0, 1]."""
    lo = np.percentile(arr, 2)
    hi = np.percentile(arr, 98)
    arr = np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)
    return arr.astype(np.float32)


def chip_scene(
    ds,
    chip_size: int = DEFAULT_CHIP_SIZE,
    stride: int | None = None,
    max_chips: int = MAX_CHIPS,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """
    Tile a rioxarray DataArray (NAIP RGBI) into chips.

    Returns
    -------
    chips     : np.ndarray  shape (N, 3, chip_size, chip_size) float32 [0,1]
    positions : list of (row, col) pixel offsets for each chip
    """
    if stride is None:
        stride = chip_size // 2

    arr = ds.values[:3].astype(np.float32)   # RGB only
    arr = normalize_rgb(arr)

    _, H, W = arr.shape
    chips, positions = [], []

    for y in range(0, H - chip_size, stride):
        for x in range(0, W - chip_size, stride):
            chip = arr[:, y : y + chip_size, x : x + chip_size]
            if chip.shape == (3, chip_size, chip_size):
                chips.append(chip)
                positions.append((y, x))
            if len(chips) >= max_chips:
                break
        if len(chips) >= max_chips:
            break

    return np.stack(chips), positions


# ── Geo utilities ─────────────────────────────────────────────────────────────

def pixel_to_bbox(pos: tuple[int, int], ds, chip_size: int):
    """Convert (row, col) chip origin to a shapely box in the scene CRS."""
    tf = ds.rio.transform()
    row, col = pos
    left  = tf.c + col * tf.a
    top   = tf.f + row * tf.e
    right = left + chip_size * tf.a
    bot   = top  + chip_size * tf.e
    return box(left, bot, right, top)


def build_chip_geodataframe(positions, ds, chip_size, crs="EPSG:4326"):
    """Build a GeoDataFrame with one row per chip, reprojected to crs."""
    import geopandas as gpd
    scene_crs = ds.rio.crs or "EPSG:4326"
    geoms = [pixel_to_bbox(p, ds, chip_size) for p in positions]
    gdf = gpd.GeoDataFrame(
        {"chip_id": range(len(geoms)), "pixel_row": [p[0] for p in positions],
         "pixel_col": [p[1] for p in positions]},
        geometry=geoms,
        crs=scene_crs,
    )
    if str(scene_crs) != crs:
        gdf = gdf.to_crs(crs)
    return gdf


# ── Disk cache ────────────────────────────────────────────────────────────────

def _cache_key(scene_id: str, chip_size: int, stride: int) -> str:
    raw = f"{scene_id}_{chip_size}_{stride}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def cache_path(scene_id: str, chip_size: int, stride: int, suffix: str) -> Path:
    key = _cache_key(scene_id, chip_size, stride)
    return CACHE_DIR / f"{key}{suffix}"


def save_chips(chips: np.ndarray, path: Path):
    np.save(str(path), chips)


def load_chips(path: Path) -> np.ndarray:
    return np.load(str(path))


def save_meta(meta: dict, path: Path):
    with open(path, "wb") as f:
        pickle.dump(meta, f)


def load_meta(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)
