"""Download and preprocess the Hotmaps EU heat density map for Bavaria.

Outputs (saved in this directory):
1. `heat_tot_curr_density_bayern_raw.tif`
   - 100m x 100m raster (EPSG:3035), cropped to Bavaria + 5 km safety buffer.
   - Units: MWh / (ha * a) final energy demand for space heating and DHW
     (residential + non-residential buildings).
2. `heat_tot_curr_density_bayern_smooth3x3.tif`
   - Same grid after applying a 3x3 uniform mean smoothing filter (~300m x 300m
     neighborhood) to reduce raster border artifacts and single-cell noise.
3. `heat_tot_curr_density_bayern_smooth3x3.npz`
   - Pure-NumPy bundle (`data` + affine metadata) so offline LRZ compute nodes
     using the `inference` conda env can load and index it without `rasterio`.

Source dataset (CC-BY-4.0, DOI: 10.5281/zenodo.4687129):
https://gitlab.com/hotmaps/buildings/heat/heat_tot_curr_density
"""

from pathlib import Path
import argparse
import urllib.request

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from scipy.ndimage import uniform_filter


# --------------------------------------------------------------------------- #
# ----------------------------- Configuration ------------------------------- #
# --------------------------------------------------------------------------- #
HOTMAPS_URL = (
    "https://gitlab.com/hotmaps/buildings/heat/heat_tot_curr_density"
    "/-/raw/master/data/heat_tot_curr_density.tif"
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[5]
VG250_GEM_SHP = REPO_ROOT / "Grid_Importing" / "gridreadout" / "input_data" / "VG250_GEM.shp"

# Safety buffer around the Bavarian administrative bounding box (in meters)
# so grids near state borders keep full 3x3 neighborhood context.
BUFFER_M = 5000.0
SMOOTH_KERNEL_SIZE = 3

OUT_RAW_TIF = SCRIPT_DIR / "heat_tot_curr_density_bayern_raw.tif"
OUT_SMOOTH_TIF = SCRIPT_DIR / "heat_tot_curr_density_bayern_smooth3x3.tif"
OUT_SMOOTH_NPZ = SCRIPT_DIR / "heat_tot_curr_density_bayern_smooth3x3.npz"


# --------------------------------------------------------------------------- #
# ----------------------------- Helper Functions ---------------------------- #
# --------------------------------------------------------------------------- #
def get_bavaria_bounds_epsg3035(vg250_path: Path, buffer_m: float = BUFFER_M):
    """Return Bavaria bounding box (minx, miny, maxx, maxy) in EPSG:3035,
    expanded by `buffer_m` and snapped outward to clean 100 m grid multiples.
    """
    gdf = gpd.read_file(vg250_path)

    # Bavarian municipality codes (ARS / AGS) start with state prefix "09"
    code_col = "ARS" if "ARS" in gdf.columns else "AGS"
    bavaria = gdf[gdf[code_col].astype(str).str.startswith("09")]
    if bavaria.empty:
        raise ValueError(f"No Bavarian features (prefix '09') found in {vg250_path}")

    minx, miny, maxx, maxy = bavaria.to_crs("EPSG:3035").total_bounds

    # Expand by buffer and snap to exact 100 m raster grid boundaries
    minx = float(np.floor((minx - buffer_m) / 100.0) * 100.0)
    miny = float(np.floor((miny - buffer_m) / 100.0) * 100.0)
    maxx = float(np.ceil((maxx + buffer_m) / 100.0) * 100.0)
    maxy = float(np.ceil((maxy + buffer_m) / 100.0) * 100.0)
    return minx, miny, maxx, maxy


def download_hotmaps_eu(dest_path: Path):
    """Download the full EU Hotmaps heat density GeoTIFF if not already present."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Hotmaps EU raster to {dest_path} ...")
    urllib.request.urlretrieve(HOTMAPS_URL, dest_path)


def crop_and_smooth(eu_tif_path: Path, bounds_3035):
    """Crop the EU raster to Bavaria and apply a 3x3 uniform mean filter."""
    minx, miny, maxx, maxy = bounds_3035

    with rasterio.open(eu_tif_path) as src:
        # Compute integer pixel window aligned to source 100 m cells
        win = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
        win = win.round_offsets().round_lengths()

        raw_data = src.read(1, window=win).astype(np.float32)
        out_transform = src.window_transform(win)

        # Clean any non-finite or negative values (nodata is 0.0 MWh/ha = no heat demand)
        raw_data = np.where(np.isfinite(raw_data) & (raw_data > 0.0), raw_data, 0.0).astype(np.float32)

        # 3x3 uniform neighborhood mean in MWh/(ha*a); empty cells contribute 0.0
        smooth_data = uniform_filter(
            raw_data, size=SMOOTH_KERNEL_SIZE, mode="constant", cval=0.0
        ).astype(np.float32)

        profile = src.profile.copy()
        profile.update(
            {
                "driver": "GTiff",
                "height": raw_data.shape[0],
                "width": raw_data.shape[1],
                "count": 1,
                "dtype": "float32",
                "crs": src.crs,
                "transform": out_transform,
                "nodata": 0.0,
                "compress": "deflate",
                "zlevel": 9,
                "tiled": True,
                "predictor": 3,
            }
        )

    return raw_data, smooth_data, profile


def save_outputs(raw_data, smooth_data, profile):
    """Write raw GeoTIFF, smoothed GeoTIFF, and pure-NumPy `.npz` lookup bundle."""
    with rasterio.open(OUT_RAW_TIF, "w", **profile) as dst:
        dst.write(raw_data, 1)

    with rasterio.open(OUT_SMOOTH_TIF, "w", **profile) as dst:
        dst.write(smooth_data, 1)

    # Save lightweight pure-NumPy archive for offline LRZ `inference` env (no rasterio needed)
    tf = profile["transform"]
    np.savez_compressed(
        OUT_SMOOTH_NPZ,
        data=smooth_data,
        x_origin=np.float64(tf.c),
        y_origin=np.float64(tf.f),
        res_x=np.float64(tf.a),
        res_y=np.float64(tf.e),
        crs=str(profile["crs"]),
        unit="MWh/(ha*a)",
        smoothing=f"uniform_{SMOOTH_KERNEL_SIZE}x{SMOOTH_KERNEL_SIZE}",
    )


def lookup_heat_density_npz(x_epsg3035, y_epsg3035, npz_path: Path = OUT_SMOOTH_NPZ):
    """Example vectorized offline lookup using pure NumPy (no rasterio required).

    Args:
        x_epsg3035: Scalar or 1D array of EPSG:3035 easting coordinates (meters).
        y_epsg3035: Scalar or 1D array of EPSG:3035 northing coordinates (meters).
        npz_path: Path to `heat_tot_curr_density_bayern_smooth3x3.npz`.

    Returns:
        Heat density in MWh/(ha*a) (float32), 0.0 for out-of-bounds points.
    """
    bundle = np.load(npz_path)
    grid = bundle["data"]
    x0, y0 = float(bundle["x_origin"]), float(bundle["y_origin"])
    rx, ry = float(bundle["res_x"]), float(bundle["res_y"])

    x_arr = np.asarray(x_epsg3035, dtype=np.float64)
    y_arr = np.asarray(y_epsg3035, dtype=np.float64)

    cols = np.floor((x_arr - x0) / rx).astype(np.int64)
    rows = np.floor((y_arr - y0) / ry).astype(np.int64)

    in_bounds = (
        (rows >= 0) & (rows < grid.shape[0]) & (cols >= 0) & (cols < grid.shape[1])
    )
    out = np.zeros(x_arr.shape, dtype=np.float32)
    out[in_bounds] = grid[rows[in_bounds], cols[in_bounds]]
    return out


# --------------------------------------------------------------------------- #
# ---------------------------------- Main ----------------------------------- #
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Crop and smooth Hotmaps EU heat density for Bavaria.")
    parser.add_argument(
        "--eu-tif",
        type=Path,
        default=SCRIPT_DIR / "_tmp_heat_tot_curr_density_eu.tif",
        help="Path to the full EU Hotmaps GeoTIFF (downloaded automatically if missing).",
    )
    parser.add_argument(
        "--keep-eu-tif",
        action="store_true",
        help="Keep the temporary 129 MB EU GeoTIFF after cropping.",
    )
    args = parser.parse_args()

    downloaded_here = False
    if not args.eu_tif.exists():
        download_hotmaps_eu(args.eu_tif)
        downloaded_here = True

    bounds = get_bavaria_bounds_epsg3035(VG250_GEM_SHP, buffer_m=BUFFER_M)
    print(f"Bavaria EPSG:3035 bounds (+{int(BUFFER_M)}m buffer): {bounds}")

    raw_data, smooth_data, profile = crop_and_smooth(args.eu_tif, bounds)
    save_outputs(raw_data, smooth_data, profile)

    if downloaded_here and not args.keep_eu_tif:
        args.eu_tif.unlink()

    print(f"Saved: {OUT_RAW_TIF} ({OUT_RAW_TIF.stat().st_size / 1e6:.2f} MB)")
    print(f"Saved: {OUT_SMOOTH_TIF} ({OUT_SMOOTH_TIF.stat().st_size / 1e6:.2f} MB)")
    print(f"Saved: {OUT_SMOOTH_NPZ} ({OUT_SMOOTH_NPZ.stat().st_size / 1e6:.2f} MB)")
    print(
        f"Raster shape: {raw_data.shape} | "
        f"Raw max: {raw_data.max():.1f} MWh/ha | "
        f"Smooth max: {smooth_data.max():.1f} MWh/ha | "
        f"Total heat (raw vs smooth): {raw_data.sum()/1e6:.2f} vs {smooth_data.sum()/1e6:.2f} TWh/a"
    )


if __name__ == "__main__":
    main()
