"""Multiprocessing worker and orchestrator for exporting pylovo grids.

Grouping tasks by 1 km Zensus cell allows each cell's PVGIS weather to be fetched
once, while spawning isolated worker processes keeps HDF5/PyTables C state,
SQLAlchemy pools, and HTTP sessions completely independent and thread-safe.
"""

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import multiprocessing
import os
import shutil
import tempfile
import time
import pandas as pd
import requests

from config import config
import src.db_read as dbrd
import src.grid_topol as grdtpl
import src.save_grid as svgrd
import src.weather as wth

_WORKER_DB = None
_WORKER_SESSION = None
_WORKER_STORAGE_DIR = None
_WORKER_STAGING_DIR = None

_TEMP_REGION_COLS = ["x_census", "y_census", "census_lat", "census_lon"]


def _init_worker(storage_dir, staging_root):
    """Initialize per-process DB pool, HTTP keep-alive session, and local staging dir."""
    global _WORKER_DB, _WORKER_SESSION, _WORKER_STORAGE_DIR, _WORKER_STAGING_DIR
    # Prevent HDF5 file-locking failures when moving files onto sshfs/FUSE mounts
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    _WORKER_DB = dbrd.DataBase()
    _WORKER_SESSION = requests.Session()
    _WORKER_STORAGE_DIR = storage_dir
    _WORKER_STAGING_DIR = os.path.join(staging_root, f"worker_{os.getpid()}")
    os.makedirs(_WORKER_STAGING_DIR, exist_ok=True)


def export_single_grid(db, region_specs_row, precomputed_weather, target_storage_dir, staging_dir=None, session=None):
    """Read topology/buildings for one grid, attach weather, and write its `.h5` file.

    When `staging_dir` is provided, the HDF5 file is assembled locally in `staging_dir`
    (avoiding slow multi-open FUSE/sshfs metadata traffic), copied once to a `.part`
    file in `target_storage_dir`, atomically renamed, and immediately deleted locally.
    """
    grid_specs = {
        "grid_index": region_specs_row["grid_index"],
        "dso_short": region_specs_row["dso_short"],
        "cell_id": f"N{int(region_specs_row['y'])}E{int(region_specs_row['x'])}",
        "plz": int(region_specs_row["plz_pylovo"]),
        "kcid": int(region_specs_row["kcid"]),
        "bcid": int(region_specs_row["bcid"]),
    }

    net = db.read_single_ppgrid(grid_specs)
    net = grdtpl.assign_min_linelen(net)
    net = grdtpl.remove_duplicate_loads(net)
    df_buildings = db.read_buildings(grid_specs, net.bus)

    if precomputed_weather is None:
        df_weather_raw, altitude = wth.fetch_weather_for_location(
            region_specs_row["lat"],
            region_specs_row["lon"],
            include_soil_temp=False,
            session=session,
        )
    else:
        df_weather_raw, altitude = precomputed_weather

    # Strip temporary weather-grouping columns so /raw_data/region preserves exact schema
    region_specs = region_specs_row.drop(labels=_TEMP_REGION_COLS, errors="ignore").copy()
    region_specs["altitude"] = altitude

    write_dir = staging_dir or target_storage_dir
    sf = svgrd.SaveFile(grid_specs, storage_dir=write_dir, verbose=False)
    try:
        sf.save_topology(net, "/raw_data/")
        sf.save_df(region_specs, "/raw_data/region")
        sf.save_df(df_buildings, "/raw_data/buildings")
        sf.save_df(df_weather_raw, "/raw_data/weather")

        if staging_dir:
            filename = os.path.basename(sf.path)
            dst_final = os.path.join(target_storage_dir, filename)
            dst_part = os.path.join(target_storage_dir, f"{filename}.part")
            try:
                shutil.copyfile(sf.path, dst_part)
                os.replace(dst_part, dst_final)
            except Exception:
                if os.path.exists(dst_part):
                    os.remove(dst_part)
                raise
    finally:
        if staging_dir and os.path.exists(sf.path):
            os.remove(sf.path)


def export_census_cell(cell_task):
    """Fetch weather once for one 1 km Zensus cell and export all grids in that cell."""
    census_lat, census_lon, row_dicts = cell_task
    errors = []
    exported = 0

    try:
        cell_weather = wth.fetch_weather_for_location(
            census_lat,
            census_lon,
            include_soil_temp=False,
            session=_WORKER_SESSION,
        )
    except Exception as exc:
        for r in row_dicts:
            errors.append((r["grid_index"], r["plz_pylovo"], r["kcid"], r["bcid"], f"Weather error: {exc}"))
        return exported, errors

    for row_dict in row_dicts:
        try:
            export_single_grid(
                _WORKER_DB,
                pd.Series(row_dict),
                cell_weather,
                _WORKER_STORAGE_DIR,
                staging_dir=_WORKER_STAGING_DIR,
                session=_WORKER_SESSION,
            )
            exported += 1
        except Exception as exc:
            errors.append(
                (row_dict["grid_index"], row_dict["plz_pylovo"], row_dict["kcid"], row_dict["bcid"], str(exc))
            )

    return exported, errors


def _iter_cell_tasks(df_remaining_grids):
    """Yield `(census_lat, census_lon, row_dicts)` tasks grouped by 1 km Zensus cell."""
    for _, cell_group in df_remaining_grids.groupby(["x_census", "y_census"], sort=True):
        first_row = cell_group.iloc[0]
        yield (
            float(first_row["census_lat"]),
            float(first_row["census_lon"]),
            cell_group.to_dict("records"),
        )


def run_parallel_export(df_remaining_grids, storage_dir=None, max_workers=8, progress_every_cells=100):
    """Export remaining grids in parallel using `max_workers` isolated processes.

    Returns a dict summary with exported count, elapsed seconds, and any failed grids.
    """
    target_dir = storage_dir or config.STORAGE_DIR
    n_remaining_grids = len(df_remaining_grids)
    if n_remaining_grids == 0:
        print("All grids are already exported.")
        return {"exported": 0, "failed": [], "elapsed_s": 0.0}

    n_remaining_cells = df_remaining_grids[["x_census", "y_census"]].drop_duplicates().shape[0]
    print(f"Target storage directory: {target_dir}")
    print(
        f"Starting parallel export with {max_workers} workers: "
        f"{n_remaining_grids} grids across {n_remaining_cells} unique 1 km Zensus cells"
    )

    t_start = time.perf_counter()
    exported_total = 0
    failed_grids = []
    cells_done = 0

    # On Linux, fork works reliably from Jupyter notebooks and stdin without spawn's
    # __main__ re-import; each worker initializes its own DataBase engine and Session.
    start_method = "fork" if "fork" in multiprocessing.get_all_start_methods() else "spawn"
    ctx = multiprocessing.get_context(start_method)
    task_iter = _iter_cell_tasks(df_remaining_grids)
    max_in_flight = max(max_workers * 4, 1)

    with tempfile.TemporaryDirectory(prefix="surrogrid_stage_") as staging_root, ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(target_dir, staging_root),
    ) as executor:
        in_flight = set()
        for _ in range(max_in_flight):
            try:
                in_flight.add(executor.submit(export_census_cell, next(task_iter)))
            except StopIteration:
                break

        while in_flight:
            done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
            for fut in done:
                exported, errors = fut.result()
                exported_total += exported
                failed_grids.extend(errors)
                cells_done += 1

                if cells_done % progress_every_cells == 0 or cells_done == n_remaining_cells:
                    elapsed = time.perf_counter() - t_start
                    rate = exported_total / elapsed if elapsed > 0 else 0.0
                    eta_h = ((n_remaining_grids - exported_total) / rate / 3600.0) if rate > 0 else 0.0
                    print(
                        f"Progress: {exported_total}/{n_remaining_grids} grids | "
                        f"{cells_done}/{n_remaining_cells} weather cells | "
                        f"{rate:.2f} grids/s | elapsed {elapsed/60:.1f} min | ETA ~{eta_h:.1f} h | "
                        f"errors={len(failed_grids)}"
                    )

                try:
                    in_flight.add(executor.submit(export_census_cell, next(task_iter)))
                except StopIteration:
                    pass

    elapsed_total = time.perf_counter() - t_start
    if failed_grids:
        print(f"Completed with {len(failed_grids)} failed grids (first 10): {failed_grids[:10]}")
    return {"exported": exported_total, "failed": failed_grids, "elapsed_s": elapsed_total}
