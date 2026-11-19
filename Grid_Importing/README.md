# Step 1: Bavaria Grid Importing & Export (`1.Grid_Importing`)

> **Origin note:** This folder (and this README) was originally copied from [`GridExpand/1.grid_sampling/`](../GridExpand/1.grid_sampling/README.md) and then modified for full-Bavaria surrogate inference (`Inference/`).

This step reads all valid Bavarian low-voltage (LV) distribution grids from the `pylovo` PostgreSQL database, enriches each grid with geographic/DSO metadata and PVGIS hourly weather, and writes one `.h5` file per grid directly to DSS (mounted via `sshfs` at `hpc_mount/`).

---

## Summary of Changes vs. `GridExpand/1.grid_sampling`

- **Full-region export (no sampling & no filler filtering):**
  - Reads all `86,721` grids directly from `pylovo.transformer_positions_with_grid` instead of running `pylovo-classify` or removing filler grids.
  - Removed the `1,500`-grid census sampling step in Notebook 2 so every valid grid (`85,613` total) is exported.
- **Reproduced `2.5 km` cable-distance filter (`Notebook 1`):**
  - Added `DataBase.read_grids_exceeding_trafo_distance(max_distance_km=2.5)` in [gridreadout/src/db_read.py](gridreadout/src/db_read.py), computing Dijkstra shortest cable paths from `LVbus` to `Consumer Nodebus` directly on `pandapower_bus` and `pandapower_line` (filtering `5` grids: `85,618 -> 85,613`).
  - The `> 150` households-per-building filter from `pylovo-classify` was intentionally omitted at this stage.
- **Coordinate System (SRID) fix (`EPSG:3035` -> `EPSG:25832`):**
  - Updated `Config.PYLOVO_COORD_FORMAT = "EPSG:25832"` in [gridreadout/config.py](gridreadout/config.py) to match the updated Bavaria `pylovo` database geometries (previously misprojected into `EPSG:3035`), and fixed Notebook 1's transformer reprojection (`EPSG:25832 -> EPSG:3035`).
- **Standardized filenames with `grid_index` and DSO short name:**
  - Sorted grids deterministically by `(plz_pylovo, kcid, bcid)` and assigned a zero-padded 5-digit `grid_index` (`00000`–`85612`).
  - Mapped `plz_pylovo` via [gridreadout/input_data/Zuordnung_PLZ-VNB.xlsx](gridreadout/input_data/Zuordnung_PLZ-VNB.xlsx) (`plz_vnb_bayern` sheet) to short DSO codes (`ByWrk`, `Nrgie`, `LEW`, `SWM`, `AllgN`, `SWA`, `RegB`, `MFrak`, `SWU`), with `nVNB` fallback for unmapped border PLZs (`74731`, `88147`).
  - Output filename format: `{grid_index}_{dso_short}_N{y}E{x}_{plz}_{kcid}_{bcid}.h5`.
- **Speed, API-limit & reliability optimizations:**
  - **1 km weather grouping:** Reused PVGIS TMY weather once per `1 km` Zensus 2022 cell centroid (`32,334` unique cells instead of `85,613` per-grid queries).
  - **Skipped Open-Meteo `soil_temp`:** Kept the `soil_temp` column in `/raw_data/weather` as `NaN` to avoid Open-Meteo's `10,000` calls/day free-tier cap (`GridForecast` only reads `temp_air`, `ghi`, `dni`, `dhi`, and ground-source heat pumps are disabled in `GridExpand`).
  - **Removed artificial sleep & added HTTP pooling:** Removed `time.sleep(0.5)` (`~11.9 h` saved) after confirming PVGIS's `30 req/s` cap, and added a reusable `requests.Session()`.
  - **Parallel export with local staging & DSS mount guard:** Added [gridreadout/src/parallel_export.py](gridreadout/src/parallel_export.py) (`8` isolated worker processes, local `/tmp` HDF5 assembly followed by single atomic `.part -> .h5` copy to `hpc_mount`), plus a mount check (`os.path.ismount`) so files are never written accidentally to local disk.
  - **Vectorized spatial joins & DB reconnect resilience:** Switched PLZ (`plz-5stellig.shp`) and municipality (`VG250_GEM.shp`) lookups to `geopandas.sjoin`, and added SQLAlchemy `pool_pre_ping`, TCP keepalives, and automatic reconnect retries in `db_read.py`.

---

## Folder Layout

- [gridreadout/](gridreadout): Notebooks (`1_filter_valid_grids.ipynb`, `2_sample_grids.ipynb`), `config.py`, and helper modules in `src/`
- [gridreadout/input_data/](gridreadout/input_data): Zensus 2022 CSV, PLZ/VG250 shapefiles, RegioStaR7 reference Excel, `Zuordnung_PLZ-VNB.xlsx`, and exported `valid_grids` HDF5
- [gridreadout/results/](gridreadout/results): Example output `.h5` files (e.g., [51431_Nrgie_N2894973E4345153_91555_1_36.h5](gridreadout/results/51431_Nrgie_N2894973E4345153_91555_1_36.h5))

---

## Setup

### 1) Create the Conda Environment

```bash
cd 1.Grid_Importing
conda env create -f environment.yml
conda activate grid_sampling
```

*(Note: ignore or remove the trailing `prefix:` line in [environment.yml](environment.yml) if it points to another machine's path.)*

### 2) Configure Database Access (`.env`)

Create `1.Grid_Importing/gridreadout/.env` (next to [gridreadout/config.py](gridreadout/config.py)):

```bash
DB_HOST=...
DB_PORT=54325
DB_NAME=...
DB_USER=...
DB_PASSWORD=...
```

Required read-only tables in the `pylovo` schema (used by [gridreadout/src/db_read.py](gridreadout/src/db_read.py)):
- `pylovo.transformer_positions_with_grid` (grid identifiers + transformer geometry in `EPSG:25832`)
- `pylovo.grid_result` (serialized pandapower grid JSON)
- `pylovo.pandapower_bus` & `pylovo.pandapower_line` (used by the `2.5 km` cable-path distance filter)
- `pylovo.buildings_result_with_grid` (building attributes and `EPSG:25832` centroids)

### 3) Mount DSS Storage (`hpc_mount`)

`Config.STORAGE_DIR` points to `/home/pedro/Linux-AntigravityProjects/SurroGrid-linux/hpc_mount`. Mount the DSS target folder via `sshfs` before running Notebook 2's export cell:

```bash
sshfs <lrz_user>@<lrz_login_host>:/dss/dssfs05/lwp-dss-0003/pn98cu/pn98cu-dss-0001/PedroC/INFERENCE/Raw_grids/ /home/pedro/Linux-AntigravityProjects/SurroGrid-linux/hpc_mount
```

---

## How to Run

Run Jupyter from inside `1.Grid_Importing/gridreadout/` so relative `src.*` and `input_data/` paths resolve properly:

```bash
cd 1.Grid_Importing/gridreadout
jupyter lab
```

### Notebook 1: [1_filter_valid_grids.ipynb](gridreadout/1_filter_valid_grids.ipynb)
1. Reads all `86,721` grids from `pylovo.transformer_positions_with_grid` and transforms transformer coordinates from `EPSG:25832` to `EPSG:3035`.
2. Matches each grid to the nearest inhabited `1 km` cell in `Zensus2022_Bevoelkerungszahl_1km-Gitter.csv` and removes grids farther than `sqrt(2 * 500^2)` m (`~707.1 m`) -> `85,618` grids remaining.
3. Removes grids with any consumer cable distance `> 2.5 km` from `LVbus` -> **`85,613` valid grids**.
4. Saves the filtered table to `gridreadout/input_data/valid_grids` (HDF5 key `grids`).

### Notebook 2: [2_sample_grids.ipynb](gridreadout/2_sample_grids.ipynb)
*(You do not need to rerun Notebook 1 unless filtering logic changes.)*
1. Loads `input_data/valid_grids` (`85,613` grids) without sampling.
2. Converts transformer `(x, y)` and Zensus cell centroid `(x_census, y_census)` to WGS84 (`EPSG:4326`).
3. Assigns `plz`, `gemeindeschlüssel`, `name`, `regio7`, `dso_short`, and deterministic `grid_index` (`00000`–`85612`).
4. Skips grids already present in `config.STORAGE_DIR`, then runs `pexp.run_parallel_export()` grouped by `1 km` Zensus cell across `8` worker processes (`~11.9 h` for `84,420` remaining grids in the full run).

---

## Output Format (`.h5`)

- **Filename:** `{grid_index}_{dso_short}_N{y}E{x}_{plz}_{kcid}_{bcid}.h5` (e.g., `51431_Nrgie_N2894973E4345153_91555_1_36.h5`), where `{x, y}` are transformer coordinates in `EPSG:3035`.
- **HDF5 entries written via [gridreadout/src/save_grid.py](gridreadout/src/save_grid.py):**
  - `/raw_data/net` (HDF5 dataset): JSON-serialized pandapower network (`pandapower.to_json(net)`).
  - `/raw_data/consumers` (Pandas table): consumer bus mapping (`src/grid_topol.get_consumers(net)`).
  - `/raw_data/region` (Pandas `Series`): `grid_index`, `dso_short`, `plz_pylovo`, `kcid`, `bcid`, `x`, `y`, `lat`, `lon`, `plz`, `gemeindeschlüssel`, `name`, `regio7`, `pop_density_1km2_cell`, `altitude`.
  - `/raw_data/buildings` (Pandas `DataFrame`): building attributes, coordinates (`lat`/`lon`), and assigned `bus`.
  - `/raw_data/weather` (Pandas `DataFrame`, `8760` rows): PVGIS SARAH3 TMY series (`temp_air`, `relative_humidity`, `ghi`, `dni`, `dhi`, `IR(h)`, `wind_speed`, `wind_direction`, `pressure`), `dew_point`, timestamps (`time(UTC)`, `time(UTC+1)`, `time(inst)`), and `soil_temp` (`NaN`).

### Skipping Step 1 (Using Existing `.h5` Grids)
If you already have compatible `.h5` files (matching the structure above), you can skip both notebooks and pass your folder directly to downstream steps.

---

## Code Overview (`gridreadout/src/`)

- [db_read.py](gridreadout/src/db_read.py): Read-only PostgreSQL access with TCP keepalives, pool pre-ping, automatic reconnect retries, and the `2.5 km` Dijkstra distance filter.
- [parallel_export.py](gridreadout/src/parallel_export.py): Multiprocessing orchestrator grouped by `1 km` Zensus cell, with per-worker `/tmp` HDF5 staging and atomic `.part -> .h5` transfer to `hpc_mount`.
- [weather.py](gridreadout/src/weather.py): PVGIS SARAH3 TMY fetcher with shared `requests.Session()`, dew-point calculation, and optional Open-Meteo `soil_temp` (`NaN` by default).
- [save_grid.py](gridreadout/src/save_grid.py): Standardized filename builder (`build_grid_filename`) and HDF5 writer (`SaveFile`).
- [grid_topol.py](gridreadout/src/grid_topol.py): Grid topology cleanup (`assign_min_linelen`, `remove_duplicate_loads`, `get_consumers`).
- [powerflow.py](gridreadout/src/powerflow.py): Standalone pandapower power-flow helper for quick sanity checks.