#!/usr/bin/env python3
"""Progressive doubling tuner for building refurbishment parameters (p_max_res, p_max_ghd).

Algorithm:
1. Retrieves ground-truth grids from `DATA_GRID_DIR` and shuffles them deterministically.
2. Evaluates increasing batch sizes: 50 -> 100 -> 200 -> 400 -> 800 -> all grids.
3. At the beginning of each iteration, clears `STORAGE_DIR` so only the `.h5` files from
   the current iteration remain in `STORAGE_DIR`.
4. Simulates space-heating demand across the batch in parallel across `n_cpu` workers and
   saves the new `.h5` files into `STORAGE_DIR`.
5. Computes the area-weighted specific space-heating demand (Nutzwärmeenergie per m² net
   use area = 0.75 * floor_area) for purely residential buildings and tunes `p_max_res`
   against the Genehmigung SR 2025 (Scenario B 2045) reference (103.8 kWh_th/(m²·a)).
6. Repeats the same calculation for purely non-residential (GHD) buildings against the
   GHD reference (90.6 kWh_th/(m²·a)) to tune `p_max_ghd`.
7. Saves the iteration history to `tuning_history.csv` inside `STORAGE_DIR`.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import sys
import time
import zlib

import numpy as np
import pandas as pd

# Disable HDF5 file locking on network / HPC filesystems
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

# Configure import paths so gridalloc and pipeline_config resolve cleanly
BASE_DIR = Path(__file__).resolve().parent
GRIDALLOC_DIR = BASE_DIR / "gridalloc"
for path_str in (str(BASE_DIR), str(GRIDALLOC_DIR)):
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import pipeline_config as pcfg
from config import config
from gridalloc.main import run_single_grid_heat


def list_shuffled_grid_files(input_dir, seed=pcfg.TUNING_SHUFFLE_SEED):
    """Return all raw `.h5` grid filenames in `input_dir`, shuffled deterministically."""
    exclude_suffixes = (config.OUTPUT_SUFFIX + ".h5", "_pwrflw.h5")
    all_files = sorted(
        f for f in os.listdir(input_dir)
        if f.endswith(".h5") and not f.endswith(exclude_suffixes)
    )
    if not all_files:
        raise FileNotFoundError(f"No raw .h5 grid files found in '{input_dir}'.")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(all_files))
    return [all_files[i] for i in indices]


def build_batch_schedule(total_grids, base_schedule=None):
    """Build strictly increasing batch sizes ending with `total_grids` (e.g. 50, 100, 200, 400, 800, N)."""
    if base_schedule is None:
        base_schedule = pcfg.TUNING_BATCH_SIZES
    schedule = [int(n) for n in base_schedule if 0 < int(n) < total_grids]
    schedule.append(int(total_grids))
    return sorted(set(schedule))


def clear_output_directory(output_dir):
    """Remove all existing `.h5` files from `output_dir` before starting a new iteration."""
    os.makedirs(output_dir, exist_ok=True)
    removed = 0
    for entry in os.listdir(output_dir):
        if entry.endswith(".h5"):
            try:
                os.remove(os.path.join(output_dir, entry))
                removed += 1
            except OSError as e:
                print(f"Warning: could not remove '{entry}': {e}")
    return removed


def update_p_max(p_max_curr, q_curr, q_target, q_baseline):
    """Update `p_max` using the linear secant relation anchored at the unrenovated baseline `q_baseline`.

    Since expected heat demand scales linearly with `p_max`:
        q(p_max) = q_baseline - p_max * (q_baseline - q_full_renov),
    the updated parameter that hits `q_target` is:
        p_max_next = p_max_curr * (q_baseline - q_target) / (q_baseline - q_curr).
    """
    if np.isnan(q_curr) or q_curr <= 0.0:
        return float(p_max_curr)

    denom = float(q_baseline - q_curr)
    numer = float(q_baseline - q_target)

    # Guard against small-sample noise if observed demand is close to or above the unrenovated baseline
    if denom <= 1e-3:
        scaled = p_max_curr * (q_curr / q_target)
        return float(np.clip(scaled, 0.01, 1.0))

    p_max_next = float(p_max_curr) * (numer / denom)
    return float(np.clip(p_max_next, 0.01, 1.0))


def _worker_task(args):
    """Top-level multiprocessing worker wrapper for a single grid."""
    grid_filename, input_dir, output_dir, p_max_res, p_max_ghd = args
    # Deterministic per-grid seed so stochastic occupancy / age sampling is reproducible
    seed = zlib.adler32(grid_filename.encode("utf-8")) & 0x7FFFFFFF
    return run_single_grid_heat(
        grid_filename=grid_filename,
        input_dir=input_dir,
        output_dir=output_dir,
        p_max_res=p_max_res,
        p_max_ghd=p_max_ghd,
        n_cpu=1,
        seed=seed,
    )


def run_tuning(
    input_dir=None,
    output_dir=None,
    n_cpu=1,
    p_max_res_init=None,
    p_max_ghd_init=None,
    batch_sizes=None,
    max_grids=None,
):
    """Execute the progressive doubling tuning loop for `p_max_res` and `p_max_ghd`."""
    input_dir = input_dir if input_dir is not None else pcfg.DATA_GRID_DIR
    output_dir = output_dir if output_dir is not None else pcfg.STORAGE_DIR
    p_max_res = float(p_max_res_init if p_max_res_init is not None else config.RETROFIT_P_MAX_RES)
    p_max_ghd = float(p_max_ghd_init if p_max_ghd_init is not None else config.RETROFIT_P_MAX_GHD)

    if os.path.abspath(input_dir) == os.path.abspath(output_dir):
        raise ValueError("input_dir and output_dir must be distinct directories.")

    all_grids = list_shuffled_grid_files(input_dir)
    if max_grids is not None and int(max_grids) > 0:
        all_grids = all_grids[: int(max_grids)]

    total_grids = len(all_grids)
    schedule = build_batch_schedule(total_grids, base_schedule=batch_sizes)

    print("=" * 80)
    print(f"Starting Gebäudesanierung p_max tuning across {total_grids} ground-truth grid(s)")
    print(f"  Input directory:  {input_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Batch schedule:   {schedule}")
    print(f"  Parallel workers: {n_cpu}")
    print(f"  Initial p_max:    Res = {p_max_res:.4f}, GHD = {p_max_ghd:.4f}")
    print(f"  Target q [kWh/(m²·a)]: Res = {pcfg.TARGET_Q_RES:.2f} (q0={pcfg.BASELINE_Q0_RES:.2f}), "
          f"GHD = {pcfg.TARGET_Q_GHD:.2f} (q0={pcfg.BASELINE_Q0_GHD:.2f})")
    print("=" * 80, flush=True)

    history = []

    for iter_idx, batch_n in enumerate(schedule, start=1):
        t_iter = time.perf_counter()
        batch_files = all_grids[:batch_n]

        # Step 2a: Clear output folder so only the current iteration's grids remain
        removed_count = clear_output_directory(output_dir)
        print(
            f"\n[Iteration {iter_idx}/{len(schedule)}] Batch size = {batch_n} grid(s) "
            f"(cleared {removed_count} previous .h5 file(s))"
        )
        print(f"  Using p_max_res = {p_max_res:.4f}, p_max_ghd = {p_max_ghd:.4f}", flush=True)

        # Step 2b: Simulate heat demand for all grids in the current batch
        worker_args = [
            (fname, input_dir, output_dir, p_max_res, p_max_ghd)
            for fname in batch_files
        ]

        batch_results = []
        if int(n_cpu) > 1 and batch_n > 1:
            with ProcessPoolExecutor(max_workers=min(int(n_cpu), batch_n)) as executor:
                futures = {executor.submit(_worker_task, arg): arg[0] for arg in worker_args}
                for done_idx, future in enumerate(as_completed(futures), start=1):
                    fname = futures[future]
                    res = future.result()
                    batch_results.append(res)
                    if done_idx % max(1, batch_n // 10) == 0 or done_idx == batch_n:
                        print(f"    Processed {done_idx}/{batch_n} grids (latest: {fname})", flush=True)
        else:
            for done_idx, arg in enumerate(worker_args, start=1):
                res = _worker_task(arg)
                batch_results.append(res)
                if done_idx % max(1, batch_n // 10) == 0 or done_idx == batch_n:
                    print(f"    Processed {done_idx}/{batch_n} grids (latest: {arg[0]})", flush=True)

        # Step 3: Calculate Nutzwärmeenergie for purely residential buildings and tune p_max_res
        tot_res_sh = sum(r["res_space_heat_kwh"] for r in batch_results)
        tot_res_gross_area = sum(r["res_floor_area_m2"] for r in batch_results)
        tot_res_net_area = pcfg.NET_AREA_FACTOR * tot_res_gross_area
        n_res_bldgs = sum(r["n_pure_res"] for r in batch_results)
        n_res_renov = sum(r["n_pure_res_renov"] for r in batch_results)

        q_res = (tot_res_sh / tot_res_net_area) if tot_res_net_area > 0 else np.nan
        p_max_res_next = update_p_max(
            p_max_curr=p_max_res,
            q_curr=q_res,
            q_target=pcfg.TARGET_Q_RES,
            q_baseline=pcfg.BASELINE_Q0_RES,
        )

        # Step 4: Calculate Nutzwärmeenergie for purely GHD buildings and tune p_max_ghd
        tot_ghd_sh = sum(r["ghd_space_heat_kwh"] for r in batch_results)
        tot_ghd_gross_area = sum(r["ghd_floor_area_m2"] for r in batch_results)
        tot_ghd_net_area = pcfg.NET_AREA_FACTOR * tot_ghd_gross_area
        n_ghd_bldgs = sum(r["n_pure_ghd"] for r in batch_results)
        n_ghd_renov = sum(r["n_pure_ghd_renov"] for r in batch_results)

        q_ghd = (tot_ghd_sh / tot_ghd_net_area) if tot_ghd_net_area > 0 else np.nan
        p_max_ghd_next = update_p_max(
            p_max_curr=p_max_ghd,
            q_curr=q_ghd,
            q_target=pcfg.TARGET_Q_GHD,
            q_baseline=pcfg.BASELINE_Q0_GHD,
        )

        elapsed = time.perf_counter() - t_iter
        rec = {
            "iteration": iter_idx,
            "n_grids": batch_n,
            "n_pure_res_buildings": n_res_bldgs,
            "n_pure_res_renovated": n_res_renov,
            "p_max_res_used": p_max_res,
            "q_res_kwh_per_m2a": q_res,
            "q_res_target": pcfg.TARGET_Q_RES,
            "p_max_res_tuned": p_max_res_next,
            "n_pure_ghd_buildings": n_ghd_bldgs,
            "n_pure_ghd_renovated": n_ghd_renov,
            "p_max_ghd_used": p_max_ghd,
            "q_ghd_kwh_per_m2a": q_ghd,
            "q_ghd_target": pcfg.TARGET_Q_GHD,
            "p_max_ghd_tuned": p_max_ghd_next,
            "wall_seconds": round(elapsed, 2),
        }
        history.append(rec)

        # Persist tuning history after every iteration
        df_hist = pd.DataFrame(history)
        hist_path = os.path.join(output_dir, "tuning_history.csv")
        df_hist.to_csv(hist_path, index=False)

        print(
            f"  [Res] n={n_res_bldgs} ({n_res_renov} renov) | q_res = {q_res:.2f} vs {pcfg.TARGET_Q_RES:.2f} kWh/(m²·a) "
            f"| p_max_res: {p_max_res:.4f} -> {p_max_res_next:.4f}"
        )
        print(
            f"  [GHD] n={n_ghd_bldgs} ({n_ghd_renov} renov) | q_ghd = {q_ghd:.2f} vs {pcfg.TARGET_Q_GHD:.2f} kWh/(m²·a) "
            f"| p_max_ghd: {p_max_ghd:.4f} -> {p_max_ghd_next:.4f}"
        )
        print(f"  Iteration {iter_idx} finished in {elapsed:.1f}s (history saved to {hist_path})", flush=True)

        # Advance tuned parameters to the next iteration
        p_max_res = p_max_res_next
        p_max_ghd = p_max_ghd_next

    print("\n" + "=" * 80)
    print("Tuning complete! Final summary across iterations:")
    print(pd.DataFrame(history).to_string(index=False))
    print(f"Final tuned parameters: RETROFIT_P_MAX_RES = {p_max_res:.4f}, RETROFIT_P_MAX_GHD = {p_max_ghd:.4f}")
    print("=" * 80, flush=True)
    return pd.DataFrame(history)


def main():
    parser = argparse.ArgumentParser(
        description="Progressive doubling tuner for residential and GHD refurbishment probabilities (p_max)."
    )
    parser.add_argument("--n_cpu", type=int, default=4, help="Number of parallel worker processes across grids")
    parser.add_argument("--input_dir", default=None, help="Override DATA_GRID_DIR in pipeline_config.py")
    parser.add_argument("--output_dir", default=None, help="Override STORAGE_DIR in pipeline_config.py")
    parser.add_argument("--p_max_res", type=float, default=None, help="Initial residential p_max (default 0.40)")
    parser.add_argument("--p_max_ghd", type=float, default=None, help="Initial GHD p_max (default 0.40)")
    parser.add_argument(
        "--batch_sizes",
        default=None,
        help="Optional comma-separated batch sizes (e.g. '2,4' for local testing; default '50,100,200,400,800')",
    )
    parser.add_argument(
        "--max_grids",
        type=int,
        default=None,
        help="Optional cap on total number of grids to process (useful for quick local smoke tests)",
    )
    args = parser.parse_args()

    batch_sizes = (
        [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]
        if args.batch_sizes
        else None
    )

    run_tuning(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        n_cpu=args.n_cpu,
        p_max_res_init=args.p_max_res,
        p_max_ghd_init=args.p_max_ghd,
        batch_sizes=batch_sizes,
        max_grids=args.max_grids,
    )


if __name__ == "__main__":
    main()
