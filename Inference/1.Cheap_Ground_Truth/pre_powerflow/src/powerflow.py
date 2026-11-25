"""Pandapower execution helpers for time-series power flow.

This module provides:

- `prepare_grid(net)`: relaxes constraints and replaces the transformer with a
    closed bus-bus switch to simplify the boundary condition.
- `pf(net, df_demand, parallel, n_cpu)`: runs a pandapower load-flow for each
    timestep in `df_demand` and returns external-grid imports, bus voltages, and
    line loadings.

Expected demand format:

- `df_demand` rows are timesteps.
- Columns are a 2-level MultiIndex `(bus, component)` where `component` is
    `electricity` (P) or `electricity-reactive` (Q). Values are assumed to be in
    kW/kVAr and are converted to MW/MVAr internally.
"""

import sys
from copy import deepcopy
from multiprocessing import Pool

import numpy as np
import pandapower as pp
import pandas as pd
from pandapower.pd2ppc import _ppc2ppci
from pandapower.pypower.idx_brch import BR_STATUS, F_BUS
from pandapower.pypower.idx_bus import BASE_KV, BUS_TYPE, VA, VM
from pandapower.pypower.makeYbus import makeYbus
from scipy.sparse.linalg import splu

def prepare_grid(grid):
    """ Set max_i to very high value. Remove transformer and replace with small line. """
    # Guard against zero-length lines that can crash pandapower
    df_lines = grid.line
    df_lines.loc[df_lines["length_km"] == 0.0, "length_km"] = 0.000001
    df_lines["max_i_ka"] = 1000
    grid.line = df_lines

    # Remove load max restrictions
    df_loads = grid.load
    df_loads["max_p_mw"] = 1000
    grid.load = df_loads

    # Remove voltage restrictions
    df_buses = grid.bus
    df_buses[["min_vm_pu", "max_vm_pu"]] = (0, 10)
    grid.bus = df_buses

    # Remove trafo and replace with switch
    trafo_buses = grid.trafo[["hv_bus", "lv_bus"]].values[0]
    grid.trafo.drop(index=0, inplace=True)

    ext_grid_bus = int(grid.ext_grid.loc[0, "bus"]) # bus which is the external import bus
    lv_bus = [bus for bus in trafo_buses if bus!=ext_grid_bus][0]
    grid.bus.loc[ext_grid_bus, "vn_kv"] = grid.bus.loc[lv_bus, "vn_kv"]

    pp.create_switch(
        grid,
        bus     = ext_grid_bus,
        element = lv_bus,
        et      = "b",
        closed  = True,
        type    = "CB",
        name    = f"SW_replacing_T0"
    )

    return grid


def run_single_pf(grid, new_load):
    # 1. Ensure that 'bus' is the index in both DataFrames
    df_load = grid.load.copy()
    df_load_indexed = df_load.set_index('bus')
    new_load_indexed = new_load.set_index('bus')
    # 2. Use DataFrame.update to overwrite only the overlapping entries
    df_load_indexed.update(new_load_indexed)
    # 3. If you need to restore 'bus' as a column rather than the index:
    df_load_updated = df_load_indexed.reset_index()

    grid.load = df_load_updated # Updates the grid's loads: Writes the loads from 2.demand_allocation (bzw. from 3.urbs) in the grid
    try:
        pp.runpp(grid, algorithm="bfsw", init = "flat", max_iteration=50, tolerance_mva=1e-6) #has pandapower run the powerflow
    except pp.LoadflowNotConverged:
        pp.diagnostic(grid, report_style='detailed')
        raise
    return grid


def _build_load_matrices(grid, df_demand):
    """Extract (n_t, n_loads) active [MW] and reactive [MVAr] load arrays matching `grid.load` order."""
    n_t = len(df_demand)
    load_buses = grid.load["bus"].to_numpy()
    p_mat = np.tile(grid.load["p_mw"].to_numpy(dtype=np.float64), (n_t, 1))
    q_mat = np.tile(grid.load["q_mvar"].to_numpy(dtype=np.float64), (n_t, 1))

    p_cols = df_demand.xs("electricity", level=1, axis=1)
    q_cols = df_demand.xs("electricity-reactive", level=1, axis=1)
    for idx_l, bus in enumerate(load_buses):
        if bus in p_cols.columns:
            p_mat[:, idx_l] = p_cols[bus].to_numpy(dtype=np.float64) / 1000.0
        if bus in q_cols.columns:
            q_mat[:, idx_l] = q_cols[bus].to_numpy(dtype=np.float64) / 1000.0

    return p_mat, q_mat


def _can_use_batched_solver(grid):
    """Return True if the network meets all assumptions of the exact batched Z-bus AC solver."""
    # Must have exactly one in-service ext_grid and no PV/controlled/non-linear elements
    if len(grid.ext_grid) != 1 or not bool(grid.ext_grid["in_service"].iloc[0]):
        return False

    for tbl_name in ("gen", "sgen", "shunt", "ward", "xward", "dcline", "trafo", "trafo3w", "impedance", "storage"):
        tbl = getattr(grid, tbl_name, None)
        if tbl is not None and len(tbl) > 0:
            if "in_service" not in tbl.columns or tbl["in_service"].any():
                return False

    # Batched constant-power formulation requires zero ZIP voltage-dependent load share
    for col in ("const_z_percent", "const_i_percent", "const_z_p_percent", "const_i_p_percent",
                "const_z_q_percent", "const_i_q_percent"):
        if col in grid.load.columns and (grid.load[col].fillna(0.0).to_numpy() != 0.0).any():
            return False

    return True


def _run_batched_ac_pf(grid, df_demand, p_mat, q_mat):
    """Solve exact 8760-hour AC power flow simultaneously using pandapower's own Ybus/Yf.

    Builds pandapower's internal `ppc`/`ppci` matrices via a single `pp.runpp(algorithm="bfsw")`
    call on timestep 0, factorizes `Y_LL` once, and runs batched Z-bus fixed-point iterations
    across all timesteps simultaneously. Cross-checks timestep 0 against pandapower's BFSW result
    and returns `None` to trigger an automatic fallback if any assumption or check fails.
    """
    n_t = len(df_demand)
    if n_t == 0 or not _can_use_batched_solver(grid):
        return None

    # Step 1: Run pandapower BFSW once on timestep 0 to construct exact ppc/ppci & lookups
    grid.load["p_mw"] = p_mat[0]
    grid.load["q_mvar"] = q_mat[0]
    try:
        pp.runpp(grid, algorithm="bfsw", init="flat", max_iteration=50, tolerance_mva=1e-6)
    except Exception:
        return None

    ppc = grid.get("_ppc")
    if ppc is None:
        return None
    baseMVA = float(ppc["baseMVA"])
    ppci = _ppc2ppci(ppc, grid)
    bus_i = ppci["bus"]
    branch_i = ppci["branch"]
    n_bus_i = bus_i.shape[0]

    bus_types = bus_i[:, BUS_TYPE].real.astype(np.int64)
    ref = np.where(bus_types == 3)[0]
    pv = np.where(bus_types == 2)[0]
    pq = np.where(bus_types == 1)[0]
    if len(ref) != 1 or len(pv) != 0 or len(pq) == 0:
        return None

    # Build pandapower's exact sparse admittance matrices once
    Ybus, Yf, _ = makeYbus(baseMVA, bus_i, branch_i)
    Ybus = Ybus.tocsc()
    Yf = Yf.tocsr()

    # Map pandapower bus/load/line indices -> internal ppci indices
    bus_lookup = grid._pd2ppc_lookups["bus"]
    is_bus_ppc = ppc["bus"][:, BUS_TYPE].real != 4
    ppc_to_ppci = np.full(ppc["bus"].shape[0], -1, dtype=np.int64)
    ppc_to_ppci[is_bus_ppc] = np.arange(int(is_bus_ppc.sum()), dtype=np.int64)
    pp_to_ppci = ppc_to_ppci[bus_lookup[grid.bus.index.to_numpy()]]

    # Assemble complex power injections S_bus_pu (n_bus_i, n_t) for all in-service loads
    load_buses = grid.load["bus"].to_numpy()
    load_in_service = grid.load["in_service"].to_numpy(dtype=bool)
    load_scaling = grid.load["scaling"].to_numpy(dtype=np.float64)
    load_ppci = ppc_to_ppci[bus_lookup[load_buses]]
    valid_load = load_in_service & (load_ppci >= 0)

    S_bus_pu = np.zeros((n_bus_i, n_t), dtype=np.complex128)
    if valid_load.any():
        p_eff = p_mat[:, valid_load] * load_scaling[valid_load]
        q_eff = q_mat[:, valid_load] * load_scaling[valid_load]
        S_load = -(p_eff + 1j * q_eff) / baseMVA
        np.add.at(S_bus_pu, load_ppci[valid_load], S_load.T)

    # Factorize Y_LL once and solve all timesteps simultaneously
    V_ref = bus_i[ref, VM] * np.exp(1j * np.deg2rad(bus_i[ref, VA]))
    Y_LL = Ybus[pq, :][:, pq].tocsc()
    Y_L0 = Ybus[pq, :][:, ref].tocsc()
    try:
        lu = splu(Y_LL)
    except Exception:
        return None

    V_noload = -lu.solve((Y_L0 @ V_ref).reshape(-1, 1)).ravel()
    S_pq = S_bus_pu[pq, :]
    V_pq = np.broadcast_to(V_noload[:, None], (len(pq), n_t)).copy()
    tol_pu = 1e-8 / baseMVA

    converged = False
    for _ in range(50):
        I_pq = np.conj(S_pq / V_pq)
        V_pq = V_noload[:, None] + lu.solve(I_pq)
        if not np.isfinite(V_pq).all():
            return None
        I_calc = Y_LL @ V_pq + (Y_L0 @ V_ref)[:, None]
        max_mis = float(np.max(np.abs(V_pq * np.conj(I_calc) - S_pq)))
        if max_mis < tol_pu:
            converged = True
            break

    if not converged:
        return None

    V = np.empty((n_bus_i, n_t), dtype=np.complex128)
    V[pq, :] = V_pq
    V[ref, :] = V_ref[:, None]

    # 1. External grid import (MW, MVAr)
    I_bus = Ybus @ V
    S_ref_mva = (V[ref, :] * np.conj(I_bus[ref, :]) - S_bus_pu[ref, :]) * baseMVA
    ext_fast = np.column_stack([S_ref_mva[0].real, S_ref_mva[0].imag])

    # 2. Bus voltage magnitudes (p.u.)
    vm_fast = np.full((n_t, len(grid.bus)), np.nan, dtype=np.float64)
    active_pp_bus = pp_to_ppci >= 0
    vm_fast[:, active_pp_bus] = np.abs(V[pp_to_ppci[active_pp_bus], :]).T

    # 3. Line power flows and currents (p_from_mw, q_from_mvar, i_from_ka)
    if "line" not in grid._pd2ppc_lookups["branch"]:
        return None
    f_l, t_l = grid._pd2ppc_lookups["branch"]["line"]
    is_brch_ppc = ppc["branch"][:, BR_STATUS].real != 0
    ppc_b_to_ppci = np.full(ppc["branch"].shape[0], -1, dtype=np.int64)
    ppc_b_to_ppci[is_brch_ppc] = np.arange(int(is_brch_ppc.sum()), dtype=np.int64)
    line_ppci = ppc_b_to_ppci[np.arange(f_l, t_l)]
    active_line = line_ppci >= 0

    n_line = len(grid.line)
    line_fast = np.zeros((n_t, n_line, 3), dtype=np.float64)
    if active_line.any():
        If = Yf[line_ppci[active_line], :] @ V
        f_bus_ppci = branch_i[line_ppci[active_line], F_BUS].real.astype(np.int64)
        Sf = V[f_bus_ppci, :] * np.conj(If) * baseMVA
        vn_kv_f = bus_i[f_bus_ppci, BASE_KV].real
        i_ka_f = np.abs(If) * (baseMVA / (np.sqrt(3.0) * vn_kv_f[:, None]))
        line_fast[:, active_line, 0] = Sf.real.T
        line_fast[:, active_line, 1] = Sf.imag.T
        line_fast[:, active_line, 2] = i_ka_f.T
    line_fast_2d = line_fast.reshape(n_t, n_line * 3)

    # Runtime verification on timestep 0 against pandapower's own BFSW results
    ref_ext_0 = grid.res_ext_grid[["p_mw", "q_mvar"]].to_numpy(dtype=np.float64)[0]
    ref_vm_0 = grid.res_bus["vm_pu"].to_numpy(dtype=np.float64)
    ref_line_0 = grid.res_line[["p_from_mw", "q_from_mvar", "i_from_ka"]].to_numpy(dtype=np.float64).ravel()

    ext_err = float(np.max(np.abs(ext_fast[0] - ref_ext_0)))
    vm_err = float(np.nanmax(np.abs(vm_fast[0] - ref_vm_0))) if np.isfinite(ref_vm_0).any() else 0.0
    line_err = float(np.nanmax(np.abs(line_fast_2d[0] - ref_line_0))) if len(ref_line_0) > 0 else 0.0
    if max(ext_err, vm_err, line_err) > 1e-4:
        return None

    ext_imports = pd.DataFrame(ext_fast, columns=grid.res_ext_grid.columns)
    vm = pd.DataFrame(vm_fast, columns=grid.bus.index)
    line_cols = pd.MultiIndex.from_product([grid.line.index, ["p_from_mw", "q_from_mvar", "i_from_ka"]])
    line_loads = pd.DataFrame(line_fast_2d, columns=line_cols)
    return ext_imports, vm, line_loads


def _run_full_pf_pandapower(grid, df_demand, p_mat, q_mat):
    """Fallback per-timestep pandapower BFSW loop with direct NumPy load updates & preallocated outputs."""
    n_t = len(df_demand)
    n_bus = len(grid.bus)
    n_line = len(grid.line)
    ext_arr = np.empty((n_t, 2), dtype=np.float64)
    vm_arr = np.empty((n_t, n_bus), dtype=np.float64)
    line_arr = np.empty((n_t, n_line * 3), dtype=np.float64)

    for idx_t in range(n_t):
        grid.load["p_mw"] = p_mat[idx_t]
        grid.load["q_mvar"] = q_mat[idx_t]
        try:
            pp.runpp(
                grid,
                algorithm="bfsw",
                init="flat" if idx_t == 0 else "results",
                max_iteration=50,
                tolerance_mva=1e-6,
                check_connectivity=(idx_t == 0),
            )
        except Exception as e:
            row_idx = df_demand.index[idx_t]
            total_p = float(p_mat[idx_t].sum())
            total_q = float(q_mat[idx_t].sum())
            max_p_load = float(p_mat[idx_t].max()) if len(p_mat[idx_t]) else 0.0
            max_q_load = float(q_mat[idx_t].max()) if len(q_mat[idx_t]) else 0.0
            print("\n=========================================", file=sys.stderr)
            print(f"[POWER FLOW FAILURE] Timestep index: {row_idx}", file=sys.stderr)
            print(f"Total active power load (P): {total_p:.6f} MW", file=sys.stderr)
            print(f"Total reactive power load (Q): {total_q:.6f} MVar", file=sys.stderr)
            print(f"Maximum single active load: {max_p_load:.6f} MW", file=sys.stderr)
            print(f"Maximum single reactive load: {max_q_load:.6f} MVar", file=sys.stderr)
            print("=========================================\n", file=sys.stderr)
            sys.stderr.flush()
            if isinstance(e, pp.LoadflowNotConverged):
                pp.diagnostic(grid, report_style="detailed")
            raise

        ext_arr[idx_t] = grid.res_ext_grid[["p_mw", "q_mvar"]].to_numpy(dtype=np.float64)[0]
        vm_arr[idx_t] = grid.res_bus["vm_pu"].to_numpy(dtype=np.float64)
        line_arr[idx_t] = grid.res_line[["p_from_mw", "q_from_mvar", "i_from_ka"]].to_numpy(dtype=np.float64).ravel()

    ext_imports = pd.DataFrame(ext_arr, columns=grid.res_ext_grid.columns)
    vm = pd.DataFrame(vm_arr, columns=grid.bus.index)
    line_cols = pd.MultiIndex.from_product([grid.line.index, ["p_from_mw", "q_from_mvar", "i_from_ka"]])
    line_loads = pd.DataFrame(line_arr, columns=line_cols)
    return ext_imports, vm, line_loads


def run_full_pf(grid, df_demand):
    p_mat, q_mat = _build_load_matrices(grid, df_demand)

    # Fast path: exact batched AC power flow using pandapower's Ybus/Yf (verified on timestep 0)
    batched_res = _run_batched_ac_pf(grid, df_demand, p_mat, q_mat)
    if batched_res is not None:
        return batched_res

    # Automatic fallback for any non-standard grid topology or convergence edge case
    return _run_full_pf_pandapower(grid, df_demand, p_mat, q_mat)


def pf(grid, df, parallel, n_cpu):
    # First try the single-process batched AC solver (completes 8760h in ~0.1-2s, faster than spawning Pool)
    p_mat, q_mat = _build_load_matrices(grid, df)
    batched_res = _run_batched_ac_pf(deepcopy(grid), df, p_mat, q_mat)
    if batched_res is not None:
        return batched_res

    if parallel and n_cpu > 1:
        # Partition df into n_cpu equal chunks for fallback pandapower BFSW execution
        chunk_size = (len(df) + n_cpu - 1) // n_cpu  # ceiling division
        chunks = [
            df.iloc[i * chunk_size : (i + 1) * chunk_size]
            for i in range(n_cpu)
            if i * chunk_size < len(df)
        ]
        job_args = [(deepcopy(grid), chunk) for chunk in chunks]
        with Pool(processes=len(chunks)) as pool:
            results = pool.starmap(run_full_pf, job_args)
        ext_imports = pd.concat([results[i][0] for i in range(len(results))], axis=0).reset_index(drop=True)
        vm = pd.concat([results[i][1] for i in range(len(results))], axis=0).reset_index(drop=True)
        line_loads = pd.concat([results[i][2] for i in range(len(results))], axis=0).reset_index(drop=True)
    else:
        ext_imports, vm, line_loads = _run_full_pf_pandapower(grid, df, p_mat, q_mat)

    return ext_imports, vm, line_loads