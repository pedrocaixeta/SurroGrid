# -*- coding: utf-8 -*-
"""
Script to generate heating and cooling demands sccording to DIN EN ISO 13790.
"""

from __future__ import division
import numpy as np
import numpy.linalg as linalg


def _solve(A, b):
    return linalg.solve(A, b)


def _calculateNoHeat(zoneParameters, T_e, t_m_previous, dt, timestep):
    """
    Calculate the temperatures (T_op, T_m, T_air, T_s) if neither heating nor
    cooling devices are activated.
    This is necessary to enable a deadband between cooling and heating mode.

    Parameters
    ----------
    zoneParameters : ZoneParameters
        Resistances and capacity
    zoneInputs : ZoneInputs
        External inputs (solar, internal gains, set temperatures)
    T_m_init : float
        Initial temperature of the thermal mass in degree Celsius.
    timestep : integer, optional
        Define which index is relevant (zoneInputs, H_ve)

    Returns
    -------
    T_op : float
        .
    T_m : float
        .
    T_air : float
        .
    T_s : float
        .
    """

    # Note: If not stated differently, all equations, pages and sections
    # refer to DIN EN ISO 13790:2008 (the official German version of
    # ISO 13790:2008).

    # Extract parameters
    H_tr_is = zoneParameters.H_tr_is  # in W/K
    H_tr_ms = zoneParameters.H_tr_ms  # in W/K
    H_tr_w = zoneParameters.H_tr_w  # in W/K
    H_ve = zoneParameters.H_ve  # in W/K
    C_m = zoneParameters.C_m  # in J/K
    H_tr_em = zoneParameters.H_tr_em[0]  # in W/K

    Phi_ia = zoneParameters.phi_ia
    Phi_m = zoneParameters.phi_m
    Phi_st = zoneParameters.phi_st

    # Initialize A*x = b
    # x: T_m, T_s, T_air (T_i), Q_HC
    A = np.zeros((3, 3))
    b = np.zeros(3)

    # Row wise entering
    A[0, 0] = H_tr_em + H_tr_ms + C_m / (3600 * dt)
    A[0, 1] = - H_tr_ms
    A[1, 0] = - H_tr_ms
    A[1, 1] = H_tr_ms + H_tr_is + H_tr_w
    A[1, 2] = - H_tr_is
    A[2, 1] = - H_tr_is
    A[2, 2] = H_ve + H_tr_is

    b[0] = Phi_m[timestep] + H_tr_em * T_e[timestep] + C_m * t_m_previous / (3600 * dt)
    b[1] = Phi_st[timestep] + H_tr_w * T_e[timestep]
    b[2] = Phi_ia[timestep] + H_ve * T_e[timestep]

    # Solve for "x"
    x = _solve(A, b)

    T_i = x[2]
    T_s = x[1]
    T_m = x[0]

    weight = 0.3
    T_op = weight * T_i + (1 - weight) * T_s
    return (T_op, T_m, T_i, T_s)


def _calculateHeat(zoneParameters, T_e, T_set, T_m_init, dt, timestep):
    """
    Calculate the temperatures (Q_HC, T_op, T_m, T_air, T_s) that result when
    reaching a given set temperature T_set.

    Parameters
    ----------
    zoneParameters : ZoneParameters
        Resistances and capacity
    zoneInputs : ZoneInputs
        External inputs (solar, internal gains, set temperatures)
    T_m_init : float
        Initial temperature of the thermal mass in degree Celsius.
    T_set : float
        Set temperature in degree Celsius.
    timestep : integer, optional
        Define which index is relevant (zoneInputs, H_ve)

    Returns
    -------
    Q_HC : float
        Heating (positive) or cooling (negative) load for the current time
        step in Watt.
    T_op : float
        .
    T_m : float
        .
    T_air : float
        .
    T_s : float
        .
    """

    # Note: If not stated differently, all equations, pages and sections
    # refer to DIN EN ISO 13790:2008 (the official German version of
    # ISO 13790:2008).

    # Extract parameters
    H_tr_is = zoneParameters.H_tr_is  # in W/K
    H_tr_ms = zoneParameters.H_tr_ms  # in W/K
    H_tr_w = zoneParameters.H_tr_w  # in W/K
    H_ve = zoneParameters.H_ve  # in W/K
    C_m = zoneParameters.C_m  # in J/K
    H_tr_em = zoneParameters.H_tr_em[0]  # in W/K
    Q_nHC = zoneParameters.heatload  # design (nominal) heat load

    Phi_ia = zoneParameters.phi_ia
    Phi_m = zoneParameters.phi_m
    Phi_st = zoneParameters.phi_st

    # Initialize A*x = b
    # x: T_m, T_s, T_air (T_i), Q_HC
    A = np.zeros((4, 4))
    b = np.zeros(4)

    # Row wise entering
    A[0, 0] = H_tr_em + H_tr_ms + C_m / (3600 * dt)
    A[0, 1] = - H_tr_ms
    A[1, 0] = - H_tr_ms
    A[1, 1] = H_tr_ms + H_tr_is + H_tr_w
    A[1, 2] = - H_tr_is
    A[2, 1] = - H_tr_is
    A[2, 2] = H_ve + H_tr_is
    A[2, 3] = -1
    A[3, 2] = 0.3
    A[3, 1] = 1 - A[3, 2]

    b[0] = Phi_m[timestep] + H_tr_em * T_e[timestep] + C_m * T_m_init / (3600 * dt)
    b[1] = Phi_st[timestep] + H_tr_w * T_e[timestep]
    b[2] = Phi_ia[timestep] + H_ve * T_e[timestep]
    b[3] = T_set

    # Solve for "x"
    x = _solve(A, b)

    # Linear system of equations to determine T_i, T_s, T_m, Q_HC (in kW)
    T_i = x[2]
    T_s = x[1]
    T_m = x[0]
    Q_HC = x[3]

    # If Q_HC exceeds Q_nHC, re-solve assuming Q_HC = Q_nHC
    if Q_HC > Q_nHC:
        # Remove the row and column corresponding to Q_HC
        A_reduced = A[:3, :3]  # Exclude the last row and column
        b_reduced = b[:3]  # Exclude the last element of b

        # Adjust b to account for Q_HC = Q_nHC
        b_reduced[2] += Q_nHC  # Subtract fixed Q_HC from the third equation

        # Re-solve the system
        x_reduced = _solve(A_reduced, b_reduced)  # Exclude Q_HC column and row

        # Update results
        T_i = x_reduced[2]
        T_s = x_reduced[1]
        T_m = x_reduced[0]
        Q_HC = Q_nHC

    weight = 0.3
    T_op = weight * T_i + (1 - weight) * T_s
    return (Q_HC, T_op, T_m, T_i, T_s)


def calc(zoneParameters, T_e, holidays, dt, building_type):
    """Fast pre-inverted 5R1C solver (DIN EN ISO 13790).

    Since the 3x3 (free-float / max-power) and 4x4 (setpoint) conductance matrices A
    are time-invariant for a building, pre-inverting them once per building avoids
    ~15,000 calls to `np.linalg.solve` per building while producing identical results.
    """
    is_res = building_type in {"SFH", "TH", "MFH", "AB"}
    T_m_init = 19.0 if is_res else 12.0

    T_set = float(zoneParameters.T_set_min)
    T_set_ub = float(zoneParameters.T_set_max)
    numberTimesteps = len(T_e)

    # Extract time-invariant building thermal parameters
    H_tr_is = float(np.squeeze(zoneParameters.H_tr_is))
    H_tr_ms = float(np.squeeze(zoneParameters.H_tr_ms))
    H_tr_w = float(np.squeeze(zoneParameters.H_tr_w))
    H_ve = float(np.squeeze(zoneParameters.H_ve))
    C_m = float(np.squeeze(zoneParameters.C_m))
    H_tr_em = float(np.squeeze(zoneParameters.H_tr_em[0]))
    Q_nHC = float(np.squeeze(zoneParameters.heatload))
    c_dt = C_m / (3600.0 * dt)

    # Build and pre-invert the 4x4 setpoint matrix A4 and 3x3 free-floating matrix A3
    A4 = np.zeros((4, 4), dtype=float)
    A4[0, 0] = H_tr_em + H_tr_ms + c_dt
    A4[0, 1] = -H_tr_ms
    A4[1, 0] = -H_tr_ms
    A4[1, 1] = H_tr_ms + H_tr_is + H_tr_w
    A4[1, 2] = -H_tr_is
    A4[2, 1] = -H_tr_is
    A4[2, 2] = H_ve + H_tr_is
    A4[2, 3] = -1.0
    A4[3, 2] = 0.3
    A4[3, 1] = 0.7

    A3 = A4[:3, :3]
    invA3 = linalg.inv(A3)
    invA4 = linalg.inv(A4)

    a3_00, a3_01, a3_02 = invA3[0, 0], invA3[0, 1], invA3[0, 2]
    a3_10, a3_11, a3_12 = invA3[1, 0], invA3[1, 1], invA3[1, 2]
    a3_20, a3_21, a3_22 = invA3[2, 0], invA3[2, 1], invA3[2, 2]

    a4_00, a4_01, a4_02, a4_03 = invA4[0, 0], invA4[0, 1], invA4[0, 2], invA4[0, 3]
    a4_10, a4_11, a4_12, a4_13 = invA4[1, 0], invA4[1, 1], invA4[1, 2], invA4[1, 3]
    a4_20, a4_21, a4_22, a4_23 = invA4[2, 0], invA4[2, 1], invA4[2, 2], invA4[2, 3]
    a4_30, a4_31, a4_32, a4_33 = invA4[3, 0], invA4[3, 1], invA4[3, 2], invA4[3, 3]

    # Pre-compute time-varying right-hand-side base vectors across all 8,760 hours
    T_e_arr = np.asarray(T_e, dtype=float)
    if isinstance(zoneParameters.phi_m, dict):
        phi_m_arr = np.array([float(np.squeeze(zoneParameters.phi_m[t])) for t in range(numberTimesteps)])
        phi_st_arr = np.array([float(np.squeeze(zoneParameters.phi_st[t])) for t in range(numberTimesteps)])
    else:
        phi_m_arr = np.asarray(zoneParameters.phi_m, dtype=float)
        phi_st_arr = np.asarray(zoneParameters.phi_st, dtype=float)
    phi_ia_arr = np.asarray(zoneParameters.phi_ia, dtype=float)

    b0_base = phi_m_arr + H_tr_em * T_e_arr
    b1_arr = phi_st_arr + H_tr_w * T_e_arr
    b2_arr = phi_ia_arr + H_ve * T_e_arr

    T_i = np.zeros(numberTimesteps)
    T_s = np.zeros(numberTimesteps)
    T_m = np.zeros(numberTimesteps)
    Q_H = np.zeros(numberTimesteps)
    Q_C = np.zeros(numberTimesteps)
    T_op = np.zeros(numberTimesteps)

    timesteps_per_day = numberTimesteps / 365.0
    holidays_set = set(holidays)
    t_m_previous = T_m_init

    for t in range(numberTimesteps):
        day = int(t // timesteps_per_day)
        b0 = b0_base[t] + c_dt * t_m_previous
        b1 = b1_arr[t]
        b2 = b2_arr[t]

        # 1. Free-floating temperatures without heating/cooling (deadband)
        t_m = a3_00 * b0 + a3_01 * b1 + a3_02 * b2
        t_s = a3_10 * b0 + a3_11 * b1 + a3_12 * b2
        t_i = a3_20 * b0 + a3_21 * b1 + a3_22 * b2
        t_op = 0.3 * t_i + 0.7 * t_s

        active_setpoint = None
        if is_res:
            if t_op < T_set and not (135 <= day < 259):
                active_setpoint = T_set
            elif t_op > T_set_ub:
                active_setpoint = T_set_ub
        else:
            is_workday = (day % 7 not in (0, 6)) and (day not in holidays_set)
            if is_workday:
                if t_op < T_set and not (135 <= day < 259):
                    active_setpoint = T_set
                elif t_op > T_set_ub:
                    active_setpoint = T_set_ub

        if active_setpoint is not None:
            b3 = active_setpoint
            q_hc = a4_30 * b0 + a4_31 * b1 + a4_32 * b2 + a4_33 * b3
            if q_hc > Q_nHC:
                b2_red = b2 + Q_nHC
                t_m = a3_00 * b0 + a3_01 * b1 + a3_02 * b2_red
                t_s = a3_10 * b0 + a3_11 * b1 + a3_12 * b2_red
                t_i = a3_20 * b0 + a3_21 * b1 + a3_22 * b2_red
                q_hc = Q_nHC
            else:
                t_m = a4_00 * b0 + a4_01 * b1 + a4_02 * b2 + a4_03 * b3
                t_s = a4_10 * b0 + a4_11 * b1 + a4_12 * b2 + a4_13 * b3
                t_i = a4_20 * b0 + a4_21 * b1 + a4_22 * b2 + a4_23 * b3
            t_op = 0.3 * t_i + 0.7 * t_s
        else:
            q_hc = 0.0

        if q_hc >= 0.0:
            Q_H[t] = q_hc
        else:
            Q_C[t] = -q_hc

        T_m[t] = t_m
        T_i[t] = t_i
        T_s[t] = t_s
        T_op[t] = t_op
        t_m_previous = t_m

    return (Q_H, Q_C, T_op, T_m, T_i, T_s)


def calculate(zoneParameters, T_set, T_e, dt):
    """

    Parameters
    ----------
    zoneParameters : ZoneParameters
        Resistances and capacity
    zoneInputs : ZoneInputs
        External inputs (solar, internal gains, set temperatures)
    T_m_init : float
        Initial temperature of the thermal mass in degC
    """

    T_m_init = 20  # [°C]
    # T_set = zoneParameters.T_set_min

    # Note: If not stated differently, all equations, pages and sections
    # refer to DIN EN ISO 13790:2008 (the official German version of
    # ISO 13790:2008).

    # Extract parameters
    H_tr_is = zoneParameters.H_tr_is  # in W/K
    H_tr_ms = zoneParameters.H_tr_ms  # in W/K
    H_tr_em = zoneParameters.H_tr_em[0]  # in W/K
    H_tr_w = zoneParameters.H_tr_w  # in W/K
    H_ve = zoneParameters.H_ve  # in W/K
    C_m = zoneParameters.C_m  # in J/K

    T_sup = T_e  # zoneInputs.T_sup

    numberTimesteps = len(T_e)

    # Compute internal and solar heat sources
    # Equations C1-C3, section C2, page 110
    Phi_ia = zoneParameters.phi_ia
    Phi_m = zoneParameters.phi_m
    Phi_st = zoneParameters.phi_st

    # Initialize results
    T_i = np.zeros(numberTimesteps)
    T_s = np.zeros(numberTimesteps)
    T_m = np.zeros(numberTimesteps)
    Q_HC = np.zeros(numberTimesteps)

    # Initialize A*x = b
    # x: T_m, T_s, T_air (T_i), Q_HC
    A = np.zeros((4, 4))
    b = np.zeros(4)

    # Row wise entering
    # Set the time-invariant components of A
    A[0, 0] = H_tr_em + H_tr_ms + C_m / (3600 * dt)
    A[0, 1] = - H_tr_ms
    A[1, 0] = - H_tr_ms
    A[1, 1] = H_tr_ms + H_tr_is + H_tr_w
    A[1, 2] = - H_tr_is
    A[2, 1] = - H_tr_is
    A[2, 3] = -1
    # if A[3,2] = 1 T_i is used for temperature control. Otherwise T_e is used! -> check MA Dominik
    A[3, 2] = 1  # 0.3
    A[3, 1] = 1 - A[3, 2]

    # Only the right hand side (b) changes. This is done for each time step:
    for t in range(numberTimesteps):
        if t == 0:
            T_m_previous = T_m_init
        else:
            T_m_previous = T_m[t - 1]

        # Set the time-variable components of A
        A[2, 2] = H_ve + H_tr_is

        b[0] = Phi_m[t] + H_tr_em * T_e[t] + C_m * T_m_previous / (3600 * dt)
        b[1] = Phi_st[t] + H_tr_w * T_e[t]
        b[2] = Phi_ia[t] + H_ve * T_sup[t]
        b[3] = T_set

        # Solve for "x"
        x = _solve(A, b)

        T_i[t] = x[2]
        T_s[t] = x[1]
        T_m[t] = x[0]
        Q_HC[t] = x[3]

    T_op = A[3, 2] * T_i + A[3, 1] * T_s
    return (Q_HC, T_i, T_s, T_m, T_op)