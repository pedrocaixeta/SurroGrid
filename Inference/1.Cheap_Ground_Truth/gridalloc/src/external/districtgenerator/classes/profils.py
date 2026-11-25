# -*- coding: utf-8 -*-

import os
import math
import random
import numpy as np
import pylightxl as xl
import richardsonpy.classes.occupancy as occ_residential
import richardsonpy.functions.change_resolution as cr
import OpenDHW
import src.external.districtgenerator.functions.change_resolution as chres

# Process-local caches for deterministic OpenDHW lookup tables (reused across flats & grids)
_DHW_CDF_CACHE = {}
_DHW_TEMP_DT_CACHE = {}


def _get_cached_dhw_temp_dt(time_resolution):
    """Return cached seasonal DHW temperature difference array for `time_resolution` [s]."""
    cached = _DHW_TEMP_DT_CACHE.get(time_resolution)
    if cached is not None:
        return cached

    days = np.arange(365, dtype=np.float64)
    # Seasonal mixed-water (45 +/- 3 C) and cold-water (10 +/- 7 C) temperatures
    t_mixed = 45.0 + 3.0 * np.cos(math.pi * (2.0 / 365.0 * days - 2.0 * 355.0 / 365.0))
    t_cold = 10.0 + 7.0 * np.cos(math.pi * (2.0 / 365.0 * days - 2.0 * 225.0 / 365.0))
    temp_dt_hourly = np.repeat(t_mixed - t_cold, 24)
    if time_resolution == 3600:
        result = temp_dt_hourly
    else:
        result = np.asarray(
            chres.changeResolution(temp_dt_hourly.tolist(), 3600, time_resolution, "mean"),
            dtype=np.float64,
        )
    _DHW_TEMP_DT_CACHE[time_resolution] = result
    return result


def _get_cached_opendhw_cdf(building_type, weekend_weekday_factor, holidays):
    """Return cached deterministic 1-minute yearly CDF matching OpenDHW.generate_yearly_probability_profile."""
    key = (building_type, float(weekend_weekday_factor), tuple(holidays))
    cached = _DHW_CDF_CACHE.get(key)
    if cached is not None:
        return cached

    s_step = 60
    p_we = OpenDHW.generate_daily_probability_step_function(
        mode="off-day", s_step=s_step, building_type=building_type
    )
    p_wd = OpenDHW.generate_daily_probability_step_function(
        mode="work-day", s_step=s_step, building_type=building_type
    )
    p_wd_w, p_we_w, _ = OpenDHW.shift_weekend_weekday(p_wd, p_we, weekend_weekday_factor)
    p_wd_arr = np.asarray(p_wd_w, dtype=np.float64)
    p_we_arr = np.asarray(p_we_w, dtype=np.float64)

    hol_set = set(holidays)
    days = np.arange(365, dtype=np.float64)
    season = 1.0 + 0.1 * np.cos(math.pi * (2.0 / 365.0 * days - 0.25))

    p_final = np.empty((365, 1440), dtype=np.float64)
    for day in range(365):
        cur = day % 7
        if building_type in ("SFH", "MFH", "TH", "AB", "OB"):
            is_off = (cur in (5, 6)) or ((day + 1) in hol_set)
        elif building_type == "SC":
            is_off = (cur in (5, 6)) or ((day + 1) in hol_set) or (182 <= day + 1 <= 212)
        elif building_type == "GS":
            is_off = (cur == 6) or ((day + 1) in hol_set)
        else:
            is_off = False
        p_final[day] = (p_we_arr if is_off else p_wd_arr) * season[day]

    p_flat = p_final.ravel()
    p_cdf = np.cumsum(p_flat / p_flat.sum())
    result = (p_cdf, float(p_cdf[0]), float(p_cdf[-1]))
    _DHW_CDF_CACHE[key] = result
    return result

class Profiles:
    """
    Profile class.
    Calculating user related profiles of a building or flat.

    Parameters
    ----------
    number_occupants : integer
        Number of occupants who live in the house or flat.
    number_occupants_building : integer
        Number of occupants who live in the building.
    initial_day : integer
        Day of the week with which the generation starts.
        1-7 for monday-sunday.
    nb_days : integer
        Number of days for which a stochastic profile is generated.
    time_resolution : integer
        resolution of time steps of output array in seconds.

    Attributes
    ----------
    activity_profile : array-like
        Numpy-array with active occupants 10-minutes-wise.
    occ_profile : array-like
        Stochastic occupancy profile.
    app_load : array-like
        Electric load profile of appliances in W.
    light_load : array-like
        Electric load profile of lighting in W.
    """

    def __init__(self, number_occupants, number_occupants_building, initial_day, nb_days, time_resolution, building):
        """
        Constructor of Profiles class.

        Returns
        -------
        None.
        """

        self.number_occupants = number_occupants
        self.number_occupants_building = number_occupants_building
        self.initial_day = initial_day
        self.nb_days = nb_days
        self.time_resolution = time_resolution

        self.building = building
        self.activity_profile = []
        self.occ_profile = []
        self.occ_profile_building = []
        self.building_profiles = {}
        self.temperature_difference = []
        self.light_load = []
        self.app_load = []

        self.generate_activity_profile_residential()

    def generate_activity_profile_residential(self):
        """
        Generate a stochastic activity profile
        (on base of ridchardsonpy).

        Parameters
        ----------
        number_occupants : integer
            Number of occupants who live in the house or flat.
        initial_day : integer
            Day of the week with which the generation starts
            1-7 for monday-sunday.
        nb_days : integer
            Number of days for which a stochastic profile is generated.

        Returns
        -------
        None.
        """

        if self.building in {"SFH", "TH", "MFH", "AB"}:
            activity = occ_residential.Occupancy(self.number_occupants, self.initial_day, self.nb_days)
            self.activity_profile = activity.occupancy

    def generate_occupancy_profiles_residential(self):
        """
        Generate stochastic occupancy profiles for a district for calculating internal gains.
        Change time resolution of 10 min profiles to required resolution.

        Parameters
        ----------
        time_resolution : integer
            Resolution of time steps of output array in seconds.
        activity_profile : array-like
            Numpy-arry with active occupants 10-minutes-wise.

        Returns
        -------
        self.occ_profile : array-like
            Number of present occupants.
        """

        tr_min = int(self.time_resolution/60)
        sia_profile_daily_min = np.concatenate((np.ones(60*8),
                                                np.zeros(60*13),
                                                np.ones(60*3)),
                                                axis=None)

        # Vectorized equivalent of the 525,600-minute loop (identical result, ~1000x faster)
        activity_profile_arr = np.asarray(self.activity_profile)
        n_min = len(activity_profile_arr) * 10
        n_days_needed = math.ceil(n_min / len(sia_profile_daily_min))
        sia_profile = np.tile(sia_profile_daily_min, n_days_needed)[:n_min] * np.max(activity_profile_arr)

        activity_profile_min = np.maximum(np.repeat(activity_profile_arr, 10), sia_profile)
        n_steps = int(n_min / tr_min)
        self.occ_profile = np.round(activity_profile_min[: n_steps * tr_min].reshape(n_steps, tr_min).mean(axis=1))

        return self.occ_profile

    def loadProbabilitiesDhw(self):
        """
        Load probabilities of dhw usage.

        Returns
        -------
        None.
        """

        #  Define src path
        src_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        filename = 'dhw_stochastical.xlsx'
        path_DHW = os.path.join(src_path, 'districtgenerator', 'data', filename)

        # Initialization
        profiles = {"we": {}, "wd": {}}
        # book = xlrd.open_workbook(filename)
        book = xl.readxl(fn=path_DHW)
        sheetnames = book.ws_names

        # Iterate over all sheets
        for sheetname in sheetnames:
            # sheet = xl.readxl(fn=filename, ws=sheetname)

            # Read values
            values = [book.ws(ws=sheetname).index(row=i, col=1) for i in
                      range(1, 1441)]  # [sheet.cell_value(i,0) for i in range(1440)]

            # Store values in dictionary
            if sheetname in ("wd_mw", "we_mw"):
                profiles[sheetname] = np.array(values)
            elif sheetname[1] == "e":
                profiles["we"][int(sheetname[2])] = np.array(values)
            else:
                profiles["wd"][int(sheetname[2])] = np.array(values)

        # Load profiles
        self.prob_profiles_dhw = profiles

    def generate_dhw_profile(self, building, holidays):
        """
        Generate a stochastic dhw profile
        (on base of DHWclac).
        https://www.uni-kassel.de/maschinenbau/institute/thermische-energietechnik/fachgebiete/solar-und-anlagentechnik/downloads
        https://github.com/RWTH-EBC/OpenDHW

        Parameters
        ----------
        s_step : integer
            Resolution of time steps of output array in seconds.
        Categories: 1 or 4
            Either one or four categories with different mean volume rates, tapping times and frequencies can be defined.
        occupancy: integer
            Maximum number of occupants in this building.
        mean_drawoff_vol_per_day : array-like
            Total mean daily draw-off volume per person per day in liter.
        temp_dT : array-like
        The temperature difference (ΔT) between the cold water and the water at the tapping point (mixed water).

        Returns
        -------
        dhw_heat : array-like
            Numpy-array with heat demand of dhw consumption in W.
        """

        self.temperature_difference = _get_cached_dhw_temp_dt(self.time_resolution)

        is_residential = self.building in {"SFH", "TH", "MFH", "AB"}
        occupancy = self.number_occupants if is_residential else self.number_occupants_building
        weekend_weekday_factor = 1.2 if is_residential else 1
        mean_drawoff_vol_per_day = building["buildingFeatures"]["mean_drawoff_dhw"]

        # Fast exact equivalent of OpenDHW (s_step=60, categories=1) for residential buildings.
        # Uses cached yearly CDF, exact same random draws, np.searchsorted placement, and direct
        # bucket aggregation (matches OpenDHW bit-for-bit under the same RNG seed, >100x faster).
        if is_residential and self.time_resolution >= 60 and self.time_resolution % 60 == 0:
            steps_per_bucket = int(self.time_resolution // 60)
            if 525600 % steps_per_bucket == 0:
                p_cdf, min_rand, max_rand = _get_cached_opendhw_cdf(
                    self.building, weekend_weekday_factor, holidays
                )

                # Category-1 residential parameters (exact match to OpenDHW.get_data_drawoff_categories)
                mu = 480.0
                sig = 120.0
                low_lim = 240.0   # max(mu - 2*sig, min_flow=6.0)
                up_lim = 720.0    # min(mu + 2*sig, max_flow=1200.0)
                max_flow = 1200.0
                v_max = float(mean_drawoff_vol_per_day) * float(occupancy) * 365.0

                v_curr = 0.0
                drawoffs = []
                gauss = random.gauss
                while v_curr <= v_max:
                    d = gauss(mu, sig)
                    while d < low_lim or d > up_lim:
                        d = gauss(mu, sig)
                    d = 6 * round(d / 6)
                    drawoffs.append(d)
                    v_curr += d / 60.0

                unif = random.uniform
                p_drawoffs = [unif(min_rand, max_rand) for _ in range(len(drawoffs))]
                p_drawoffs.sort()

                # Place each 1-minute drawoff at the earliest minute where p_cdf[t] > p_drawoff
                # without exceeding max_flow (monotonically non-decreasing t, O(1) extra memory)
                idxs = np.searchsorted(p_cdf, p_drawoffs, side="right")
                n_buckets = 525600 // steps_per_bucket
                water_agg = np.zeros(n_buckets, dtype=np.float64)
                n_steps = 525600
                prev_t = -1
                curr_min_flow = 0.0

                for k in range(len(drawoffs)):
                    t = int(idxs[k])
                    if t < prev_t:
                        t = prev_t
                    d = drawoffs[k]
                    if t == prev_t:
                        if curr_min_flow + d > max_flow:
                            t += 1
                            curr_min_flow = 0.0
                    else:
                        curr_min_flow = 0.0
                    if t >= n_steps:
                        break
                    curr_min_flow += d
                    prev_t = t
                    water_agg[t // steps_per_bucket] += d

                water_l_per_h = water_agg / float(steps_per_bucket)
                # Exact heat formula from OpenDHW.compute_heat (rho=0.98 kg/L, cp=4180 J/kgK)
                return water_l_per_h / 3600.0 * (980.0 / 1000.0) * 4180.0 * self.temperature_difference

        # Fallback to upstream OpenDHW for any non-standard configuration
        dhw_profile = OpenDHW.generate_dhw_profile(
            s_step=60,
            categories=1,
            occupancy=occupancy,
            building_type=self.building,
            weekend_weekday_factor=weekend_weekday_factor,
            holidays=holidays,
            mean_drawoff_vol_per_day=mean_drawoff_vol_per_day,
        )

        dhw_timeseries = OpenDHW.resample_water_series(dhw_profile, self.time_resolution)
        dhw_heat = OpenDHW.compute_heat(timeseries_df=dhw_timeseries, temp_dT=self.temperature_difference)

        return dhw_heat["Heat_W"].values

    def generate_el_profile_residential(self, holidays, irradiance, el_wrapper, annual_demand, do_normalization=True):
        """
        Generate electric load profile for one household efficiently.
        """
        # Check time resolution compatibility
        timesteps_irr = int(self.nb_days * 3600 * 24 / len(irradiance))
        if self.time_resolution != timesteps_irr:
            msg = f'Time discretization mismatch: {self.time_resolution} vs {timesteps_irr} seconds.'
            raise AssertionError(msg)

        _timestep_rich = 60  # 1-minute resolution

        # Precompute entire interpolated irradiance to 1-minute resolution
        time_old = np.arange(len(irradiance)) * self.time_resolution
        time_new = np.arange(0, len(irradiance) * self.time_resolution, _timestep_rich)
        irradiance_minutewise = np.interp(time_new, time_old, irradiance)

        # Verify interpolated data length matches expected days
        expected_minutes = self.nb_days * 1440
        if len(irradiance_minutewise) != expected_minutes:
            raise ValueError("Interpolated irradiance length mismatch.")

        # Preallocate arrays for results
        n_days = self.nb_days
        demand = np.zeros((n_days, 1440))
        light_load = np.zeros((n_days, 1440))
        app_load = np.zeros((n_days, 1440))

        for i in range(n_days):
            # Determine if non-working day
            day_check = i + self.initial_day
            not_working_day = (day_check % 7 in (0, 6)) or (day_check in holidays)

            # Slice precomputed 1-minute irradiance for the day
            start, end = i * 1440, (i + 1) * 1440
            irrad_day = irradiance_minutewise[start:end]

            # Extract 10-minute occupancy data
            occ_start, occ_end = 144 * i, 144 * (i + 1)
            current_occupancy = self.activity_profile[occ_start:occ_end]

            # Simulate power usage
            el_p_curve, light_p_curve, app_p_curve = el_wrapper.power_sim(
                irradiation=irrad_day,
                weekend=not_working_day,
                day=i,
                occupancy=current_occupancy
            )

            # Store results directly in preallocated arrays
            demand[i] = el_p_curve
            light_load[i] = light_p_curve
            app_load[i] = app_p_curve

        # Flatten arrays and adjust resolution
        res = demand.ravel()
        self.light_load = light_load.ravel()
        self.app_load = app_load.ravel()

        # Adjust time resolution
        loadcurve = cr.change_resolution(res, _timestep_rich, self.time_resolution)
        self.light_load = cr.change_resolution(self.light_load, _timestep_rich, self.time_resolution)
        self.app_load = cr.change_resolution(self.app_load, _timestep_rich, self.time_resolution)

        # Normalize to annual demand if required
        if do_normalization:
            # Convert to energy (Ws) and sum
            energy_light = self.light_load * self.time_resolution
            energy_app = self.app_load * self.time_resolution

            # Calculate factors using vectorized operations
            total_days = self.nb_days / 365
            curr_light = np.sum(energy_light) / 3_600_000
            curr_app = np.sum(energy_app) / 3_600_000

            factor_light = (0.1 * annual_demand * total_days) / curr_light
            factor_app = (0.9 * annual_demand * total_days) / curr_app

            # Apply normalization factors
            self.light_load *= factor_light
            self.app_load *= factor_app
            loadcurve = self.light_load + self.app_load

        return loadcurve


    def generate_gain_profile_residential(self):
        """
        Generate profile of internal gains

        Parameters
        -------
        personGain : float
            Heat dissipation of one person
            Source: SIA 2024/2015 D - Raumnutzungsdaten für Energie- und Gebäudetechnik
        lightGain : float
            share of waste heat (LED)
            Source: Elsland, Rainer ; Peksen, Ilhan ; Wietschel, Martin: Are Internal Heat
            Gains Underestimated in Thermal Performance Evaluation of Buildings? In: Energy Procedia
            62 (2014), January, 32–41.
        appGain :
            share of waste heat (assumed)
            Source: Elsland, Rainer ; Peksen, Ilhan ; Wietschel, Martin: Are Internal Heat
            Gains Underestimated in Thermal Performance Evaluation of Buildings? In: Energy Procedia
            62 (2014), January, 32–41.
        occ_profile : float
             stochastic occupancy profiles for a district.
        app_load : array-like
            Electric load profile of appliances in W.
        light_load : array-like
            Electric load profile of lighting in W.

        Returns
        -------
        gains : array-like
            Internal gain of each flat.
        """

        personGain = 70.0  # [Watt]
        lightGain = 0.65
        appGain = 0.33

        gains = self.occ_profile * personGain + self.light_load * lightGain + self.app_load * appGain

        return gains
