from config import config
import pandas as pd
import numpy as np
from multiprocessing import Pool
import heapq

import src.classes.save_grid as svgrd
import src.functions.weather as wth
import src.functions.electricity as elc
import src.functions.heat as heat


class Grid:
    """Slim Grid processor for tuning refurbishment probabilities (p_max_res, p_max_ghd).

    Skips rooftop PV, OpenDHW hot water, HP COP, mobility, MV disconnection, and pre-powerflow.
    Retains electrical demand generation because internal appliance/lighting heat gains feed
    into the 5R1C space-heating balance.
    """

    def __init__(self, settings):
        self.settings = settings
        self.SF = svgrd.SaveFile(
            settings["grid_filename"],
            input_dir=settings.get("input_dir"),
            output_dir=settings.get("output_dir"),
        )

        # Load raw building, region, and weather tables
        self.df_buildings, self.df_region, self.df_weather_raw = self.SF.get_input_data()

        # Backwards compatibility for older grid schemas without explicit mixed-building floor areas
        if (
            "residential_floor_area" not in self.df_buildings.columns
            and "nonresidential_floor_area" not in self.df_buildings.columns
            and "nonresidential_use" not in self.df_buildings.columns
        ):
            self.df_buildings["residential_floor_area"] = 0.0
            self.df_buildings["nonresidential_floor_area"] = 0.0
            self.df_buildings["nonresidential_use"] = pd.NA

            res_mask = self.df_buildings["use"] == "Residential"
            self.df_buildings.loc[res_mask, "residential_floor_area"] = (
                self.df_buildings.loc[res_mask, "area"] * self.df_buildings.loc[res_mask, "floors"]
            )

            mask_ghd = self.df_buildings["use"].isin(["Commercial", "Public"])
            self.df_buildings.loc[mask_ghd, "nonresidential_floor_area"] = (
                self.df_buildings.loc[mask_ghd, "area"] * self.df_buildings.loc[mask_ghd, "floors"]
            )
            self.df_buildings.loc[mask_ghd, "nonresidential_use"] = self.df_buildings.loc[mask_ghd, "use"]

        self.plz = str(int(self.df_region["plz"])).zfill(5)
        self.location = {
            "lat": self.df_region["lat"],
            "lon": self.df_region["lon"],
        }
        self.altitude = self.df_region["altitude"]

        if not self.settings.get("weather_data_exists", True):
            self.df_weather_raw = pd.DataFrame()
        self.df_demand_elec = pd.DataFrame()
        self.df_demand_elec_res = pd.DataFrame()
        self.df_demand_elec_ghd = pd.DataFrame()
        self.df_demand_heat_space = pd.DataFrame()
        self.df_demand = pd.DataFrame()

    ############################################
    ########### Timeseries Generators ##########
    ############################################
    def retrieve_weather(self):
        if self.settings.get("weather_data_exists", True) and self.df_weather_raw is not None:
            return
        self.df_weather_raw, self.altitude, _ = wth.get_pvgis_tmy_sarah3_dataframe(
            self.location["lat"], self.location["lon"]
        )

    def generate_electricity(self):
        """Sample household occupancy and generate electrical demand profiles for internal heat gains."""
        self.df_buildings = elc.sample_statistics(self.df_buildings)
        self.df_buildings, self.df_demand_elec, self.df_demand_elec_res, self.df_demand_elec_ghd = (
            elc.get_elec_demand(self.df_buildings)
        )

    def generate_heat(self, p_max_res=None, p_max_ghd=None):
        """Sample building construction ages, retrofit levels, and compute space-heating demand."""
        self.df_buildings = heat.sample_statistics(
            self.df_buildings,
            p_max_res=p_max_res,
            p_max_ghd=p_max_ghd,
        )

        # Apply daylight-saving shift to align human occupancy/electrical gains with weather
        df_wth_input = self._add_input_data_daylight_saving_shift(self.df_weather_raw)
        df_elec_input = self._add_input_data_daylight_saving_shift(self.df_demand_elec)
        df_elec_res_input = self._add_input_data_daylight_saving_shift(self.df_demand_elec_res)
        df_elec_ghd_input = self._add_input_data_daylight_saving_shift(self.df_demand_elec_ghd)

        if self.settings.get("parallel", False) and self.settings.get("n_cpu", 1) > 1:
            building_subsets = self.partition_df_by_cpu(
                self.df_buildings, self.settings["n_cpu"], "houses_per_building"
            )
            col_subsets = [[(col, "electricity") for col in subset["bus"].values] for subset in building_subsets]
            job_args = [
                (
                    subset.reset_index(drop=True),
                    df_elec_input[col_subsets[i]],
                    df_elec_res_input[col_subsets[i]],
                    df_elec_ghd_input[col_subsets[i]],
                    [np.array(df_wth_input[col]) for col in ["dni", "dhi", "temp_air"]],
                    self.plz,
                )
                for i, subset in enumerate(building_subsets)
            ]
            with Pool(processes=self.settings["n_cpu"]) as pool:
                results = pool.starmap(heat.generate_heat_demands, job_args)
            self.df_demand_heat_space = pd.concat(results, axis=1).sort_index(axis=1, level=[0])
        else:
            self.df_demand_heat_space = heat.generate_heat_demands(
                self.df_buildings,
                df_elec_input,
                df_elec_res_input,
                df_elec_ghd_input,
                [np.array(df_wth_input[col]) for col in ["dni", "dhi", "temp_air"]],
                self.plz,
            )

        self.df_demand_heat_space = self._add_output_data_daylight_saving_shift(self.df_demand_heat_space)

        # Store annual space-heating energy (kWh) per building directly in df_buildings for fast aggregation
        sh_totals = self.df_demand_heat_space.droplevel(1, axis=1).sum(axis=0).to_dict()
        self.df_buildings["space_heat_kwh"] = [
            float(sh_totals.get(bus, 0.0)) for bus in self.df_buildings["bus"]
        ]

    def create_demand(self):
        """Assemble urbs_in/demand with space_heat time series."""
        self.df_demand = self.df_demand_heat_space.reset_index(drop=True)
        self.df_demand.index.name = "t"

    def save_grid_data(self):
        """Copy raw grid .h5 to STORAGE_DIR, save updated buildings and space-heat demand, and rename."""
        self.SF.copy_save_file()
        self.SF.save_df(self.df_buildings, "raw_data/buildings")
        self.SF.save_df(self.df_demand, "urbs_in/demand")
        final_path = self.SF.append_suffix(self.SF.output_path, suffix=config.OUTPUT_SUFFIX)
        self.SF.output_path = final_path
        return final_path

    ############################################
    ################# Helpers ##################
    ############################################
    @staticmethod
    def _add_input_data_daylight_saving_shift(df_ts):
        if len(df_ts) == 0:
            return df_ts.copy()
        ts_hour1 = 2090
        ts_hour2 = 7130
        df_ts = df_ts.copy()
        df_ts = df_ts.drop(index=ts_hour2).reset_index(drop=True)
        new_row = df_ts.iloc[ts_hour1 - 1].copy()
        new_row_df = pd.DataFrame([new_row], columns=df_ts.columns)
        df_ts = pd.concat([df_ts.iloc[:ts_hour1], new_row_df, df_ts.iloc[ts_hour1:]]).reset_index(drop=True)
        return df_ts

    @staticmethod
    def _add_output_data_daylight_saving_shift(df_ts):
        if len(df_ts) == 0:
            return df_ts.copy()
        ts_hour1 = 2090
        ts_hour2 = 7130
        df_ts = df_ts.copy()
        new_row = df_ts.iloc[ts_hour2].copy()
        new_row_df = pd.DataFrame([new_row], columns=df_ts.columns)
        df_ts = pd.concat([df_ts.iloc[: ts_hour2 + 1], new_row_df, df_ts.iloc[ts_hour2 + 1 :]]).reset_index(drop=True)
        df_ts = df_ts.drop(index=ts_hour1).reset_index(drop=True)
        return df_ts

    @staticmethod
    def partition_df_by_cpu(df: pd.DataFrame, n_cpus: int, count_column: str) -> list[pd.DataFrame]:
        building_list = list(df[count_column].items())
        building_list.sort(key=lambda x: x[1], reverse=True)
        heap: list[tuple[int, int]] = [(0, bin_id) for bin_id in range(n_cpus)]
        heapq.heapify(heap)
        bins_indices: list[list[pd.Index]] = [[] for _ in range(n_cpus)]
        for idx, n_count in building_list:
            weight = max(1, int(n_count)) if not pd.isna(n_count) else 1
            current_load, bin_id = heapq.heappop(heap)
            bins_indices[bin_id].append(idx)
            heapq.heappush(heap, (current_load + weight, bin_id))
        return [df.loc[indices].copy() for indices in bins_indices if indices]
