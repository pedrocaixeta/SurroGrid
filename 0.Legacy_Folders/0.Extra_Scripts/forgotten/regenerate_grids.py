#!/usr/bin/env python3
import os
import sys
import argparse
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from pyproj import Transformer

# Add src and current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import src.db_read as dbrd
import src.save_grid as svgrd
import src.grid_topol as grdtpl
import src.weather as wth

def main():
    parser = argparse.ArgumentParser(description="Regenerate grid sampling output for specific indexes.")
    parser.add_argument("indexes", type=int, nargs="+", help="One or more grid indexes to regenerate.")
    args = parser.parse_args()

    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print("Reconstructing df_sampled_grids...")
    # Load census
    df_census = pd.read_csv("input_data/Zensus2022_Bevoelkerungszahl_1km-Gitter.csv", sep=";")
    probabilities = df_census['Einwohner'] / df_census['Einwohner'].sum()
    df_sampled_cells = df_census.sample(n=1500, weights=probabilities, replace=True, random_state=3)
    df_grid_set = pd.read_hdf("input_data/valid_grids")

    def assign_grid(x_cell, y_cell, N_inh, df_grids):
        df_g = df_grids.copy()
        df_same_cell = df_g[(df_g["x_census"]==x_cell) & (df_g["y_census"]==y_cell)]
        if len(df_same_cell)!=0:
            sample_grid = df_same_cell.sample(n=1, random_state=x_cell)
            sample_grid.drop(columns=["x_census", "y_census"], inplace=True)
            return sample_grid
        
        df_g['diff'] = (df_g['Einwohner'] - N_inh).abs()
        min_diff = df_g['diff'].min()
        closest_rows = df_g[df_g['diff'] == min_diff].drop(columns='diff')
        sample_grid = closest_rows.sample(n=1, random_state=x_cell)
        sample_grid[["x","y"]] = (x_cell, y_cell)
        sample_grid.drop(columns=["x_census", "y_census"], inplace=True)
        return sample_grid

    df_sampled_grids = df_sampled_cells.apply(lambda row: assign_grid(row["x_mp_1km"], row["y_mp_1km"], row["Einwohner"], df_grid_set), axis=1)
    df_sampled_grids = pd.concat(df_sampled_grids.values, ignore_index=True).rename(columns={"plz":"plz_pylovo", "Einwohner":"pop_density_1km2_cell"})

    # Projections and geo-lookups to match Notebook 2 filters
    transformer = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
    df_sampled_grids[["lon","lat"]] = df_sampled_grids.apply(lambda row: transformer.transform(row["x"], row["y"]), axis=1).apply(pd.Series)

    shapefile_path = "input_data/plz-5stellig.shp"
    gdf_zip = gpd.read_file(shapefile_path)

    def lookup_zip_code(lon, lat, gdf_zip):
        point = Point(lon, lat)
        if gdf_zip.crs is None:
            gdf_zip.set_crs(epsg=4326, inplace=True)
        elif gdf_zip.crs.to_epsg() != 4326:
            gdf_zip = gdf_zip.to_crs(epsg=4326)
        matching_row = gdf_zip[gdf_zip.contains(point)]
        if not matching_row.empty: return matching_row.iloc[0]["plz"]
        else: return np.nan

    df_sampled_grids["plz"] = df_sampled_grids.apply(lambda row: lookup_zip_code(row["lon"], row["lat"], gdf_zip), axis=1)
    df_sampled_grids = df_sampled_grids.dropna(subset=['plz'])

    MUNICI_SHP = 'input_data/VG250_GEM.shp'
    gdf_munic = gpd.read_file(MUNICI_SHP)
    if gdf_munic.crs.to_epsg() != 4326:
        gdf_munic = gdf_munic.to_crs(epsg=4326)
    sindex = gdf_munic.sindex

    def lookup_gemeindeschluessel(lat, lon):
        pt = Point(lon, lat)
        idx_candidates = list(sindex.intersection(pt.bounds))
        candidates = gdf_munic.iloc[idx_candidates]
        match = candidates[candidates.contains(pt)]
        if match.empty:
            return None
        row = match.iloc[0]
        return row['AGS'], row['GEN']

    df_sampled_grids[["gemeindeschlüssel", "name"]] = df_sampled_grids.apply(lambda row: lookup_gemeindeschluessel(row["lat"], row["lon"]), axis=1).apply(pd.Series)
    df_sampled_grids = df_sampled_grids.dropna(subset=['gemeindeschlüssel'])

    df_regiostar = pd.read_excel("input_data/regiostar-referenzdateien.xlsx", sheet_name="ReferenzGebietsstand2020")
    def get_regiostar_region(AGS, df_regiostar):
        row = df_regiostar[df_regiostar["gem_20"]==AGS]
        try: return int(row["RegioStaR7"].values[0])
        except: return np.nan

    df_sampled_grids["regio7"] = df_sampled_grids.apply(lambda row: get_regiostar_region(int(row["gemeindeschlüssel"]), df_regiostar), axis=1)
    df_sampled_grids = df_sampled_grids.dropna(subset=['regio7'])

    df_sampled_grids = df_sampled_grids.drop_duplicates().reset_index(drop=True)

    print("Initializing Database connection...")
    DB = dbrd.DataBase()

    # Process each requested index
    for idx in args.indexes:
        if idx < 0 or idx >= len(df_sampled_grids):
            print(f"Error: Index {idx} is out of bounds (0 to {len(df_sampled_grids)-1})")
            continue

        print(f"\n--- Processing Index {idx} ---")
        df_region_specs = df_sampled_grids.iloc[idx].copy()
        
        grid_specs = {
            "cell_id": f"N{int(df_region_specs['y'])}E{int(df_region_specs['x'])}",
            "plz":  df_region_specs["plz_pylovo"],
            "kcid": df_region_specs["kcid"],
            "bcid": df_region_specs["bcid"]
        }

        print(f"Retrieving pylovo grid data for cell_id: {grid_specs['cell_id']}, plz: {grid_specs['plz']}, kcid: {grid_specs['kcid']}, bcid: {grid_specs['bcid']}...")
        net = DB.read_single_ppgrid(grid_specs)
        net = grdtpl.assign_min_linelen(net)
        net = grdtpl.remove_duplicate_loads(net)

        print("Reading buildings...")
        df_buildings = DB.read_buildings(grid_specs, net.bus)

        print("Fetching weather and soil temp data...")
        df_weather_raw, altitude, selected_months = wth.get_pvgis_tmy_sarah3_dataframe(df_region_specs["lat"], df_region_specs["lon"])
        df_weather_raw["dew_point"] = wth.get_dew_point(df_weather_raw["temp_air"], df_weather_raw["relative_humidity"])
        df_weather_raw["soil_temp"] = wth.get_open_meteo_soil_temperature(df_region_specs["lat"], df_region_specs["lon"], selected_months)

        df_region_specs["altitude"] = altitude

        print("Saving grid to H5...")
        SF = svgrd.SaveFile(grid_specs)
        SF.save_topology(net, "/raw_data/")
        SF.save_df(df_region_specs, "/raw_data/region")
        SF.save_df(df_buildings, "/raw_data/buildings")
        SF.save_df(df_weather_raw, "/raw_data/weather")

        # Rename the file to include the index prefix
        original_path = SF.path
        new_filename = f"{idx}_{os.path.basename(original_path)}"
        new_path = os.path.join(os.path.dirname(original_path), new_filename)
        if os.path.exists(new_path):
            os.remove(new_path)
        os.rename(original_path, new_path)
        print(f"Successfully generated and saved prefixed grid: {new_path}")

if __name__ == "__main__":
    main()
