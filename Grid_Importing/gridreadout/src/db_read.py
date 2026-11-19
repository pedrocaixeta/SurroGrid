"""Readout helpers for the pylovo grid database.

This module provides the `DataBase` class, a thin SQLAlchemy wrapper around a
PostgreSQL database that stores:

- pandapower grid JSON blobs
- transformer positions / grid identifiers (PLZ/KCID/BCID)
- building attributes used to map buildings to consumer buses

Connection credentials are taken from environment variables loaded in
`gridreadout/config.py`.
"""

from config import config

from sqlalchemy import create_engine, text
from sqlalchemy.exc import InterfaceError, OperationalError
import pandapower as pp
from pyproj import Transformer

import pandas as pd
import heapq
import json
import re
import time

class DataBase:
    def __init__(self):
        self.connection_settings = {
            "host": config.DB_HOST,
            "port": config.DB_PORT,
            "name": config.DB_NAME,
            "user": config.DB_USER,
            "password": config.DB_PASSWORD
        }
        self.schema = "pylovo"  # Default schema to look for results
        self.engine = self._get_engine()


    def _get_engine(self):
        """ Establish connection engine to database """
        # Access the database settings
        host = self.connection_settings["host"]
        port = self.connection_settings["port"]
        name = self.connection_settings["name"]
        user = self.connection_settings["user"]
        password = self.connection_settings["password"]

        # Create a database engine using SQLAlchemy with pre-ping, bounded pool size,
        # and TCP keepalives so parallel/long-running jobs recover cleanly from timeouts.
        engine = create_engine(
            f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}',
            pool_pre_ping=True,
            pool_recycle=1800,
            pool_size=2,
            max_overflow=2,
            connect_args={
                "connect_timeout": 15,
                "keepalives": 1,
                "keepalives_idle": 30,
                "keepalives_interval": 10,
                "keepalives_count": 5,
            },
        )

        return engine


    def _execute_read(self, query_fn, max_retries=5):
        """Run a read-only database operation with pool disposal and retry on connection timeouts."""
        for attempt in range(max_retries):
            try:
                return query_fn()
            except (OperationalError, InterfaceError) as exc:
                # Dispose dead pooled connections before retrying
                self.engine.dispose()
                if attempt == max_retries - 1:
                    raise
                sleep_time = min(5 * (2 ** attempt), 60)
                print(
                    f"Database connection error ({exc.__class__.__name__}). "
                    f"Retrying in {sleep_time}s (attempt {attempt + 1}/{max_retries - 1})..."
                )
                time.sleep(sleep_time)


    def _read_sql_with_retry(self, query, params=None):
        """Execute a read-only SQL query into a DataFrame with automatic reconnect retries."""
        def _run():
            with self.engine.connect() as conn:
                return pd.read_sql(query, conn, params=params)

        return self._execute_read(_run)


    def show_contents(self):
        """ Show all database sheets """
        with self.engine.connect() as conn:
            result = conn.execute(text(f"SELECT table_name FROM information_schema.tables WHERE table_schema = '{self.schema}';"))
            tables = [row[0] for row in result]
            print("Available tables:")
            print(tables)


    def read_all_grid_identifiers(self):
        """ Retrieve all grids' (PLZ, kcid, bcid, location) stored in the grids database.

        Queries `transformer_positions_with_grid` (a view joining `transformer_positions`
        and `grid_result`) so every grid in `grid_result` is returned together with its
        transformer geometry without filtering out filler grids.
        """

        query = text(f"""
            SELECT plz, kcid, bcid, ST_AsText(geom) as loc
            FROM {self.schema}.transformer_positions_with_grid;
        """)
        # Execute the query with Pandas. This will only read data.
        df_generated_grids = self._read_sql_with_retry(query)
        print(f"Retrieved {len(df_generated_grids)} generated grids!")

        return df_generated_grids

    def read_grids_exceeding_trafo_distance(self, max_distance_km=2.5):
        """Return grids where max shortest-path distance from LVbus to Consumer Nodebus exceeds `max_distance_km`.

        Reproduces the `pylovo-classify` `THRESHOLD_MAX_TRAFO_DIS` filter (`max_trafo_dis > 2.5` km)
        directly from `pandapower_bus` and `pandapower_line` so all grids in `grid_result` are covered.
        """
        # Mathematical lower-bound pre-filter: a shortest path cannot exceed total cable length
        buses_query = text(f"""
            WITH candidates AS (
                SELECT grid_result_id
                FROM {self.schema}.pandapower_line
                GROUP BY grid_result_id
                HAVING SUM(length_km) > :max_dist
            )
            SELECT b.grid_result_id, b.pp_index, b.name
            FROM {self.schema}.pandapower_bus b
            JOIN candidates c ON b.grid_result_id = c.grid_result_id
            WHERE b.name LIKE '%LVbus%' OR b.name LIKE '%Consumer Nodebus%';
        """)
        lines_query = text(f"""
            WITH candidates AS (
                SELECT grid_result_id
                FROM {self.schema}.pandapower_line
                GROUP BY grid_result_id
                HAVING SUM(length_km) > :max_dist
            )
            SELECT l.grid_result_id, l.from_bus, l.to_bus, l.length_km
            FROM {self.schema}.pandapower_line l
            JOIN candidates c ON l.grid_result_id = c.grid_result_id;
        """)

        def _fetch_bus_and_line_rows():
            with self.engine.connect() as conn:
                b_rows = conn.execute(buses_query, {"max_dist": float(max_distance_km)}).fetchall()
                l_rows = conn.execute(lines_query, {"max_dist": float(max_distance_km)}).fetchall()
                return b_rows, l_rows

        bus_rows, line_rows = self._execute_read(_fetch_bus_and_line_rows)

        roots = {}
        consumers = {}
        for gid, pp_index, name in bus_rows:
            if "LVbus" in name and gid not in roots:
                roots[gid] = pp_index
            if "Consumer Nodebus" in name:
                consumers.setdefault(gid, []).append(pp_index)

        adj_all = {}
        for gid, u, v, w in line_rows:
            g = adj_all.setdefault(gid, {})
            g.setdefault(u, []).append((v, w))
            g.setdefault(v, []).append((u, w))

        exceeding = {}
        for gid, adj in adj_all.items():
            root = roots.get(gid)
            cons = consumers.get(gid)
            if root is None or not cons:
                continue

            # Shortest-path distance from LVbus root (matches nx.dijkstra_path_length in pylovo)
            dist = {root: 0.0}
            pq = [(0.0, root)]
            while pq:
                d, u = heapq.heappop(pq)
                if d > dist[u]:
                    continue
                for v, w in adj.get(u, ()):
                    nd = d + w
                    if nd < dist.get(v, float("inf")):
                        dist[v] = nd
                        heapq.heappush(pq, (nd, v))

            max_trafo_dis = max((dist[c] for c in cons if c in dist), default=0.0)
            if max_trafo_dis > max_distance_km:
                exceeding[gid] = max_trafo_dis

        if not exceeding:
            return pd.DataFrame(columns=["plz", "kcid", "bcid", "max_trafo_dis"])

        ids_query = text(f"""
            SELECT grid_result_id, plz, kcid, bcid
            FROM {self.schema}.transformer_positions_with_grid
            WHERE grid_result_id = ANY(:gids);
        """)
        def _fetch_id_rows():
            with self.engine.connect() as conn:
                return conn.execute(ids_query, {"gids": list(exceeding.keys())}).fetchall()

        id_rows = self._execute_read(_fetch_id_rows)

        records = [
            {"plz": plz, "kcid": kcid, "bcid": bcid, "max_trafo_dis": exceeding[gid]}
            for gid, plz, kcid, bcid in id_rows
        ]
        return pd.DataFrame(records)

    def read_single_ppgrid(self, grid_specs):
        """ Reads out single database grid based on plz, kcid, bcid
            
            Args:
                engine: SQLAlchemy engine to connect to database
                grid_specs: dict including plz, kcid, bcid

            Returns:
                net: pandapower grid topology (with lines, transformer, buses)
        """

        query = text(f"""
            SELECT grid
            FROM {self.schema}.grid_result
            WHERE (plz= :plz) AND (kcid= :kcid) AND (bcid= :bcid);
        """)

        # Execute the query with Pandas. This will only read data.
        df_grid = self._read_sql_with_retry(
            query,
            params={"plz": int(grid_specs["plz"]), "kcid": int(grid_specs["kcid"]), "bcid": int(grid_specs["bcid"])},
        )

        # Transform to pandapower net
        net = pp.from_json_string(json.dumps(df_grid.loc[0, "grid"]))

        return net


    def read_trafo_pos(self, grid_specs):
        """ Read out position of transformer position for given grid from database """

        query = text(f"""
            SELECT ST_AsText(geom) as loc
            FROM {self.schema}.transformer_positions_with_grid
            WHERE (plz= :plz) AND (kcid= :kcid) AND (bcid= :bcid);
        """)

        # Execute the query with Pandas. This will only read data.
        df_trafo = self._read_sql_with_retry(
            query,
            params={"plz": int(grid_specs["plz"]), "kcid": int(grid_specs["kcid"]), "bcid": int(grid_specs["bcid"])},
        )

        # Read out location string
        loc = df_trafo.loc[0, "loc"]
        match = re.match(r"POINT\(([-+]?[0-9]*\.?[0-9]+)\s*([-+]?[0-9]*\.?[0-9]+)\)", loc)
        if match:
            x = float(match.group(1))
            y = float(match.group(2))

        # Define the projections
        transformer = Transformer.from_crs(config.PYLOVO_COORD_FORMAT, config.TARGET_COORD_FORMAT, always_xy=True)
        # Convert from EPSG:3857 to EPSG:4326
        lon, lat = transformer.transform(x, y)
        # print(f"EPSG-4326:")
        # print(f"lat: {lat}, lon: {lon}")

        trafo_pos = {
            "lat": lat,
            "lon": lon
        }

        return trafo_pos
    
    def read_regional_stats(self, plz):
        query = text(f"""
            SELECT plz, pop, area, name_city, pop_den, regio7
            FROM {self.schema}.municipal_register
            WHERE plz=:plz;
        """)

        # Execute the query with Pandas. This will only read data.
        df_region_specs = self._read_sql_with_retry(query, params={"plz": int(plz)})

        return df_region_specs


    def read_buildings(self, grid_specs, df_bus): #r.occupants, r.constructi, r.refurb_wal, r.refurb_roo, r.refurb_bas, r.refurb_win
        # query = text("""
        #     SELECT b.osm_id, b.vertice_id, b.type, r.use, b.houses_per_building, r.occupants, r.free_walls, b.floors, r.constructi, b.area, ST_AsText(b.center) as center, r.refurb_wal, r.refurb_roo, r.refurb_bas, r.refurb_win
        #     FROM public.buildings_result b
        #     RIGHT OUTER JOIN public.res r 
        #     ON b.osm_id = r.osm_id
        #     WHERE (b.plz= :plz) AND (b.kcid= :kcid) AND (b.bcid= :bcid);
        # """)

        query = text(f''' 
            SELECT 
                br.vertice_id, 
                br.type,
                br.height,
                br.building_use AS use,
                br.households AS houses_per_building,
                br.occupants,
                br.floor_number AS floors, 
                br.residential_floor_area,
                br.nonresidential_floor_area,
                br.nonresidential_use,
                br.building_use_id,
                br.building_type,
                br.construction_year AS constructi,
                br.floor_area AS area, 
                ST_AsText(br.centroid) AS center,
                br.connection_point
            FROM {self.schema}.buildings_result_with_grid br
            WHERE 
                br.plz = :plz
                AND br.kcid = :kcid
                AND br.bcid = :bcid;
        ''')

        df_buildings = self._read_sql_with_retry(
            query,
            params={"plz": int(grid_specs["plz"]), "kcid": int(grid_specs["kcid"]), "bcid": int(grid_specs["bcid"])},
        )
        if df_buildings.empty:
            raise ValueError(
                f"No buildings found for PLZ={grid_specs['plz']}, KCID={grid_specs['kcid']}, BCID={grid_specs['bcid']}."
            )
        ### Match bus to building
        df_id = pd.DataFrame()
        df_id["vertice_id"] = df_bus['name'].str.extract(r'^Consumer Nodebus (\d+)$')[0].dropna().astype(int)
        df_id = df_id.reset_index().rename(columns={"index":"bus"})
        df_buildings = df_buildings.merge(df_id, on='vertice_id', how="left")
        if "connection_point" in df_buildings.columns: # Fallback to connection_point if bus is NaN
            df_buildings["bus"] = df_buildings["bus"].fillna(df_buildings["connection_point"])
            df_buildings.drop(columns=["connection_point"], inplace=True)

        ### Take bus to front and order by it
        cols = df_buildings.columns.tolist()
        cols.insert(0, cols.pop(cols.index('bus')))
        df_buildings = df_buildings[cols]
        df_buildings = df_buildings.sort_values(by='bus').reset_index(drop=True)

        ### Verification of imported information and use of Fallback values if necessary
        ## Floors
        if "floors" in df_buildings.columns: 
            # Ensure 'floors' is 'height'/3 if 'floors' is missing. If 'height' is also missing, then 'floors' falls back to 1
            if "height" in df_buildings.columns:
                average_floor_height = 3.0 # It is assumed that, in average, every floor (story) is 3m high. 
                height_numeric = pd.to_numeric(df_buildings["height"], errors="coerce") # average height of a floor in meters. Source: German Building Energy Act (GEG § 25) / IWU building typologies????
                estimated_floors = (height_numeric/ average_floor_height).round().fillna(1).clip(lower=1)
                df_buildings["floors"] = df_buildings["floors"].fillna(estimated_floors).clip(lower=1)
            else:
                df_buildings["floors"] = df_buildings["floors"].fillna(1).clip(lower=1)
        ## Households per building
        if "houses_per_building" in df_buildings.columns and "use" in df_buildings.columns:
            # Estimate houses per building if missing
            is_res = df_buildings["use"] == "Residential"
            if "type" in df_buildings.columns and "floors" in df_buildings.columns:
                is_mfh_or_ab = df_buildings["type"].isin(["MFH", "AB"])
                is_sfh_or_th = df_buildings["type"].isin(["SFH", "TH"])
                
                # Assume 2.0 households/floor for MFH/AB, and 1.0 for SFH/TH
                estimated_houses = df_buildings["floors"] * 2.0
                estimated_houses = estimated_houses.where(is_mfh_or_ab, 1.0)
                
                # For unknown types, if floors > 1 assume 2.0 per floor, else 1.0
                unknown_type = ~(is_mfh_or_ab | is_sfh_or_th)
                estimated_houses = estimated_houses.where(
                    ~unknown_type, 
                    (df_buildings["floors"] * 2.0).where(df_buildings["floors"] > 1, 1.0)
                )
                
                df_buildings.loc[is_res, "houses_per_building"] = (
                    df_buildings.loc[is_res, "houses_per_building"]
                    .fillna(estimated_houses)
                    .clip(lower=1)
                )
            else:
                df_buildings.loc[is_res, "houses_per_building"] = (
                    df_buildings.loc[is_res, "houses_per_building"].fillna(1).clip(lower=1)
                )
       
        ## Occupants
        if "occupants" in df_buildings.columns and "use" in df_buildings.columns:
            is_res = df_buildings["use"] == "Residential"
            # Non-residential buildings should not have occupants. In case they have, delete them.
            df_buildings.loc[~is_res, "occupants"] = None
            # Ensures there are at least 1 occupant per building and assumes average of 2 occupants per household otherwise (only for Residential)
            fallback_occ = df_buildings["houses_per_building"] * 2
            df_buildings.loc[is_res, "occupants"] = df_buildings.loc[is_res, "occupants"].fillna(fallback_occ)
            df_buildings.loc[is_res, "occupants"] = df_buildings.loc[is_res, "occupants"].where(
                df_buildings.loc[is_res, "occupants"] >= 1, fallback_occ
            ) # Be Aware: SurroGrid does not use the number of occupants, but the number of households and samples occupants for each household


        ### Read out location from string
        # Build transformer once before row-wise conversion from pylovo DB CRS (EPSG:25832) to WGS84 (EPSG:4326)
        transformer = Transformer.from_crs(config.PYLOVO_COORD_FORMAT, config.TARGET_COORD_FORMAT, always_xy=True)

        def _get_loc(loc_string):
            # Set dummy to Munich
            lat = 48.1351
            lon = 11.5820

            match = re.match(r"POINT\(([-+]?[0-9]*\.?[0-9]+)\s*([-+]?[0-9]*\.?[0-9]+)\)", loc_string)
            if match:
                x = float(match.group(1))
                y = float(match.group(2))
                lon, lat = transformer.transform(x, y)

            return lat, lon

        df_buildings[["lat", "lon"]] = df_buildings["center"].apply(_get_loc).apply(pd.Series)
        df_buildings.drop(columns=["center"], inplace=True)

        return df_buildings