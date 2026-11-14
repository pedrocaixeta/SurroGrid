# HDF5 File Structure Reference

This document provides a detailed reference for the HDF5 (`.h5`) file structure used as the central data contract between the various steps of the **SurroGrid** pipeline.

---

## 1. What is an HDF5 (.h5) File?

An **HDF5 (Hierarchical Data Format version 5)** file is a file format designed to store and organize large amounts of data. You can think of an HDF5 file like a **local file/folder system** enclosed within a single file (similar to `.zip`). 
The first level of organization inside .h5 are called **Groups**, which act like **directories (folders)**. They are used to create hierarchy (e.g., `/raw_data`, `/urbs_in`, `/pwrflw`). Inside these groups, there are **Datasets**, which act like **files** within those folders. In this project, they typically store multidimensional arrays, tables, or serialized text.

Most of the datasets in this project are stored as compressed tables via `pandas.HDFStore`, meaning they can be loaded directly into Python as `pandas.DataFrame` objects.

---

## 2. Top-Level Directory Tree

Each scenario `.h5` file contains the following group hierarchy, representing the lifecycle of data from initial sampling to final power-flow calculation:

```text
/
├── raw_data/                       # Step 1: Raw network & regional data
├── urbs_in/                        # Step 2: Demand allocation & optimization inputs
├── urbs_out/                       # Step 3: Optimization outputs
└── pwrflw/                         # Step 4: Time-series power flow inputs and outputs
```

---

## 3. Detailed presentation of some Datasets

In the following, some of the Datasets of this project are presented in more detail:

---

### 3.1. Grid Sampling: `raw_data/`
```text
/raw_data/
├── buildings                      # Pandas DataFrame (building attributes)
├── consumers                      # Pandas DataFrame (consumer bus mapping)
├── net                            # JSON String (The pandapower Network!)
├── region                         # Pandas DataFrame (geographic & regional metadata)
└── weather                        # Pandas DataFrame (meteorological timeseries)
```
#### `raw_data/net` - Pandapower Network JSON Representation 

The pandapower network dataset `/raw_data/net` is saved as a UTF-8 encoded JSON string representation of a `pandapowerNet` object. It uses the following structure:

```json
{
  "_module": "pandapower.auxiliary",
  "_class": "pandapowerNet",
  "_object": { // under this the Datasets (tables) are stored
    "bus": { 
      "_module": "pandas.core.frame",
      "_class": "DataFrame",
      "_object": "{\"columns\":[\"name\",\"vn_kv\",\"in_service\",...], \"index\":[0,1,...], \"data\":[[...]]}"
    },
    "line": {
      "_module": "pandas.core.frame",
      "_class": "DataFrame",
      "_object": "{\"columns\":[\"name\",\"from_bus\",\"to_bus\",\"length_km\",\"r_ohm_per_km\",\"x_ohm_per_km\",...], \"index\":[0,1,...], \"data\":[[...]]}"
    }
    // ... more component and result tables are stored under this object
  }
}
```

The JSON contains metadata fields (`_module` and `_class`) describing the class types for deserialization. The core network data is contained under the `_object` key, which stores individual Pandas DataFrames (serialized as nested JSON strings) representing the components and results of the network.


* **Component Tables:** Define the physical topology and constraints of the grid.
  * `bus`: Bus node definitions.
  * `line`: Transmission line specs.
  * `load`: Power consumption connections (demand).
  * `trafo`: Transformers linking voltage levels.
  * `switch`: Network switches.
  * `ext_grid`: External grid slack connections.
  * `sgen`: Static generators (e.g., PV panels).
  * `storage`: Energy storage devices (e.g., batteries, EVs).
* **Result Tables:** Store the results of the power-flow calculation (filled/updated during Step 4.powerflow).
  * `res_bus`, `res_line`, `res_trafo`, `res_ext_grid`, and `res_load`.

Below is a detailed breakdown of the main tables and their respective columns inside the `pandapowerNet` JSON object:

#### A. Bus Table (`bus`)
Stores all physical nodes (busbars and connection points) in the grid.
* `name`: Name or label of the bus node.
* `vn_kv`: Nominal voltage level of the node in kV.
* `type`: Type of bus (e.g., `"b"` for busbar, `"n"` for node).
* `zone`: Grid zone name or identifier.
* `in_service`: Boolean flag indicating if the node is active.
* `geo`: Geographic coordinates/geometry.
* `min_vm_pu` / `max_vm_pu`: Minimum/maximum voltage limits in per-unit (p.u.).

#### B. Line Table (`line`)
Defines the connections and electrical parameters of cables or overhead lines.
* `name`: Name or ID of the line.
* `std_type`: Standard type categorizing the cable/line specs.
* `from_bus` / `to_bus`: Integer indices of the buses connected by the line.
* `length_km`: Physical length of the line in kilometers.
* `r_ohm_per_km`: Resistance per km ($\Omega$/km).
* `x_ohm_per_km`: Reactance per km ($\Omega$/km).
* `c_nf_per_km`: Capacitance per km (nF/km).
* `g_us_per_km`: Conductance per km ($\mu$S/km).
* `max_i_ka`: Thermal current limit in kA.
* `in_service`: Boolean flag.
* `parallel` / `df` / `type`: Paralleling factor, derating factor, and line type.

#### C. Load Table (`load`)
Represents consumers connected to the buses.
* `name`: Name of the load connection.
* `bus`: The Bus ID where the load is attached.
* `p_mw` / `q_mvar`: Active/reactive power draw in MW/MVAr.
* `scaling`: Scaling factor applied to the load values.
* `in_service`: Boolean flag.
* `type`: Type of load category.
* `max_p_mw`: Maximum active power constraint.
* `const_z_p_percent` / `const_i_p_percent` / `const_z_q_percent` / `const_i_q_percent`: Voltage-dependency percentages (impedance and current components).

#### D. Transformer Table (`trafo`)
Describes step-down or step-up transformers.
* `name` / `std_type`: Transformer label and catalog type.
* `hv_bus` / `lv_bus`: Bus IDs connected to the High-Voltage (HV) and Low-Voltage (LV) sides.
* `sn_mva`: Rated apparent power in MVA.
* `vn_hv_kv` / `vn_lv_kv`: Rated voltages on the HV/LV sides.
* `vk_percent`: Short-circuit voltage percentage.
* `vkr_percent`: Real part of short-circuit voltage percentage.
* `pfe_kw`: Iron losses in kW.
* `i0_percent`: Open-loop current percentage.
* `tap_pos` / `tap_neutral` / `tap_min` / `tap_max`: Tap changer settings and position constraints.
* `in_service` / `parallel` / `df` / `vector_group` / `oltc`: Auxiliary operation parameters.

---

### 3.2. Demand Allocation: `urbs_in/demand`


```text
/urbs_in/
├── demand        # MultiIndexed DataFrame (demands for all buses)
├── commodity     # DataFrame (urbs commodity specs)
├── process       # DataFrame (urbs process specs)
├── storage       # DataFrame (urbs storage specs)
└── ...           # (other urbs input tables)
```

The `/urbs_in/demand` dataset is a key table holding hourly demand profiles. It uses a `pandas.MultiIndex` for column headers, where each column is represented as a **tuple of two elements**:

$$\text{Column Header} = (\text{Building ID}, \, \text{Commodity})$$

* **First Element (Level 0):** The building or bus identifier (stored as an `integer` like `112`, `118`, etc.).
* **Second Element (Level 1):** The commodity type, which can be `'electricity'`, `'space_heat'`, `'water_heat'`, `'mobility0'`,`'mobility1'`, or `'mobility2'`... (depending on how many BEVs the building has). Their unit is kW (electricity is therefore active power)

 Because there are columns for every combination of building and commodity, this table contains a very large number of columns. You can think of it as a series of individual timeseries tables (one for electricity, one for space heating, one for water heating, and one for mobility) stacked side-by-side across all buildings in the grid.

Each row in the dataset represents a specific **hourly timestep** of the year (ranging from `0` to `8759`). Below is an example of the `urbs_in/demand` table:

| Timestep | `(112, 'electricity')` | `(112, 'space_heat')` | `(118, 'electricity')` | `(118, 'mobility0')` | ... |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | 1.25 | 4.30 | 0.89 | 0.00 | ... |
| **1** | 1.10 | 4.55 | 0.80 | 1.50 | ... |
| **...** | ... | ... | ... | ... | ... |
| **8759** | 1.15 | 4.10 | 0.95 | 0.00 | ... |

---

### 3.3. Urbs Optimization: `urbs_out`

The `/urbs_out` group stores the results from the **urbs optimization** step (Step 3). It contains three main sub-groups:

* **`/urbs_out/reduced_data/`**: The (possibly TSAM-reduced or node-reduced) input data tables that were actually passed to the solver (e.g., `process`, `storage`, `commodity`, `demand`, etc.).
* **`/urbs_out/tsam/`**: Time-series aggregation metadata. Key datasets include:
  * `kept_timesteps`: A vector listing the actual original timesteps selected as representative.
* **`/urbs_out/MILP/`**: Serialization of the Pyomo MILP entity dumps, merged across parallel solver worker processes. Key datasets include:
  * `costs`: Total and itemized cost components for the optimization model.
  * `cap_pro`, `cap_pro_new`: Active power generation/process capacities (existing and newly expanded).
  * `cap_sto_c`, `cap_sto_c_new`, `cap_sto_p`, `cap_sto_p_new`: Storage capacity expansion results.
  * `e_pro_in`, `e_pro_out`, `e_sto_in`, `e_sto_out`: Hourly active power process and storage input/output energy timeseries.
  * `tau_pro`: Hourly process power timeseries (critical for power flow reconstruction).

```text
/urbs_out/
├── reduced_data/              # Data tables used by the solver (possibly TSAM-reduced)
├── tsam/                      # Time Series Aggregation metadata
│   └── kept_timesteps         # Vector of representative timesteps
└── MILP/                      # Pyomo entity dumps (costs, capacities, and tau_pro)
```

---

### 3.4. Power Flow: `pwrflw`

The `pwrflw` group contains the following subgroups:

```text
/pwrflw/
├── input/
│   ├── demand_pre             # Reconstructed pre expansion P/Q timeseries per bus 
│   └── demand_post            # Reconstructed post urbs expansion P/Q timeseries per bus
└── output/
    ├── pre/                   # Pre-expansion power flow results
    │   ├── demand_import      # Net external grid active/reactive imports
    │   ├── vm                 # Voltage magnitude (p.u.) per bus
    │   └── line_loads         # Loading and currents per line
    └── post/                  # Post-expansion power flow results
        ├── demand_import
        ├── vm
        └── line_loads
```

#### A. Power Flow Inputs (`/pwrflw/input/`)
Reconstructed active ($P$) and reactive ($Q$) demand profiles for every bus in the grid, derived from `urbs_in/demand` (pre) and `urbs_out/MILP/tau_pro` (post):
* **`demand_pre`** (pre-expansion) and **`demand_post`** (post-expansion):
  * **Columns:** MultiIndexed columns of `(Bus ID, Component)` where the component is `'electricity'` (active power in kW) and `'electricity-reactive'` (reactive power in kVAr).
  * **Index:** Timestep integers (`0` to `8759`) representing the hours of the year.
  * **Illustration:**
    | Timestep | `(112, 'electricity')` | `(112, 'electricity-reactive')` | `(118, 'electricity')` | `(118, 'electricity-reactive')` | ... |
    | :--- | :--- | :--- | :--- | :--- | :--- |
    | **0** | 1.250 | 0.411 | 0.890 | 0.293 | ... |
    | **1** | 1.100 | 0.362 | 0.800 | 0.263 | ... |
    | **...** | ... | ... | ... | ... | ... |
    | **8759** | 1.150 | 0.378 | 0.950 | 0.312 | ... |

#### B. Power Flow Outputs (`/pwrflw/output/`)
The resulting power flow timeseries (8,760 hourly timesteps) calculated via pandapower. These are standard `pandas.DataFrame` objects stored inside the HDFStore:

* **Pre-Expansion Results (`/pwrflw/output/pre/`)** and **Post-Expansion Results (`/pwrflw/output/post/`)**:
  * **`demand_import`**: DataFrame storing net active (`p_mw`) and reactive (`q_mvar`) power imported from or fed back into the external grid: the transformer's demand!. Its columns are `['p_mw', 'q_mvar']`, and each row (index) represents a timestamp:
      | Timestep | `p_mw` | `q_mvar` |
      | :--- | :--- | :--- |
      | **0** | 0.432 | 0.082 |
      | **1** | 0.415 | 0.079 |
      | **...** | ... | ... |
      | **8759** | 0.390 | 0.074 |

  * **`vm`**: DataFrame storing bus voltage magnitude in per-unit (p.u.) for all buses. Each column represents a bus ID (integer), and each row represents a timestep:
      | Timestep | `0` | `1` | `2` | ... | `N` |
      | :--- | :--- | :--- | :--- | :--- | :--- |
      | **0** | 1.000 | 1.012 | 1.015 | ... | 0.985 |
      | **1** | 1.000 | 1.011 | 1.014 | ... | 0.986 |
      | **...** | ... | ... | ... | ... | ... |
      | **8759** | 1.000 | 1.013 | 1.016 | ... | 0.983 |

  * **`line_loads`**: DataFrame storing flow and current details for all transmission lines.The columns are tuples of (line_id, parameter), where the parameter is the active flow (`p_from_mw`), reactive flow (`q_from_mvar`), and line current (`i_from_ka`) at the "from" end of the line. Each row represents a timestep:
      | Timestep | `(0, 'p_from_mw')` | `(0, 'q_from_mvar')` | `(0, 'i_from_ka')` | `(1, 'p_from_mw')` | ... |
      | :--- | :--- | :--- | :--- | :--- | :--- |
      | **0** | 0.015 | 0.003 | 0.022 | -0.004 | ... |
      | **1** | 0.014 | 0.003 | 0.020 | -0.005 | ... |
      | **...** | ... | ... | ... | ... | ... |
      | **8759** | 0.012 | 0.002 | 0.018 | -0.003 | ... |

#### C. Traceability (`/pwrflw/urbs_out/MILP/`)
* **`reactive`**: DataFrame storing synthesized reactive power timeseries components (household demand, heat pump, PV capability limits) generated during demand reconstruction for post-expansion modeling.

> [!IMPORTANT]
> **Output Location Constraints:**
> All timeseries power flow results (voltages, line loading, and grid imports) are written **only** under the `/pwrflw` group. 
> The central pandapower network representation inside **`/raw_data/net` is NOT modified or updated** with these timeseries results. It remains a static network model representation (though it may contain a single-snapshot `res_bus` table representing initial network states or solver initializations).
