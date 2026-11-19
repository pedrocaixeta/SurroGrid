"""Pre-expansion demand reconstruction for time-series power flow.

Converts per-bus active electricity demand stored in `/urbs_in/demand` into
per-bus active (`electricity`, kW) and reactive (`electricity-reactive`, kVAr)
load profiles using the fixed power factor `PF_ELC = 0.959`.
"""

import numpy as np
import pandas as pd

from pre_powerflow.config import config


def _process_pre_demands(df_pre_demand):
    """Extract per-bus electricity demand and derive reactive power via PF_ELC."""
    df_raw_demand_elec = df_pre_demand.loc[
        :, df_pre_demand.columns.get_level_values(1) == "electricity"
    ]
    df_raw_demand_react = df_raw_demand_elec.copy() * np.tan(np.arccos(config.PF_ELC))
    df_raw_demand_react.columns = df_raw_demand_react.columns.map(
        lambda x: (x[0], "electricity-reactive")
    )
    return df_raw_demand_elec, df_raw_demand_react


def obtain_pre_demand(df_raw_demand):
    """Return combined pre-expansion active + reactive demand DataFrame."""
    df_pre_demand_elec, df_pre_demand_react = _process_pre_demands(df_raw_demand)
    return pd.concat([df_pre_demand_elec, df_pre_demand_react], axis=1)
