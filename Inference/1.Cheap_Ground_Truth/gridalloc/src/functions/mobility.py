"""Statistical car-ownership sampling for the baseline (without-mobility) model.

Retains only the statistical vehicle-count sampling logic (`sample_statistics`)
from GridExpand's mobility.py to populate `cars_by_flat`, `n_cars_tot`, and
`car_dict` on `raw_data/buildings`. All `emobpy` imports and driving/charging
simulation functions were removed for the baseline surrogate workflow.
"""

from config import config
import numpy as np
import pandas as pd


##############################################################
#################### Sampling Statistics #####################
##############################################################
def _sample_cars_in_building(occ_list, df_cars_per_household):
    car_list = []
    for n_occ in occ_list:
        if n_occ > 5:
            n_occ = 5
        subset = df_cars_per_household[df_cars_per_household["hh_size"] == n_occ]
        car_list.append(
            np.random.choice(
                subset["vehicle_count"].values, size=1, p=subset["probability"].values
            ).item()
        )
    return car_list, sum(car_list)


def _sample_model_and_commuting(n_cars, prob_commuter, df_car_model_dist, df_region, bus):
    if n_cars == 0:
        return {}
    else:
        buses = pd.DataFrame([bus] * n_cars, columns=["bus"])
        ids = pd.DataFrame(range(n_cars), columns=["id"])
        car_model = pd.DataFrame(
            np.random.choice(
                df_car_model_dist["model"].values,
                size=n_cars,
                p=df_car_model_dist["probability"].values,
            ),
            columns=["model"],
        )
        commuter = pd.DataFrame(
            np.random.choice(
                ["commuter", "non-commuter"],
                size=n_cars,
                p=[prob_commuter, 1 - prob_commuter],
            ),
            columns=["schedule"],
        )
        seeds = pd.DataFrame(
            [
                int(
                    "".join(
                        str(i)
                        for i in [
                            df_region["bcid"],
                            df_region["plz"],
                            df_region["kcid"],
                            bus,
                            idx,
                        ]
                    )
                )
                for idx in range(n_cars)
            ],
            columns=["seed"],
        )

        df_cars = pd.concat([buses, ids, car_model, commuter, seeds], axis=1)
        car_dict = (df_cars.set_index(["bus", "id"]))[
            ["model", "schedule", "seed"]
        ].to_dict(orient="index")
        return car_dict


##############################################################
###################### Publicly callable #####################
##############################################################
def sample_statistics(df_buildings, df_region):
    # Select correct car owner statistics for the grid's RegioStar7 region
    region = int(df_region["regio7"])
    df_cars_per_hh = config.CARS_PER_HH_BY_REGION[
        config.CARS_PER_HH_BY_REGION["region"] == region
    ]

    # Sample number of owned cars per flat/building, plus model & commuting metadata
    df_buildings[["cars_by_flat", "n_cars_tot"]] = df_buildings["occ_list"].apply(
        lambda x: pd.Series(_sample_cars_in_building(x, df_cars_per_hh))
    )
    df_buildings["car_dict"] = df_buildings.apply(
        lambda x: _sample_model_and_commuting(
            x["n_cars_tot"],
            config.PROB_COMMUTING,
            config.CAR_MODEL_DISTRIBUTION,
            df_region,
            x["bus"],
        ),
        axis=1,
    )

    return df_buildings
