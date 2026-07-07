"""
Feature Engineering — External Area Features (crime & green space)

"""

import os
import pandas as pd
from pathlib import Path

FE_EXTERNAL_DIR = Path("../data/external")
FE_FEATURED_DIR = Path("../data/featured_engineering")
os.makedirs(FE_FEATURED_DIR, exist_ok=True)

FE_CRIME_XLSX = FE_EXTERNAL_DIR / "ons_csp_recorded_crime_ye_dec2022.xlsx"
FE_POPULATION_XLSX = FE_EXTERNAL_DIR / "ons_mid2022_population_estimates_lad.xlsx"
FE_GARDENS_XLSX = FE_EXTERNAL_DIR / "ons_os_private_outdoor_space_gb_april2020.xlsx"
FE_PARKS_XLSX = FE_EXTERNAL_DIR / "ons_os_public_green_space_gb_april2020.xlsx"

FE_AREA_FEATURE_COLS = ["crime_rate_per_1000", "green_space_pct", "park_distance_m"]

FE_QUINTILE_LABELS = ["Q1", "Q2", "Q3", "Q4", "Q5"]


# A. BOROUGH CRIME RATE (ONS YE Dec 2022 counts / mid-2022 population)
#
# NOTE: ONS CSP Table C3 is used rather than the MPS Datastore extract because
# it covers all 33 boroughs including E09000001 City of London, which has its
# own police force and is absent from MPS data. The City's per resident rate
# (~634 per 1,000) is inflated by its daytime population — flag in analysis.


def fe_load_borough_crime():
    crime = pd.read_excel(FE_CRIME_XLSX, sheet_name="Table C3", header=7)
    crime = crime[crime["Local Authority code"].astype(str).str.startswith("E09")]
    crime = crime[["Local Authority code", "Total recorded crime\n (excluding fraud)"]]
    crime.columns = ["lad_code", "crime_count_ye_dec2022"]

    population = pd.read_excel(FE_POPULATION_XLSX, sheet_name="MYE2 - Persons", header=7)
    population = population[population["Code"].astype(str).str.startswith("E09")]
    population = population[["Code", "All ages"]]
    population.columns = ["lad_code", "population_mid2022"]

    out = crime.merge(population, on="lad_code", how="inner")
    out["crime_rate_per_1000"] = out["crime_count_ye_dec2022"] / out["population_mid2022"] * 1000
    return out.sort_values("lad_code").reset_index(drop=True)


# B. MSOA GREEN SPACE (ONS/OS April 2020 snapshot)


def fe_load_msoa_greenspace():
    gardens = pd.read_excel(FE_GARDENS_XLSX, sheet_name="MSOA gardens", header=[0, 1])
    # NOTE: source percentage column is a 0-1 fraction; 'adresses' is ONS typo // note guys
    gardens = pd.DataFrame({
        "msoa_code": gardens[("MSOA code", "Unnamed: 6_level_1")],
        "lad_code": gardens[("LAD code", "Unnamed: 4_level_1")],
        "green_space_pct": gardens[("Property type: Total",
                                    "Percentage of adresses with private outdoor space")] * 100,
    })
    gardens = gardens[gardens["lad_code"].astype(str).str.startswith("E09")]
    gardens = gardens.drop(columns="lad_code")

    parks = pd.read_excel(FE_PARKS_XLSX, sheet_name="LSOA Parks only")
    parks = parks[parks["LAD code"].astype(str).str.startswith("E09")]
    # NOTE: LSOA figures are means over built up area postcodes, so the
    # postcode weighted mean reconstructs the MSOA level postcode mean
    weights = parks["Number of postcodes within built up area"]
    parks = parks.assign(
        weighted_distance=parks["Average distance to nearest Park or Public Garden (m)"] * weights)
    parks = (parks.groupby("MSOA code")
                  .agg(weighted_distance=("weighted_distance", "sum"),
                       n_postcodes=("Number of postcodes within built up area", "sum"))
                  .reset_index())
    parks["park_distance_m"] = parks["weighted_distance"] / parks["n_postcodes"]
    parks = parks.rename(columns={"MSOA code": "msoa_code"})

    out = gardens.merge(parks[["msoa_code", "park_distance_m"]], on="msoa_code", how="inner")
    return out.sort_values("msoa_code").reset_index(drop=True)


# C. AREA FEATURE JOIN


def fe_add_area_features(df, crime_lad=None, green_msoa=None):
    crime_lad = fe_load_borough_crime() if crime_lad is None else crime_lad
    green_msoa = fe_load_msoa_greenspace() if green_msoa is None else green_msoa
    n_rows = len(df)
    out = df.merge(crime_lad[["lad_code", "crime_rate_per_1000"]], on="lad_code", how="left")
    out = out.merge(green_msoa[["msoa_code", "green_space_pct", "park_distance_m"]],
                    on="msoa_code", how="left")
    assert len(out) == n_rows
    assert out[FE_AREA_FEATURE_COLS].notna().all().all()
    return out


def fe_quintile(series):
    return pd.qcut(series, 5, labels=FE_QUINTILE_LABELS)
