"""
EDA — Food & Fuel Security Dataset

"""

import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"

EDA_OUTPUT_DIR = Path("../outputs")
os.makedirs(EDA_OUTPUT_DIR, exist_ok=True)

EDA_BROAD_INSECURE = {"Low food security", "Very low food security", "Marginal food security",
                      "Low fuel-security", "Very low fuel-security", "Marginal fuel-security"}
EDA_SEVERE_INSECURE = {"Very low food security", "Very low fuel-security"}

EDA_DEMOGRAPHIC_COLS = ["gender", "age_group", "health_condition", "household_type",
                         "employment_status_group", "income_band"]

EDA_HOUSING_COLS = ["housing_tenure_group", "epc_rating", "prepayment_meter",
                     "private_renter", "social_renter", "owner_occupier"]

EDA_CORRELATION_COLS = ["age_midpoint", "income_midpoint", "life_satisfaction",
                         "social_isolation_score", "social_support_score",
                         "imd_decile", "food_security_score", "fuel_security_score"]

EDA_FUEL_HOUSING_CHARTS = {
    "housing_tenure_group": "fuel_insecurity_by_housing_tenure.png",
    "prepayment_meter":     "fuel_insecurity_by_prepayment_meter.png",
    "epc_rating":            "fuel_insecurity_by_epc_rating.png",
}


EDA_EMPLOYMENT_COLS = [
    "Working full-time",
    "Working part-time",
    "Unemployed",
    "Retired",
    "Not working- looking after house/ children",
    "Not working- long term sick or disabled",
    "Student",
]

 # question headers renamed to short working names. Substring matching avoids
EDA_COLUMN_RENAME_SUBSTRINGS = {
    "How would you describe your gender?": "gender",
    "What range best describes your age group?": "age_group",
    "physical or mental health conditions": "health_condition",
    "best describes your household?": "household_type",
    "how satisfied are you with your life?": "life_satisfaction",
    "Compared to others in my neighbourhood": "life_comparison",
    "you lack companionship?": "lack_companionship",
    "isolated from others or left out?": "feel_isolated",
    "turn to friends and family for help": "family_help",
    "borrow things and exchange favours": "neighbour_exchange",
    "best describes the house you currently live in?": "housing_tenure_group",
    "Energy Performance Certificate (EPC) rating": "epc_rating",
    "prepayment or pay-as-you-go meter": "prepayment_meter",
    "worry about making rent/ mortgage payments?": "rent_mortgage_worry",
    "worry about making energy payments?": "energy_payment_worry",
    "annual household income before tax?": "income_band",
}


EDA_EXACT_RENAMES = {
    "household_food_security_label": "food_security_label",
    "fuel-security_label": "fuel_security_label",
    "fuel-security_score": "fuel_security_score",
}

EDA_AGE_MIDPOINTS = {
    "16 to 19": 17.5, "20 to 24": 22, "25 to 29": 27, "30 to 34": 32,
    "35 to 39": 37, "40 to 44": 42, "45 to 49": 47, "50 to 54": 52,
    "55 to 59": 57, "60 to 64": 62, "65 to 69": 67, "70 to 74": 72,
    "75 to 79": 77, "80 or over": 82,
}

EDA_INCOME_MIDPOINTS = {
    "Less than £14,900 p.a.": 12000,
    "£14,901- £24,300 p.a.": 19600,
    "£24,301- £37,900 p.a.": 31100,
    "£37,901- £58,900 p.a.": 48400,
    "More than £58,900 p.a.": 68900,
}

EDA_ORDINAL_MAPS = {
    "lack_companionship": {"Hardly ever or never": 0, "Some of the time": 1, "Often": 2},
    "feel_isolated": {"Hardly ever or never": 0, "Some of the time": 1, "Often": 2},
    "family_help": {"Not at all": 0, "To some extent": 1, "To a large extent": 2},
    "neighbour_exchange": {"Definitely disagree": 0, "Tend to disagree": 1,
                            "Tend to agree": 2, "Definitely agree": 3},
}


def eda_rename_columns(df):

    rename_map = {}
    for col in df.columns:
        for substring, new_name in EDA_COLUMN_RENAME_SUBSTRINGS.items():
            if substring in col:
                rename_map[col] = new_name
                break
    rename_map.update({k: v for k, v in EDA_EXACT_RENAMES.items() if k in df.columns})
    return df.rename(columns=rename_map)


def eda_build_employment_status(df):
    present = [c for c in EDA_EMPLOYMENT_COLS if c in df.columns]

    def pick(row):
        for col in present:
            if row[col] != "missing" and pd.notna(row[col]):
                return col
        return "Not specified"

    df["employment_status_group"] = df[present].apply(pick, axis=1)
    return df


def eda_build_housing_flags(df):
    df["private_renter"] = df["housing_tenure_group"] == "Renting from private landlord"
    df["social_renter"] = df["housing_tenure_group"].isin([
        "Living in rented accommodation from Housing Association",
        "Living in rented accommodation arranged by the Local Authority",
    ])
    df["owner_occupier"] = df["housing_tenure_group"].isin([
        "Buying the house on a mortgage", "Owning the house outright",
    ])
    return df


def eda_build_midpoints(df):
    df["age_midpoint"] = df["age_group"].map(EDA_AGE_MIDPOINTS)
    df["income_midpoint"] = df["income_band"].map(EDA_INCOME_MIDPOINTS)
    return df


def eda_build_ordinal_scores(df):
    for col, mapping in EDA_ORDINAL_MAPS.items():
        df[f"{col}_ord"] = df[col].map(mapping)
    df["social_isolation_score"] = df[["lack_companionship_ord", "feel_isolated_ord"]].mean(axis=1)
    df["family_help_score"] = df["family_help_ord"]
    df["neighbour_exchange_score"] = df["neighbour_exchange_ord"]
    df["social_support_score"] = df[["family_help_ord", "neighbour_exchange_ord"]].mean(axis=1)
    return df


def eda_load_and_prepare(csv_path):
    df = pd.read_csv(csv_path)
    df = eda_rename_columns(df)
    df = eda_build_employment_status(df)
    df = eda_build_housing_flags(df)
    df = eda_build_midpoints(df)
    df = eda_build_ordinal_scores(df)
    return df


def eda_save_fig(fig, filename):
    fig.savefig(f"{EDA_OUTPUT_DIR}/{filename}")
    plt.show()
    plt.close(fig)


def eda_rate_table(df, group_col, target_col, insecure_values=EDA_BROAD_INSECURE):
    tmp = df[[group_col, target_col]].dropna()
    tmp["insecure"] = tmp[target_col].isin(insecure_values)
    rates = tmp.groupby(group_col)["insecure"].mean().mul(100)
    counts = tmp.groupby(group_col).size()
    out = pd.DataFrame({"insecurity_rate_pct": rates, "n": counts}).reset_index()
    return out.sort_values("insecurity_rate_pct", ascending=False)


def eda_rate_bar(df, group_col, target_col, title, filename,
                  insecure_values=EDA_BROAD_INSECURE, sort_by_group=False):
    rates = eda_rate_table(df, group_col, target_col, insecure_values)
    if sort_by_group:
        rates = rates.sort_values(group_col)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=rates, x=group_col, y="insecurity_rate_pct", ax=ax)
    for i, row in rates.reset_index(drop=True).iterrows():
        ax.text(i, row["insecurity_rate_pct"] + 0.5, f'n={int(row["n"])}', ha="center", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel("Insecurity rate (%)")
    plt.xticks(rotation=40, ha="right")
    eda_save_fig(fig, filename)
    return rates


def eda_boxplot(df, score_col, label_col, title, filename):
    tmp = df[[score_col, label_col]].dropna()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=tmp, x=label_col, y=score_col, ax=ax)
    sns.stripplot(data=tmp, x=label_col, y=score_col, ax=ax, color="black", alpha=0.15, size=2)
    ax.set_title(title)
    plt.xticks(rotation=30, ha="right")
    eda_save_fig(fig, filename)
    print(tmp.groupby(label_col)[score_col].agg(["mean", "median", "std", "count"]))



# A. DATA QUALITY EDA


def section_a_data_quality(df, response_time_col="response_time", incentivised_col="incentivised"):
    miss = (df.isna().mean() * 100).sort_values(ascending=False)
    miss = miss[miss > 0].head(30)
    if len(miss):
        fig, ax = plt.subplots(figsize=(9, max(4, 0.3 * len(miss))))
        sns.barplot(x=miss.values, y=miss.index, ax=ax, color="#e07a5f")
        ax.set_xlabel("% missing")
        ax.set_title("Top 30 columns by missingness")
        eda_save_fig(fig, "missing_values_top_30.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[response_time_col].dropna(), bins=40, kde=True, ax=ax, color="#3d5a80")
    ax.set_title("Response time distribution")
    ax.set_xlabel("Response time")
    eda_save_fig(fig, "response_time_histogram.png")

    print("Full-row duplicates:", df.duplicated().sum())
    print("Valid responses:", len(df))

    inc_counts = df[incentivised_col].value_counts()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(x=inc_counts.index.astype(str), y=inc_counts.values, ax=ax, color="#81b29a")
    ax.set_title("Incentivised vs non-incentivised responses")
    ax.set_ylabel("Count")
    eda_save_fig(fig, "incentivised_distribution.png")
    print(inc_counts)


# B. TARGET DISTRIBUTION EDA


def section_b_target_distribution(df, food_col="food_security_label", fuel_col="fuel_security_label"):
    food_counts = df[food_col].value_counts()
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x=food_counts.index.astype(str), y=food_counts.values, ax=ax, color="#f2cc8f")
    ax.set_title("Food security label distribution")
    plt.xticks(rotation=30, ha="right")
    eda_save_fig(fig, "food_security_distribution.png")
    print(food_counts)
    print(f"Broad food insecurity rate: {df[food_col].isin(EDA_BROAD_INSECURE).mean() * 100:.2f}%")
    print(f"Severe food insecurity rate: {df[food_col].isin(EDA_SEVERE_INSECURE).mean() * 100:.2f}%")

    fuel_counts = df[fuel_col].value_counts()
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x=fuel_counts.index.astype(str), y=fuel_counts.values, ax=ax, color="#81b29a")
    ax.set_title("Fuel security label distribution")
    plt.xticks(rotation=30, ha="right")
    eda_save_fig(fig, "fuel_security_distribution.png")
    print(fuel_counts)
    print(f"Broad fuel insecurity rate: {df[fuel_col].isin(EDA_BROAD_INSECURE).mean() * 100:.2f}%")
    print(f"Severe fuel insecurity rate: {df[fuel_col].isin(EDA_SEVERE_INSECURE).mean() * 100:.2f}%")

    tmp = df[[food_col, fuel_col]].dropna()
    tmp["food_insecure"] = tmp[food_col].isin(EDA_BROAD_INSECURE)
    tmp["fuel_insecure"] = tmp[fuel_col].isin(EDA_BROAD_INSECURE)
    overlap = pd.crosstab(tmp["food_insecure"], tmp["fuel_insecure"])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(overlap, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Food vs fuel insecurity overlap")
    ax.set_xlabel("Fuel insecure")
    ax.set_ylabel("Food insecure")
    eda_save_fig(fig, "food_vs_fuel_overlap.png")
    print(f"Insecure in BOTH: {((tmp['food_insecure']) & (tmp['fuel_insecure'])).mean() * 100:.2f}%")



# C. DEMOGRAPHIC EDA

EDA_FOOD_DEMO_CHARTS = {
    "income_band":              "food_insecurity_by_income.png",
    "employment_status_group":  "food_insecurity_by_employment.png",
    "household_type":           "food_insecurity_by_household_type.png",
    "health_condition":         "food_insecurity_by_health_condition.png",
    "age_group":                "food_insecurity_by_age.png",
}


def section_c_food_demographics(df, target_col="food_security_label"):
    for demo_col in EDA_DEMOGRAPHIC_COLS:
        filename = EDA_FOOD_DEMO_CHARTS.get(demo_col, f"food_insecurity_by_{demo_col}.png")
        eda_rate_bar(df, demo_col, target_col, f"Food insecurity rate by {demo_col}", filename)


def section_c_fuel_demographics(df, target_col="fuel_security_label"):
    for demo_col in EDA_DEMOGRAPHIC_COLS:
        eda_rate_bar(df, demo_col, target_col, f"Fuel insecurity rate by {demo_col}",
                     f"fuel_insecurity_by_{demo_col}.png")



# D. HOUSING EDA


def section_d_food_housing(df, target_col="food_security_label"):
    for h_col in EDA_HOUSING_COLS:
        eda_rate_bar(df, h_col, target_col, f"Food insecurity rate by {h_col}",
                     f"food_insecurity_by_{h_col}.png")


def section_d_fuel_housing(df, target_col="fuel_security_label"):
    for h_col in EDA_HOUSING_COLS:
        filename = EDA_FUEL_HOUSING_CHARTS.get(h_col, f"fuel_insecurity_by_{h_col}.png")
        eda_rate_bar(df, h_col, target_col, f"Fuel insecurity rate by {h_col}", filename)
    for worry_col in ["rent_mortgage_worry", "energy_payment_worry"]:
        print(eda_rate_table(df, worry_col, target_col))



# E. SOCIAL ISOLATION EDA


def section_e_social_isolation(df, food_col="food_security_label", fuel_col="fuel_security_label"):
    eda_boxplot(df, "social_isolation_score", food_col,
                "Social isolation score by food security label", "social_isolation_by_food_security.png")
    eda_boxplot(df, "social_isolation_score", fuel_col,
                "Social isolation score by fuel security label", "social_isolation_by_fuel_security.png")

    tmp = df[["family_help_score", "neighbour_exchange_score", food_col]].dropna()
    tmp_melt = tmp.melt(id_vars=food_col, value_vars=["family_help_score", "neighbour_exchange_score"],
                         var_name="support_type", value_name="score")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=tmp_melt, x=food_col, y="score", hue="support_type", ax=ax)
    ax.set_title("Family help & neighbour exchange scores by food security label")
    plt.xticks(rotation=30, ha="right")
    eda_save_fig(fig, "social_support_by_food_security.png")



# F. GEOGRAPHY EDA
#
# NOTE: crime_rate_per_1000 and green_space_pct aren't in this dataset, so
# the external-feature quintile charts from the original template are left
# out. lad_code, msoa_code and imd_decile are all present and used below.
# msoa_code is very high-cardinality (mostly 1-2 respondents per area) so
# it isn't charted directly here — lad_code (borough-level) is the more
# useful granularity for a top-20 bar chart.


def section_f_geography(df, food_col="food_security_label", fuel_col="fuel_security_label"):
    eda_rate_bar(df, "imd_decile", food_col, "Food insecurity rate by IMD decile",
                 "food_insecurity_by_imd_decile.png", sort_by_group=True)
    eda_rate_bar(df, "imd_decile", fuel_col, "Fuel insecurity rate by IMD decile",
                 "fuel_insecurity_by_imd_decile.png", sort_by_group=True)

    top_food = eda_rate_table(df, "lad_code", food_col).head(20)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=top_food, x="lad_code", y="insecurity_rate_pct", ax=ax)
    ax.set_title("Food insecurity rate by borough (top 20)")
    plt.xticks(rotation=45, ha="right")
    eda_save_fig(fig, "food_insecurity_by_borough.png")

    top_fuel = eda_rate_table(df, "lad_code", fuel_col).head(20)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=top_fuel, x="lad_code", y="insecurity_rate_pct", ax=ax)
    ax.set_title("Fuel insecurity rate by borough (top 20)")
    plt.xticks(rotation=45, ha="right")
    eda_save_fig(fig, "fuel_insecurity_by_borough.png")



# G. CORRELATION / ASSOCIATION EDA

def section_g_correlation(df):
    corr = df[EDA_CORRELATION_COLS].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlation heatmap — numerical engineered features")
    eda_save_fig(fig, "correlation_heatmap.png")

