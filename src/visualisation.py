import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight"})

EDA_OUTPUT_DIR = Path("../outputs/EDA")
EDA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EDA_BROAD_INSECURE = {
    "Low food security", "Very low food security", "Marginal food security",
    "Low fuel-security", "Very low fuel-security", "Marginal fuel-security",
}
EDA_DEMOGRAPHIC_COLS = [
    "gender", "age_group", "health_condition", "household_type",
    "employment_status_group", "income_band",
]
EDA_HOUSING_COLS = [
    "housing_tenure_group", "epc_rating", "prepayment_meter",
    "private_renter", "social_renter", "owner_occupier",
]
EDA_CRIME_RATE_COL = "crime_rate_per_1000"
EDA_CORRELATION_BASE_COLS = [
    "age_midpoint", "income_midpoint", "life_satisfaction",
    "social_isolation_score", "social_support_score", "imd_decile",
    "food_security_score", "fuel_security_score", "food_insecure", "fuel_insecure",
]
EDA_EMPLOYMENT_COLS = [
    "Working full-time", "Working part-time", "Unemployed", "Retired",
    "Not working- looking after house/ children",
    "Not working- long term sick or disabled", "Student",
]
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
    "Less than £14,900 p.a.": 12000, "£14,901- £24,300 p.a.": 19600,
    "£24,301- £37,900 p.a.": 31100, "£37,901- £58,900 p.a.": 48400,
    "More than £58,900 p.a.": 68900,
}
EDA_ORDINAL_MAPS = {
    "lack_companionship": {"Hardly ever or never": 0, "Some of the time": 1, "Often": 2},
    "feel_isolated": {"Hardly ever or never": 0, "Some of the time": 1, "Often": 2},
    "family_help": {"Not at all": 0, "To some extent": 1, "To a large extent": 2},
    "neighbour_exchange": {
        "Definitely disagree": 0, "Tend to disagree": 1,
        "Tend to agree": 2, "Definitely agree": 3,
    },
}
EDA_GROUP_ORDERS = {
    "age_group": list(EDA_AGE_MIDPOINTS),
    "income_band": list(EDA_INCOME_MIDPOINTS),
    "imd_decile": list(range(1, 11)),
}
EDA_FOOD_DEMO_FILES = {
    "age_group": "food_insecurity_by_age.png",
    "employment_status_group": "food_insecurity_by_employment.png",
}
EDA_FUEL_HOUSING_FILES = {
    "housing_tenure_group": "fuel_insecurity_by_housing_tenure.png",
    "prepayment_meter": "fuel_insecurity_by_prepayment_meter.png",
    "epc_rating": "fuel_insecurity_by_epc_rating.png",
}


def eda_rename_columns(df):
    rename = {
        col: new for col in df.columns
        for text, new in EDA_COLUMN_RENAME_SUBSTRINGS.items() if text in col
    }
    rename.update({old: new for old, new in EDA_EXACT_RENAMES.items() if old in df})
    return df.rename(columns=rename)


def eda_build_employment_status(df):
    present = [col for col in EDA_EMPLOYMENT_COLS if col in df]
    df["employment_status_group"] = df[present].replace("missing", pd.NA).bfill(axis=1).iloc[:, 0].fillna("Not specified")
    return df


def eda_build_features(df):
    tenure = df["housing_tenure_group"]
    df["private_renter"] = tenure.eq("Renting from private landlord")
    df["social_renter"] = tenure.isin([
        "Living in rented accommodation from Housing Association",
        "Living in rented accommodation arranged by the Local Authority",
    ])
    df["owner_occupier"] = tenure.isin(["Buying the house on a mortgage", "Owning the house outright"])
    df["age_midpoint"] = df["age_group"].map(EDA_AGE_MIDPOINTS)
    df["income_midpoint"] = df["income_band"].map(EDA_INCOME_MIDPOINTS)
    for col, mapping in EDA_ORDINAL_MAPS.items():
        df[f"{col}_ord"] = df[col].map(mapping)
    df["social_isolation_score"] = df[["lack_companionship_ord", "feel_isolated_ord"]].mean(axis=1)
    df["family_help_score"] = df["family_help_ord"]
    df["neighbour_exchange_score"] = df["neighbour_exchange_ord"]
    df["social_support_score"] = df[["family_help_ord", "neighbour_exchange_ord"]].mean(axis=1)
    return df


def eda_build_insecurity_flags(df, food_col="food_security_label", fuel_col="fuel_security_label"):
    for source, output in [(food_col, "food_insecure"), (fuel_col, "fuel_insecure")]:
        values = df[source].replace("missing", pd.NA)
        df[output] = values.isin(EDA_BROAD_INSECURE).astype(float).where(values.notna())
    return df


def eda_build_gender_dummies(df):
    gender = df["gender"].replace("missing", pd.NA)
    dummies = pd.get_dummies(gender, prefix="gender", dtype=float)
    dummies.columns = [re.sub(r"[^a-z0-9]+", "_", col.lower()).strip("_") for col in dummies]
    dummies.loc[gender.isna()] = np.nan
    return pd.concat([df, dummies], axis=1)


def eda_load_crime_rates(path):
    crime = pd.read_csv(path, dtype={"lad_code": "string"})[["lad_code", EDA_CRIME_RATE_COL]]
    crime["lad_code"] = crime["lad_code"].str.strip()
    crime[EDA_CRIME_RATE_COL] = pd.to_numeric(crime[EDA_CRIME_RATE_COL], errors="coerce")
    return crime.dropna().drop_duplicates("lad_code")


def eda_add_crime_rates(df, crime_csv_path):
    out = df.drop(columns=EDA_CRIME_RATE_COL, errors="ignore").copy()
    out["lad_code"] = out["lad_code"].astype("string").str.strip()
    return out.merge(eda_load_crime_rates(crime_csv_path), on="lad_code", how="left")


def eda_load_and_prepare(csv_path, crime_csv_path=None):
    df = eda_rename_columns(pd.read_csv(csv_path))
    df = eda_build_insecurity_flags(eda_build_features(eda_build_employment_status(df)))
    return eda_add_crime_rates(df, crime_csv_path) if crime_csv_path else df


def eda_save_fig(fig, filename):
    fig.tight_layout()
    fig.savefig(EDA_OUTPUT_DIR / filename)
    plt.show()
    plt.close(fig)


def eda_composition_bar(series, title, filename):
    shares = series.replace("missing", pd.NA).dropna().value_counts(normalize=True).mul(100)
    fig, ax = plt.subplots(figsize=(9, 2.8))
    left = 0
    for label, value in shares.items():
        ax.barh([0], value, left=left, label=f"{label} ({value:.1f}%)")
        if value >= 7:
            ax.text(left + value / 2, 0, f"{value:.1f}%", ha="center", va="center", fontsize=9)
        left += value
    ax.set(xlim=(0, 100), yticks=[], xlabel="Percentage of valid responses", title=title)
    ax.legend(bbox_to_anchor=(0.5, -0.35), loc="upper center", ncol=min(3, len(shares)), frameon=False)
    eda_save_fig(fig, filename)
    return shares


def eda_proportion_table(df, group_col, target_col):
    tmp = df[[group_col, target_col]].replace("missing", pd.NA).dropna().copy()
    tmp["status"] = np.where(tmp[target_col].isin(EDA_BROAD_INSECURE), "Insecure", "Secure")
    counts = pd.crosstab(tmp[group_col], tmp["status"]).reindex(columns=["Secure", "Insecure"], fill_value=0)
    pct = counts.div(counts.sum(axis=1), axis=0).mul(100)
    return pct.assign(n=counts.sum(axis=1))


def eda_proportion_bar(df, group_col, target_col, title, filename, top_n=None):
    table = eda_proportion_table(df, group_col, target_col)
    order = EDA_GROUP_ORDERS.get(group_col)
    if order:
        table = table.reindex([value for value in order if value in table.index])
    else:
        table = table.sort_values("Insecure", ascending=False)
    if top_n:
        table = table.nlargest(top_n, "Insecure")

    labels = [f"{value} (n={int(n)})" for value, n in zip(table.index, table["n"])]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.42 * len(table))))
    ax.barh(labels, table["Secure"], label="Secure")
    ax.barh(labels, table["Insecure"], left=table["Secure"], label="Insecure")
    for i, value in enumerate(table["Insecure"]):
        ax.text(99, i, f"{value:.1f}%", ha="right", va="center", fontsize=8)
    ax.set(xlim=(0, 100), xlabel="Proportion of respondents (%)", title=title)
    ax.invert_yaxis()
    ax.legend(loc="lower right", frameon=False)
    eda_save_fig(fig, filename)
    return table.reset_index()


def eda_boxplot(df, score_col, label_col, title, filename):
    tmp = df[[score_col, label_col]].replace("missing", pd.NA).dropna()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=tmp, x=label_col, y=score_col, ax=ax)
    sns.stripplot(data=tmp, x=label_col, y=score_col, alpha=0.15, size=2, ax=ax)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    plt.setp(ax.get_xticklabels(), ha="right")
    eda_save_fig(fig, filename)


# A. DATA QUALITY

def section_a_data_quality(df, response_time_col="response_time", incentivised_col="incentivised"):
    missing = (df.isna().mean() * 100).sort_values(ascending=False)
    missing = missing[missing > 0].head(30)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.3 * len(missing))))
    sns.barplot(x=missing.values, y=missing.index, ax=ax)
    ax.set(xlabel="Missing values (%)", title="Top 30 columns by missingness")
    eda_save_fig(fig, "missing_values_top_30.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df[response_time_col].dropna(), bins=40)
    ax.set(xlabel="Response time", title="Response time distribution")
    eda_save_fig(fig, "response_time_histogram.png")
    eda_composition_bar(df[incentivised_col], "Share of incentivised responses", "incentivised_distribution.png")


# B. TARGET DISTRIBUTIONS

def section_b_target_distribution(df, food_col="food_security_label", fuel_col="fuel_security_label"):
    for col, label, filename in [
        (food_col, "Food security composition", "food_security_distribution.png"),
        (fuel_col, "Fuel security composition", "fuel_security_distribution.png"),
    ]:
        eda_composition_bar(df[col], label, filename)

    overlap = df[[food_col, fuel_col]].replace("missing", pd.NA).dropna()
    food = np.where(overlap[food_col].isin(EDA_BROAD_INSECURE), "Insecure", "Secure")
    fuel = np.where(overlap[fuel_col].isin(EDA_BROAD_INSECURE), "Insecure", "Secure")
    overlap_pct = pd.crosstab(food, fuel, normalize="all").mul(100).reindex(
        index=["Secure", "Insecure"], columns=["Secure", "Insecure"], fill_value=0
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(overlap_pct, annot=True, fmt=".1f", cmap="Blues", cbar_kws={"label": "% of valid respondents"}, ax=ax)
    ax.set(xlabel="Fuel security", ylabel="Food security", title="Food and fuel insecurity overlap (%)")
    eda_save_fig(fig, "food_vs_fuel_overlap.png")


# C. DEMOGRAPHICS

def eda_group_section(df, columns, target_col, outcome, file_overrides=None):
    file_overrides = file_overrides or {}
    for col in columns:
        filename = file_overrides.get(col, f"{outcome.lower()}_insecurity_by_{col}.png")
        eda_proportion_bar(df, col, target_col, f"{outcome} security composition by {col}", filename)


def section_c_food_demographics(df, target_col="food_security_label"):
    eda_group_section(df, EDA_DEMOGRAPHIC_COLS, target_col, "Food", EDA_FOOD_DEMO_FILES)


def section_c_fuel_demographics(df, target_col="fuel_security_label"):
    eda_group_section(df, EDA_DEMOGRAPHIC_COLS, target_col, "Fuel")


# D. HOUSING

def section_d_food_housing(df, target_col="food_security_label"):
    eda_group_section(df, EDA_HOUSING_COLS, target_col, "Food")


def section_d_fuel_housing(df, target_col="fuel_security_label"):
    eda_group_section(df, EDA_HOUSING_COLS, target_col, "Fuel", EDA_FUEL_HOUSING_FILES)


# E. SOCIAL ISOLATION AND SUPPORT

def section_e_social_isolation(df, food_col="food_security_label", fuel_col="fuel_security_label"):
    eda_boxplot(df, "social_isolation_score", food_col, "Social isolation by food security", "social_isolation_by_food_security.png")
    eda_boxplot(df, "social_isolation_score", fuel_col, "Social isolation by fuel security", "social_isolation_by_fuel_security.png")

    tmp = df[["family_help_score", "neighbour_exchange_score", food_col]].replace("missing", pd.NA).dropna()
    melted = tmp.melt(id_vars=food_col, var_name="support_type", value_name="score")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=melted, x=food_col, y="score", hue="support_type", ax=ax)
    ax.set_title("Family help and neighbour exchange by food security")
    ax.tick_params(axis="x", rotation=30)
    plt.setp(ax.get_xticklabels(), ha="right")
    eda_save_fig(fig, "social_support_by_food_security.png")


# F. GEOGRAPHY

def section_f_geography(df, food_col="food_security_label", fuel_col="fuel_security_label"):
    for target, outcome in [(food_col, "Food"), (fuel_col, "Fuel")]:
        eda_proportion_bar(
            df, "imd_decile", target, f"{outcome} security composition by IMD decile",
            f"{outcome.lower()}_insecurity_by_imd_decile.png",
        )
        eda_proportion_bar(
            df, "lad_code", target, f"Top 20 boroughs by {outcome.lower()} insecurity",
            f"{outcome.lower()}_insecurity_by_borough.png", top_n=20,
        )


# G. INDIVIDUAL-LEVEL CORRELATION

def section_g_correlation(df, food_col="food_security_label", fuel_col="fuel_security_label"):
    corr_df = eda_build_gender_dummies(eda_build_insecurity_flags(df.copy(), food_col, fuel_col))
    gender_cols = [col for col in corr_df if col.startswith("gender_")]
    corr_cols = [col for col in EDA_CORRELATION_BASE_COLS + gender_cols if col in corr_df]
    corr = corr_df[corr_cols].apply(pd.to_numeric, errors="coerce").corr()
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title("Individual-level correlation heatmap including gender indicators")
    eda_save_fig(fig, "correlation_heatmap.png")
    return corr


# H. MAIN BOROUGH-LEVEL CRIME ANALYSIS

def eda_build_borough_crime_table(df):
    borough = df.groupby("lad_code", as_index=False).agg(
        crime_rate_per_1000=(EDA_CRIME_RATE_COL, "first"),
        food_insecurity_rate_pct=("food_insecure", lambda x: x.mean() * 100),
        fuel_insecurity_rate_pct=("fuel_insecure", lambda x: x.mean() * 100),
        mean_imd_decile=("imd_decile", "mean"),
        respondent_count=("lad_code", "size"),
    ).dropna(subset=[EDA_CRIME_RATE_COL])
    borough["log_crime_rate"] = np.log1p(borough[EDA_CRIME_RATE_COL])
    return borough


def section_h_crime_analysis(df, crime_csv_path=None):
    crime_df = eda_add_crime_rates(df, crime_csv_path) if crime_csv_path else df.copy()
    borough = eda_build_borough_crime_table(crime_df)
    outcomes = ["food_insecurity_rate_pct", "fuel_insecurity_rate_pct"]
    summary = pd.DataFrame({
        "outcome": outcomes,
        "pearson_r": [borough[[EDA_CRIME_RATE_COL, col]].corr().iloc[0, 1] for col in outcomes],
        "spearman_rho": [borough[[EDA_CRIME_RATE_COL, col]].corr(method="spearman").iloc[0, 1] for col in outcomes],
    })
    borough.to_csv(EDA_OUTPUT_DIR / "borough_crime_analysis.csv", index=False)
    summary.to_csv(EDA_OUTPUT_DIR / "crime_correlation_summary.csv", index=False)

    crime_cols = [EDA_CRIME_RATE_COL, *outcomes, "mean_imd_decile"]
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        borough[crime_cols].corr(method="spearman"), annot=True, fmt=".2f",
        cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax,
    )
    ax.set_title("Borough-level crime associations: Spearman correlation")
    eda_save_fig(fig, "crime_borough_spearman_heatmap.png")

    for outcome, label, filename in [
        ("food_insecurity_rate_pct", "Food insecurity rate (%)", "crime_vs_food_insecurity.png"),
        ("fuel_insecurity_rate_pct", "Fuel insecurity rate (%)", "crime_vs_fuel_insecurity.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.regplot(data=borough, x="log_crime_rate", y=outcome, scatter_kws={"s": 60, "alpha": 0.75}, ax=ax)
        ax.set(xlabel="Log-transformed crime rate per 1,000", ylabel=label, title=f"Borough crime rate versus {label.lower()}")
        eda_save_fig(fig, filename)

    borough["crime_quartile"] = pd.qcut(
        borough[EDA_CRIME_RATE_COL], 4,
        labels=["Lowest", "Low-medium", "High-medium", "Highest"], duplicates="drop",
    )
    quartiles = borough.groupby("crime_quartile", observed=True, as_index=False).agg(
        food_insecurity_rate_pct=("food_insecurity_rate_pct", "mean"),
        fuel_insecurity_rate_pct=("fuel_insecurity_rate_pct", "mean"),
        borough_count=("lad_code", "size"),
    )
    quartiles.to_csv(EDA_OUTPUT_DIR / "crime_quartile_summary.csv", index=False)
    plot_df = quartiles.melt(
        id_vars=["crime_quartile", "borough_count"], value_vars=outcomes,
        var_name="outcome", value_name="insecurity_rate_pct",
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=plot_df, x="crime_quartile", y="insecurity_rate_pct", hue="outcome", ax=ax)
    ax.set(xlabel="Crime-rate quartile", ylabel="Average insecurity rate (%)", title="Average insecurity rate by borough crime quartile")
    eda_save_fig(fig, "insecurity_by_crime_quartile.png")
    return borough, summary


# I. POWER BI CSV EXPORTS

POWERBI_OUTPUT_DIR = Path("../data/unified_csv")
LONDON_BOROUGH_NAMES = {
    "E09000001": "City of London", "E09000002": "Barking and Dagenham",
    "E09000003": "Barnet", "E09000004": "Bexley", "E09000005": "Brent",
    "E09000006": "Bromley", "E09000007": "Camden", "E09000008": "Croydon",
    "E09000009": "Ealing", "E09000010": "Enfield", "E09000011": "Greenwich",
    "E09000012": "Hackney", "E09000013": "Hammersmith and Fulham",
    "E09000014": "Haringey", "E09000015": "Harrow", "E09000016": "Havering",
    "E09000017": "Hillingdon", "E09000018": "Hounslow", "E09000019": "Islington",
    "E09000020": "Kensington and Chelsea", "E09000021": "Kingston upon Thames",
    "E09000022": "Lambeth", "E09000023": "Lewisham", "E09000024": "Merton",
    "E09000025": "Newham", "E09000026": "Redbridge",
    "E09000027": "Richmond upon Thames", "E09000028": "Southwark",
    "E09000029": "Sutton", "E09000030": "Tower Hamlets",
    "E09000031": "Waltham Forest", "E09000032": "Wandsworth",
    "E09000033": "Westminster",
}


def prepare_powerbi_dashboard(df):
    out = df.replace("missing", pd.NA).copy()
    if "user_id" not in out:
        out["user_id"] = np.arange(1, len(out) + 1)

    out = eda_build_insecurity_flags(out)
    out["food_security_status"] = out["food_insecure"].map({0.0: "Secure", 1.0: "Insecure"})
    out["fuel_security_status"] = out["fuel_insecure"].map({0.0: "Secure", 1.0: "Insecure"})
    valid_both = out[["food_insecure", "fuel_insecure"]].notna().all(axis=1)
    out["both_insecure"] = ((out["food_insecure"] == 1) & (out["fuel_insecure"] == 1)).astype(float).where(valid_both)
    out["severe_food_insecure"] = out["food_security_label"].eq("Very low food security").astype(float).where(out["food_security_label"].notna())
    out["severe_fuel_insecure"] = out["fuel_security_label"].eq("Very low fuel-security").astype(float).where(out["fuel_security_label"].notna())
    out["borough_name"] = out.get("borough_name", out["lad_code"].map(LONDON_BOROUGH_NAMES))
    out["age_order"] = out["age_group"].map({value: i for i, value in enumerate(EDA_AGE_MIDPOINTS, 1)})
    out["income_order"] = out["income_band"].map({value: i for i, value in enumerate(EDA_INCOME_MIDPOINTS, 1)})
    out["log_crime_rate"] = np.log1p(out[EDA_CRIME_RATE_COL])

    crime = out[["lad_code", EDA_CRIME_RATE_COL]].dropna().drop_duplicates("lad_code")
    crime["crime_quartile"] = pd.qcut(
        crime[EDA_CRIME_RATE_COL], 4,
        labels=["Lowest", "Low-medium", "High-medium", "Highest"],
        duplicates="drop",
    )
    crime["crime_quartile_order"] = crime["crime_quartile"].map(
        {"Lowest": 1, "Low-medium": 2, "High-medium": 3, "Highest": 4}
    )
    out = out.merge(crime.drop(columns=EDA_CRIME_RATE_COL), on="lad_code", how="left")
    return out


def build_security_long(dashboard):
    base = [
        "user_id", "gender", "age_group", "age_order", "health_condition",
        "household_type", "employment_status_group", "income_band", "income_order",
        "housing_tenure_group", "epc_rating", "prepayment_meter", "private_renter",
        "social_renter", "owner_occupier", "life_satisfaction",
        "social_isolation_score", "social_support_score", "lad_code", "borough_name",
        "imd_decile", EDA_CRIME_RATE_COL, "log_crime_rate", "crime_quartile",
        "crime_quartile_order",
    ]
    base = [col for col in base if col in dashboard]
    frames = []
    for security_type, prefix in [("Food", "food"), ("Fuel", "fuel")]:
        cols = base + [
            f"{prefix}_security_label", f"{prefix}_security_score",
            f"{prefix}_security_status", f"{prefix}_insecure",
            f"severe_{prefix}_insecure",
        ]
        part = dashboard[cols].copy().rename(columns={
            f"{prefix}_security_label": "security_label",
            f"{prefix}_security_score": "security_score",
            f"{prefix}_security_status": "security_status",
            f"{prefix}_insecure": "insecure_flag",
            f"severe_{prefix}_insecure": "severe_flag",
        })
        part.insert(1, "security_type", security_type)
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def build_group_summary(dashboard, columns):
    tables = []
    for variable in columns:
        tmp = dashboard[["user_id", variable, "food_insecure", "fuel_insecure", "both_insecure"]].dropna(subset=[variable])
        summary = tmp.groupby(variable, observed=True, as_index=False).agg(
            respondent_count=("user_id", "nunique"),
            food_insecurity_rate_pct=("food_insecure", lambda x: x.mean() * 100),
            fuel_insecurity_rate_pct=("fuel_insecure", lambda x: x.mean() * 100),
            both_insecure_rate_pct=("both_insecure", lambda x: x.mean() * 100),
        ).rename(columns={variable: "category"})
        summary.insert(0, "variable", variable)
        tables.append(summary)
    return pd.concat(tables, ignore_index=True)


def build_borough_summary(dashboard):
    borough = dashboard.groupby(["lad_code", "borough_name"], as_index=False).agg(
        respondent_count=("user_id", "nunique"),
        crime_rate_per_1000=(EDA_CRIME_RATE_COL, "first"),
        log_crime_rate=("log_crime_rate", "first"),
        crime_quartile=("crime_quartile", "first"),
        crime_quartile_order=("crime_quartile_order", "first"),
        food_insecurity_rate_pct=("food_insecure", lambda x: x.mean() * 100),
        fuel_insecurity_rate_pct=("fuel_insecure", lambda x: x.mean() * 100),
        both_insecure_rate_pct=("both_insecure", lambda x: x.mean() * 100),
        mean_imd_decile=("imd_decile", "mean"),
    )
    optional = {
        "crime_count_ye_dec2022": "crime_count",
        "population_mid2022": "population",
    }
    for source, output in optional.items():
        if source in dashboard:
            values = dashboard.groupby("lad_code", as_index=False)[source].first().rename(columns={source: output})
            borough = borough.merge(values, on="lad_code", how="left")
    return borough


def build_imd_summary(dashboard):
    return dashboard.groupby("imd_decile", as_index=False).agg(
        respondent_count=("user_id", "nunique"),
        food_insecurity_rate_pct=("food_insecure", lambda x: x.mean() * 100),
        fuel_insecurity_rate_pct=("fuel_insecure", lambda x: x.mean() * 100),
        both_insecure_rate_pct=("both_insecure", lambda x: x.mean() * 100),
    ).sort_values("imd_decile")


def build_missing_summary(dashboard):
    return pd.DataFrame({
        "column": dashboard.columns,
        "missing_count": dashboard.isna().sum().values,
        "missing_percentage": dashboard.isna().mean().mul(100).values,
        "unique_values": dashboard.nunique(dropna=True).values,
        "data_type": dashboard.dtypes.astype(str).values,
    }).sort_values("missing_percentage", ascending=False)


def export_powerbi_csvs(df, output_dir=POWERBI_OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dashboard = prepare_powerbi_dashboard(df)
    exports = {
        "survey_dashboard.csv": dashboard,
        "security_long.csv": build_security_long(dashboard),
        "demographic_summary.csv": build_group_summary(dashboard, EDA_DEMOGRAPHIC_COLS),
        "housing_summary.csv": build_group_summary(dashboard, EDA_HOUSING_COLS),
        "borough_summary.csv": build_borough_summary(dashboard),
        "imd_summary.csv": build_imd_summary(dashboard),
        "missing_summary.csv": build_missing_summary(dashboard),
    }
    for filename, table in exports.items():
        table.to_csv(output_dir / filename, index=False)
    print(f"Saved {len(exports)} Power BI CSV files to {output_dir}")
    return exports