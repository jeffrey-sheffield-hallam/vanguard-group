# Finalised Coded Dataset

## Drivers of Food Insecurity in London, United Kingdom

This directory contains the finalised coded datasets prepared for the VANGUARD project investigating the drivers of food insecurity and fuel/energy insecurity among London residents.

The datasets were produced following reconciliation of the original cleaned dataset and the newer coded dataset. Missing variables were restored from the original dataset after confirming that both files contained the same 2,886 participant observations in the same row order.

The final datasets preserve the original feature structure and use simple integer coding for categorical variables. Model-specific transformations such as one-hot encoding, scaling, imputation, class weighting, or resampling have **not** been applied at this stage.

---

## Dataset Files

### `coded_features_fully_coded_v2.csv`

The master canonical coded dataset.

**Dimensions:** 2,886 rows × 18 columns

This dataset contains both prediction outcomes:

* `food_security_score`
* `fuel-security_score`

It should be treated as the authoritative fully coded dataset before target-specific modelling.

### `food_security_fully_coded_v2.csv`

Target-specific dataset for food security modelling.

**Target:**

`food_security_score`

The `fuel-security_score` variable has been completely removed to prevent the fuel-security outcome from being available to the food-security model.

### `fuel_security_fully_coded_v2.csv`

Target-specific dataset for fuel/energy security modelling.

**Target:**

`fuel-security_score`

The `food_security_score` variable has been completely removed to prevent the food-security outcome from being available to the fuel-security model.

---

# Dataset Reconciliation

The canonical dataset was produced from:

* `feature_engineering/Features_File_cleaned.csv`
* `feature_engineering/coded_features.csv`

**2,886 substantive participant observations**

Row correspondence between the datasets was confirmed before restoring missing variables.

The following variables were restored from the previous cleaned dataset:

* Participant identifier
* Gender
* Work Schedule
* Crime rate per 1,000 population

The original participant identifier `#` was renamed:

`participant_id`

This variable is retained for traceability and dataset auditing but **must not be used as a machine learning predictor**.

---

# Final Dataset Structure

The canonical dataset contains:

| Variable                    | Description                                        |
| --------------------------- | -------------------------------------------------- |
| `participant_id`            | Unique participant identifier                      |
| `Gender_Code`               | Coded gender                                       |
| `Work_Schedule_Code`        | Cleaned and coded employment/work status           |
| `Age_range_Code`            | Coded age range                                    |
| `Household_type_code`       | Coded household type                               |
| `Life_satisfaction`         | Life satisfaction measure                          |
| `Isolation_score`           | Social isolation score                             |
| `Social_support_score`      | Social support score                               |
| `food_security_score`       | Food security outcome score                        |
| `Housing_tenure_group_code` | Coded housing tenure                               |
| `fuel-security_score`       | Fuel/energy security outcome score                 |
| `Income_Code`               | Coded income group                                 |
| `Lad_code_code`             | Local authority geographic identifier              |
| `msoa_code_code`            | MSOA geographic identifier                         |
| `imd_decile`                | Index of Multiple Deprivation decile               |
| `green_space_pct`           | Percentage of green space associated with the area |
| `park_distance_m`           | Distance to park/green space in metres             |
| `crime_rate_per_1000`       | Area crime rate per 1,000 population               |

---

# Newly Finalised Coding

## Gender

`Gender` was converted to `Gender_Code`.

| Code | Category           |
| ---: | ------------------ |
|    0 | Prefer not to say  |
|    1 | Female             |
|    2 | Male               |
|    3 | Non binary / other |

These codes represent categories and should not automatically be interpreted as continuous numerical quantities.

---

# Work Schedule

The original `Work_Schedule` variable required additional cleaning before coding.

The source data contained:

* Differences in capitalisation
* Spelling variations
* Free-text responses
* Multiple employment statuses within a single response
* Variations describing similar employment circumstances

Rather than assigning separate codes to every raw text variation, responses were consolidated into interpretable employment categories.

The final `Work_Schedule_Code` structure is:

| Code | Category                        |
| ---: | ------------------------------- |
|    0 | Unknown                         |
|    1 | Prefer not to say               |
|    2 | Working full-time / contract    |
|    3 | Working part-time / casual      |
|    4 | Self-employed / freelance       |
|    5 | Retired                         |
|    6 | Student                         |
|    7 | Unemployed / seeking work       |
|    8 | Long-term sick / unable to work |
|    9 | Caring / homemaking / leave     |
|   10 | Volunteering                    |
|   11 | Apprentice                      |
|   12 | Other economically inactive     |
|   13 | Other / unclear                 |

The complete transformation between the original responses and their final codes is documented in:

`work_schedule_raw_to_code_v2.csv`

---

# Existing Coded Features

The following variables were already coded in `coded_features.csv` and were retained:

* `Age_range_Code`
* `Household_type_code`
* `Housing_tenure_group_code`
* `Income_Code`
* `Lad_code_code`
* `msoa_code_code`

Their integer representation does **not** necessarily mean that they should be treated as continuous numerical variables during modelling.

For example, household type and housing tenure represent categories, while LAD and MSOA codes are geographic identifiers.

Model-specific preprocessing should account for these semantic differences.

---

# Engineered Environmental Features

The final dataset contains the environmental and area-level features developed for the project:

### `imd_decile`

Represents area-level deprivation using the Index of Multiple Deprivation.

### `green_space_pct`

Represents access/exposure to green space within the relevant geographical area.

### `park_distance_m`

Represents distance to relevant park/green-space infrastructure.

### `crime_rate_per_1000`

Represents the area crime rate per 1,000 population.

`crime_rate_per_1000` was present in the previous cleaned dataset but absent from the newer coded dataset. It was restored during dataset reconciliation after participant-level row correspondence was confirmed.

---

# Prediction Targets

## Food Security

The original `food_security_score` ranges from 0 to 9.

For classification, the agreed categories are:

| Score | Classification         |
| ----: | ---------------------- |
|   0–1 | High food security     |
|   2–3 | Marginal food security |
|   4–6 | Low food security      |
|   7–9 | Very low food security |

This produces a four-class **ordinal classification problem**.

---

## Fuel/Energy Security

The original `fuel-security_score` ranges from 0 to 6.

For classification, the agreed categories are:

| Score | Classification         |
| ----: | ---------------------- |
|     0 | High fuel security     |
|     1 | Marginal fuel security |
|   2–4 | Low fuel security      |
|   5–6 | Very low fuel security |

This also produces a four-class **ordinal classification problem**.

The raw target scores remain in the final coded datasets. Conversion to classification labels should occur as part of the documented modelling workflow rather than replacing the original target information in the canonical dataset.

---

# Target Separation

Food security and fuel security must be modelled independently.

For food security:

`Predictor Features → food_security_score`

`fuel-security_score` must not be available during preprocessing, feature selection, model training, validation, hyperparameter optimisation, testing, explainability analysis, or inference.

For fuel security:

`Predictor Features → fuel-security_score`

`food_security_score` must similarly be excluded throughout the modelling pipeline.

The target-specific CSV files supplied in this directory already enforce this initial separation.

---

# Important Modelling Considerations

## Participant ID

`participant_id` exists only for traceability.

It must be removed from the predictor matrix before model training.

## Geographic Codes

`Lad_code_code` and `msoa_code_code` are geographic identifiers.

Their numerical values do not represent continuous quantities and should not automatically be interpreted as such by machine learning models.

They may also be useful for geographical lookup, area-level feature retrieval, spatial analysis, and deployment rather than as direct predictors.

## Categorical Codes

Integer coding does not imply numerical magnitude.

Variables such as:

* `Gender_Code`
* `Work_Schedule_Code`
* `Household_type_code`
* `Housing_tenure_group_code`

should be treated according to their categorical semantics during model-specific preprocessing.

## Ordinal Variables

Variables such as age range, income group and IMD decile contain meaningful ordering and should be treated accordingly where appropriate.

---

# Data Leakage

Several safeguards must be maintained during modelling:

1. Never use `participant_id` as a predictor.
2. Never use `fuel-security_score` to predict food security.
3. Never use `food_security_score` to predict fuel security.
4. Do not introduce raw questionnaire variables that directly construct the corresponding target score as predictors.
5. Fit learned preprocessing operations using training data only.
6. Perform any oversampling or resampling only within the training/CV pipeline.

---

# Model-Specific Preprocessing

This dataset is **fully coded but not model-preprocessed**.

No:

* One-hot encoding
* Feature scaling
* Standardisation
* SMOTE
* Class balancing
* Feature selection
* PCA
* Model-specific transformation

has been permanently applied to the canonical CSV.

These operations should be performed within reproducible modelling pipelines after the train/test split.

This distinction allows different algorithms to receive appropriate preprocessing while maintaining a single authoritative source dataset.

---

The finalised coded datasets should now serve as the common data source for subsequent machine learning and deep learning experiments.
