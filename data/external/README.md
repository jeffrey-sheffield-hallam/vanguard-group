# External data provenance

All sources are published under the Open Government Licence (versions noted per file). Survey fieldwork window: 22 Nov - 5 Dec 2022; questions reference "the past year", so the target window is approximately Dec 2021 - Nov 2022.

## Files

### ons_csp_recorded_crime_ye_dec2022.xlsx
- Dataset: ONS, "Recorded crime data by Community Safety Partnership area", edition year ending December 2022 (`csptablesyedec22.xlsx`). Source data: Home Office police recorded crime.
- URL: https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/crimeandjustice/datasets/recordedcrimedatabycommunitysafetypartnershiparea/yearendingdecember2022/csptablesyedec22.xlsx
- Dataset page: https://www.ons.gov.uk/peoplepopulationandcommunity/crimeandjustice/datasets/recordedcrimedatabycommunitysafetypartnershiparea
- Vintage: year ending December 2022, published April 2023 (series discontinued October 2024; this edition remains published).
- Licence: Open Government Licence v3.0.

### ons_mid2022_population_estimates_lad.xlsx
- Dataset: ONS, "Estimates of the population for England and Wales", mid 2022 edition on 2023 local authority boundaries (`mye22tablesew2023geogs.xlsx`).
- URL: https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/populationandmigration/populationestimates/datasets/estimatesofthepopulationforenglandandwales/mid20222023localauthorityboundaires/mye22tablesew2023geogs.xlsx
- Dataset page: https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates/datasets/estimatesofthepopulationforenglandandwales
- Vintage: mid 2022 estimates (reference date 30 June 2022); revised July 2024 and again July 2025 (updated international migration estimates); file as published at the URL on the download date 2026-07-06. 2023 boundaries are identical to 2021 boundaries for all London boroughs.
- Licence: Open Government Licence v3.0.

### ons_os_private_outdoor_space_gb_april2020.xlsx
- Dataset: ONS/Ordnance Survey, "Access to garden space, Great Britain, April 2020" (private outdoor space reference tables, `osprivateoutdoorspacereferencetables.xlsx`).
- URL: https://www.ons.gov.uk/file?uri=/economy/environmentalaccounts/datasets/accesstogardensandpublicgreenspaceingreatbritain/accesstogardenspacegreatbritainapril2020/osprivateoutdoorspacereferencetables.xlsx
- Dataset page: https://www.ons.gov.uk/economy/environmentalaccounts/datasets/accesstogardensandpublicgreenspaceingreatbritain
- Vintage: April 2020 snapshot (OS AddressBase Plus epoch 74); published 14 May 2020; corrections of 20 May 2022 applied (country/LAD tabs only; the MSOA tab used here was unaffected).
- Licence: Open Government Licence v3.0. Required attribution: "Source: Ordnance Survey. (c) Crown copyright and database rights 2020 OS 100019153".

### ons_os_public_green_space_gb_april2020.xlsx
- Dataset: ONS/Ordnance Survey, "Access to public parks and playing fields, Great Britain, April 2020" (`ospublicgreenspacereferencetables.xlsx`).
- URL: https://www.ons.gov.uk/file?uri=/economy/environmentalaccounts/datasets/accesstogardensandpublicgreenspaceingreatbritain/accesstopublicparksandplayingfieldsgreatbritainapril2020/ospublicgreenspacereferencetables.xlsx
- Dataset page: https://www.ons.gov.uk/economy/environmentalaccounts/datasets/accesstogardensandpublicgreenspaceingreatbritain
- Vintage: April 2020 snapshot (OS Open Greenspace, April 2020); published 14 May 2020; corrected file of 20 May 2022 (LSOA tabs unaffected).
- Licence: Open Government Licence v3.0. Required attribution: "Source: Ordnance Survey Open Greenspace. Contains OS data (c) Crown copyright and database right 2020".

## THe flow

### crime_rate_per_1000 (lad_code level)
Read `ons_csp_recorded_crime_ye_dec2022.xlsx` sheet "Table C3" (header on row 8), keep Local Authority code/name and "Total recorded crime (excluding fraud)", filter codes starting E09 (33 rows). Read `ons_mid2022_population_estimates_lad.xlsx` sheet "MYE2 - Persons" (header on row 8), keep Code and "All ages", filter E09 (33 rows). Join on lad_code and compute crime_rate_per_1000 = crimes / population * 1000. Time window is year ending December 2022 (Jan-Dec 2022), the closest published approximation to the survey look-back window Dec 2021 - Nov 2022 (11 of 12 months overlap; monthly CSP-level data including the City of London is not published). ONS Table C3 is used as the single source for all 33 boroughs because MPS data structurally excludes the City of London (own police force) and the ONS total already excludes fraud, giving one consistent definition. The City of London rate (7,263 / 11,457 * 1000 = 634 per 1,000) is a real, documented value but is inflated by the daytime population relative to residents (ONS suppresses its own per-1,000 rate for the City in Table C5); only 5 survey respondents live there.

### green_space_pct (msoa_code level)
Read `ons_os_private_outdoor_space_gb_april2020.xlsx` sheet "MSOA gardens" (two header rows), take the "Property type: Total" block's "Percentage of adresses with private outdoor space" (a 0-1 fraction at source, multiplied by 100; the misspelling is ONS's own), filter to London via the LAD code column. April 2020 is the most recent MSOA level release of this series; the stock of gardens changes slowly, so it is used as the nearest available vintage to the Dec 2021 - Nov 2022 exposure window.

### park_distance_m (msoa_code level)
Read `ons_os_public_green_space_gb_april2020.xlsx` sheet "LSOA Parks only" (parks and public gardens; the companion "Parks and Playing Fields" sheet includes possibly-private playing fields and is not used), filter to London LSOAs, and aggregate "Average distance to nearest Park or Public Garden (m)" to MSOA as a mean weighted by "Number of postcodes within built up area". The LSOA figures are themselves means over built up area postcodes, so the postcode weighted mean exactly reconstructs the MSOA level postcode mean. 
