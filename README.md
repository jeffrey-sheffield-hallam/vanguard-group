# vanguard-group

vanguard-group/
│
├── README.md
├── requirements.txt
├── environment.yml
├── .gitignore
├── LICENSE
│
├── data/
│   ├── raw/
│   │   └── Food_insecurity_Raw_Survey_Responses.xlxs
│   ├── unified_csv/
│   │   └── 
│   ├── processed/
│   │   └── 
│   ├── featured_engineering/
│   │   └── 
│   └── README.md
│       
│
├── docs/
│   ├── project_brief.md
│   ├── data_dictionary.md
│   ├── target_definition.md
│   ├── methodology.md
│   └── client_handover_notes.md
│
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_exploratory_data_analysis.ipynb
│   ├── 05_food_insecurity_modelling.ipynb
│   ├── 06_fuel_insecurity_modelling.ipynb
│   ├── 07_tensorflow_model.ipynb
│   └── 08_explainability.ipynb
│
├── src/
│   ├── data_cleaning.py
│   ├── target_creation.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── modelling.py
│   ├── tensorflow_model.py
│   ├── evaluation.py
│   ├── explainability.py
│   ├── visualisation.py
│   └── utils.py
│
├── app/
│   ├── app.py
│   ├── pages/
│   │   ├── 1_Project_Overview.py
│   │   ├── 2_Data_Explorer.py
│   │   ├── 3_EDA_Dashboard.py
│   │   ├── 4_Model_Prediction.py
│   │   ├── 5_Model_Comparison.py
│   │   └── 6_Batch_Prediction.py 
│   └── app_utils.py
│
├── models/
│   ├── food_insecurity/
│   │   ├── logistic_regression.joblib
│   │   ├── random_forest.joblib
│   │   ├── xgboost.joblib
│   │   ├── catboost.joblib
│   │   ├── tensorflow_mlp.keras
│   │   └── model_metadata.json
│   │
│   └── fuel_insecurity/
│       ├── logistic_regression.joblib
│       ├── random_forest.joblib
│       ├── xgboost.joblib
│       ├── catboost.joblib
│       ├── tensorflow_mlp.keras
│       └── model_metadata.json
│
├── outputs/
│   ├── figures/
│   │   ├── eda/
│   │   ├── model_performance/
│   │   └── 
│   ├── tables/
│   │   
│   └── reports/
│
├── reports/
│   ├── final_client_report.md
│   ├── final_client_report.pdf
│   └── presentation_slides.pptx
│
└──  tests/


