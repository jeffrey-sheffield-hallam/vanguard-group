from datetime import datetime
from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "featured_engineering" / "Features_File_cleaned.csv"
TRAINING_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "training"
MODELS_DIR = PROJECT_ROOT / "models"

FOOD_TARGET = "food_security_score"
FUEL_TARGET = "fuel-security_score"
TARGET_COLS = [FOOD_TARGET, FUEL_TARGET]
ID_COLS = ["#"]
RANDOM_STATE = 42

TARGETS = {
    "food": {"name": "food_insecurity", "column": FOOD_TARGET, "folder": "food_insecurity"},
    "food_insecurity": {"name": "food_insecurity", "column": FOOD_TARGET, "folder": "food_insecurity"},
    FOOD_TARGET: {"name": "food_insecurity", "column": FOOD_TARGET, "folder": "food_insecurity"},
    "fuel": {"name": "fuel_insecurity", "column": FUEL_TARGET, "folder": "fuel_insecurity"},
    "fuel_insecurity": {"name": "fuel_insecurity", "column": FUEL_TARGET, "folder": "fuel_insecurity"},
    FUEL_TARGET: {"name": "fuel_insecurity", "column": FUEL_TARGET, "folder": "fuel_insecurity"},
}

MODEL_FILES = {
    "logistic_regression": "logistic_regression.joblib",
    "random_forest": "random_forest.joblib",
    "xgboost": "xgboost.joblib",
    "catboost": "catboost.joblib",
    "tensorflow_mlp": "tensorflow_mlp.keras",
}

ML_MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost", "catboost"]
ALL_MODEL_NAMES = ML_MODEL_NAMES + ["tensorflow_mlp"]
METRIC_COLS = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "f1_macro"]


def load_feature_data(path=DATA_PATH):
    return pd.read_csv(path)


def resolve_target(target):
    return TARGETS[str(target)]


def target_model_dir(target, models_dir=MODELS_DIR):
    path = Path(models_dir) / resolve_target(target)["folder"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def target_training_dir(target, output_dir=TRAINING_OUTPUT_DIR):
    path = Path(output_dir) / resolve_target(target)["folder"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def metadata_path(target, models_dir=MODELS_DIR):
    return target_model_dir(target, models_dir) / "model_metadata.json"


def split_data(X, y):
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1875,
        stratify=y_train_val, random_state=RANDOM_STATE
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def make_preprocessor(X):
    categorical_cols = X.select_dtypes(include=["object", "category", "string"]).columns
    numeric_cols = X.columns.difference(categorical_cols)

    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("numeric", numeric, numeric_cols),
        ("categorical", categorical, categorical_cols),
    ])


def prepare_target_data(target="food", path=DATA_PATH):
    info = resolve_target(target)
    df = load_feature_data(path)
    feature_cols = [c for c in df.columns if c not in TARGET_COLS + ID_COLS]
    data = df[feature_cols + [info["column"]]].dropna(subset=[info["column"]])

    encoder = LabelEncoder()
    y = encoder.fit_transform(data[info["column"]])
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data[feature_cols], y)

    return {
        "target": info["name"],
        "target_col": info["column"],
        "feature_cols": feature_cols,
        "classes": encoder.classes_.tolist(),
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }


def classification_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_ready(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_ready(val) for val in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def load_metadata(target, models_dir=MODELS_DIR):
    path = metadata_path(target, models_dir)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    info = resolve_target(target)
    return {"target": info["name"], "target_col": info["column"], "models": {}}


def update_metadata(target, model_name, entry, models_dir=MODELS_DIR):
    path = metadata_path(target, models_dir)
    metadata = load_metadata(target, models_dir)
    metadata["models"][model_name] = json_ready(entry)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def save_training_log(target, model_name, log_df):
    out_dir = target_training_dir(target)
    log_path = out_dir / f"{model_name}_training_log.csv"
    plot_path = out_dir / f"{model_name}_training_log.png"

    log_df.to_csv(log_path, index=False)
    fig, ax = plt.subplots(figsize=(8, 5))

    if "epoch" in log_df.columns:
        for col in log_df.columns:
            if col != "epoch":
                ax.plot(log_df["epoch"], log_df[col], label=col)
        ax.set_xlabel("Epoch")
        ax.legend()
    else:
        values = log_df.iloc[0][METRIC_COLS]
        ax.bar(values.index, values.values)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=35)

    ax.set_title(f"{resolve_target(target)['name']} - {model_name} training log")
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
    return log_path, plot_path


def save_model_result(target, model_name, model, data, val_metrics, params, extra=None):
    model_dir = target_model_dir(target)
    model_path = model_dir / MODEL_FILES[model_name]
    joblib.dump(model, model_path)

    log_df = pd.DataFrame([{ "split": "validation", **val_metrics }])
    log_path, plot_path = save_training_log(target, model_name, log_df)

    entry = {
        "model_path": model_path,
        "target_col": data["target_col"],
        "classes": data["classes"],
        "params": params,
        "val_metrics": val_metrics,
        "training_log_path": log_path,
        "training_plot_path": plot_path,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        entry.update(extra)
    update_metadata(target, model_name, entry)
    return model_path, log_path, plot_path


def train_sklearn_model(target, model_name, estimator, path=DATA_PATH, save=True, params=None):
    data = prepare_target_data(target, path)
    model = Pipeline([
        ("preprocess", make_preprocessor(data["X_train"])),
        ("model", estimator),
    ])
    model.fit(data["X_train"], data["y_train"])

    val_pred = model.predict(data["X_val"])
    val_metrics = classification_metrics(data["y_val"], val_pred)
    result = {"model_name": model_name, "model": model, "data": data, "val_metrics": val_metrics}

    if save:
        model_path, log_path, plot_path = save_model_result(
            target, model_name, model, data, val_metrics, params or {}
        )
        result.update({"model_path": model_path, "training_log_path": log_path, "training_plot_path": plot_path})
    return result


def train_logistic_regression(target="food", path=DATA_PATH, C=1.0, max_iter=1000,
                              class_weight=None, save=True, **kwargs):
    params = {"C": C, "max_iter": max_iter, "class_weight": class_weight, **kwargs}
    estimator = LogisticRegression(random_state=RANDOM_STATE, **params)
    return train_sklearn_model(target, "logistic_regression", estimator, path, save, params)


def train_random_forest(target="food", path=DATA_PATH, n_estimators=300, max_depth=None,
                        min_samples_split=2, class_weight=None, save=True, **kwargs):
    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "class_weight": class_weight,
        **kwargs,
    }
    estimator = RandomForestClassifier(random_state=RANDOM_STATE, **params)
    return train_sklearn_model(target, "random_forest", estimator, path, save, params)


def train_xgboost(target="food", path=DATA_PATH, n_estimators=300, learning_rate=0.05,
                  max_depth=4, subsample=1.0, colsample_bytree=1.0, save=True, **kwargs):
    params = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "eval_metric": "mlogloss",
        **kwargs,
    }
    estimator = XGBClassifier(random_state=RANDOM_STATE, **params)
    return train_sklearn_model(target, "xgboost", estimator, path, save, params)


def train_catboost(target="food", path=DATA_PATH, iterations=300, learning_rate=0.05,
                   depth=6, save=True, **kwargs):
    params = {"iterations": iterations, "learning_rate": learning_rate, "depth": depth,
              "verbose": False, **kwargs}
    estimator = CatBoostClassifier(random_seed=RANDOM_STATE, **params)
    return train_sklearn_model(target, "catboost", estimator, path, save, params)


def build_tensorflow_mlp(input_dim, n_classes, hidden_layers=(64, 32), dropout=0.2,
                         learning_rate=0.001):
    import tensorflow as tf

    model = tf.keras.Sequential([tf.keras.layers.Input(shape=(input_dim,))])
    for units in hidden_layers:
        model.add(tf.keras.layers.Dense(units, activation="relu"))
        if dropout:
            model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(n_classes, activation="softmax"))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_tensorflow_mlp(target="food", path=DATA_PATH, epochs=30, learning_rate=0.001,
                         batch_size=32, hidden_layers=(64, 32), dropout=0.2, save=True):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(RANDOM_STATE)
    data = prepare_target_data(target, path)
    preprocessor = make_preprocessor(data["X_train"])
    X_train = preprocessor.fit_transform(data["X_train"])
    X_val = preprocessor.transform(data["X_val"])
    X_test = preprocessor.transform(data["X_test"])

    model = build_tensorflow_mlp(
        X_train.shape[1], len(data["classes"]), hidden_layers, dropout, learning_rate
    )
    history = model.fit(
        X_train, data["y_train"], validation_data=(X_val, data["y_val"]),
        epochs=epochs, batch_size=batch_size, verbose=0
    )

    val_pred = model.predict(X_val, verbose=0).argmax(axis=1)
    val_metrics = classification_metrics(data["y_val"], val_pred)
    result = {
        "model_name": "tensorflow_mlp",
        "model": model,
        "preprocessor": preprocessor,
        "data": data,
        "val_metrics": val_metrics,
    }

    if save:
        model_dir = target_model_dir(target)
        model_path = model_dir / MODEL_FILES["tensorflow_mlp"]
        preprocessor_path = model_dir / "tensorflow_mlp_preprocessor.joblib"
        model.save(model_path)
        joblib.dump(preprocessor, preprocessor_path)

        history_df = pd.DataFrame(history.history)
        history_df.insert(0, "epoch", range(1, len(history_df) + 1))
        log_path, plot_path = save_training_log(target, "tensorflow_mlp", history_df)

        params = {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "hidden_layers": list(hidden_layers),
            "dropout": dropout,
        }
        update_metadata(target, "tensorflow_mlp", {
            "model_path": model_path,
            "preprocessor_path": preprocessor_path,
            "target_col": data["target_col"],
            "classes": data["classes"],
            "params": params,
            "val_metrics": val_metrics,
            "training_log_path": log_path,
            "training_plot_path": plot_path,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        })
        result.update({"model_path": model_path, "preprocessor_path": preprocessor_path,
                       "training_log_path": log_path, "training_plot_path": plot_path})

    result["X_test_processed"] = X_test
    return result


def train_all_ml_models(target="food", path=DATA_PATH):
    return {
        "logistic_regression": train_logistic_regression(target, path),
        "random_forest": train_random_forest(target, path),
        "xgboost": train_xgboost(target, path),
        "catboost": train_catboost(target, path),
    }


def train_food_ml_models(path=DATA_PATH):
    return train_all_ml_models("food", path)


def train_fuel_ml_models(path=DATA_PATH):
    return train_all_ml_models("fuel", path)


def train_both_tensorflow_models(path=DATA_PATH, **kwargs):
    return {
        "food_insecurity": train_tensorflow_mlp("food", path, **kwargs),
        "fuel_insecurity": train_tensorflow_mlp("fuel", path, **kwargs),
    }


def saved_model_names(target="food"):
    model_dir = target_model_dir(target)
    return [name for name, filename in MODEL_FILES.items() if (model_dir / filename).exists()]


def load_saved_model(target, model_name):
    model_dir = target_model_dir(target)
    if model_name == "tensorflow_mlp":
        import tensorflow as tf

        return {
            "model": tf.keras.models.load_model(model_dir / MODEL_FILES[model_name]),
            "preprocessor": joblib.load(model_dir / "tensorflow_mlp_preprocessor.joblib"),
        }
    return {"model": joblib.load(model_dir / MODEL_FILES[model_name])}


def predict_saved_model(target, model_name, X):
    saved = load_saved_model(target, model_name)
    if model_name == "tensorflow_mlp":
        X = saved["preprocessor"].transform(X)
        return saved["model"].predict(X, verbose=0).argmax(axis=1)
    return saved["model"].predict(X)


def test_saved_model(target="food", model_name="random_forest", path=DATA_PATH):
    data = prepare_target_data(target, path)
    y_pred = predict_saved_model(target, model_name, data["X_test"])
    metrics = classification_metrics(data["y_test"], y_pred)

    metadata = load_metadata(target)
    entry = metadata["models"].get(model_name, {})
    entry["test_metrics"] = metrics
    entry["tested_at"] = datetime.now().isoformat(timespec="seconds")
    update_metadata(target, model_name, entry)

    return {"model": model_name, **metrics}


def test_saved_models(target="food", path=DATA_PATH, model_names=None):
    names = saved_model_names(target) if model_names is None else model_names
    rows = [test_saved_model(target, name, path) for name in names]
    results = pd.DataFrame(rows).sort_values("accuracy", ascending=False)

    out_dir = target_training_dir(target)
    results_path = out_dir / "test_results.csv"
    results.to_csv(results_path, index=False)
    return results


def plot_model_results(target="food", path=DATA_PATH, results=None, model_names=None):
    results = test_saved_models(target, path, model_names) if results is None else results.copy()
    out_dir = target_training_dir(target)
    csv_path = out_dir / "model_results_table.csv"
    plot_path = out_dir / "model_results_comparison.png"

    results.to_csv(csv_path, index=False)
    table_df = results[["model"] + METRIC_COLS].round(3)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), gridspec_kw={"height_ratios": [2, 1]})
    results.plot(x="model", y=METRIC_COLS, kind="bar", ax=axes[0])
    axes[0].set_title(f"{resolve_target(target)['name']} model results")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Score")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].legend(loc="lower right")

    axes[1].axis("off")
    table = axes[1].table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
    return results, csv_path, plot_path


def plot_all_model_results(path=DATA_PATH):
    food_results, _, _ = plot_model_results("food", path)
    fuel_results, _, _ = plot_model_results("fuel", path)
    food_results.insert(0, "target", "food_insecurity")
    fuel_results.insert(0, "target", "fuel_insecurity")

    results = pd.concat([food_results, fuel_results], ignore_index=True)
    out_dir = TRAINING_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "all_model_results_table.csv"
    plot_path = out_dir / "all_model_results_comparison.png"
    results.to_csv(csv_path, index=False)

    labels = results["target"] + " - " + results["model"]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(labels, results["accuracy"])
    ax.set_title("All model test accuracy")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
    return results, csv_path, plot_path
