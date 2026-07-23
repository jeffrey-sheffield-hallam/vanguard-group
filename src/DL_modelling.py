from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.keras.utils.set_random_seed(RANDOM_STATE)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

TARGET_CONFIG = {
    "food": {"path": PROJECT_ROOT / "data/finalised_dataset/food_security_fully_coded_v2.csv", "score_col": "food_security_score", "bins": [-np.inf, 1, 3, 6, np.inf]},
    "fuel": {"path": PROJECT_ROOT / "data/finalised_dataset/fuel_security_fully_coded_v2.csv", "score_col": "fuel-security_score", "bins": [-np.inf, 0, 1, 3, np.inf]}
}

CLASS_NAMES = ["High", "Marginal", "Low", "Very Low"]
NOMINAL_FEATURES = ["Gender_Code", "Work_Schedule_Code", "Household_type_code", "Housing_tenure_group_code"]
ORDINAL_FEATURES = ["Age_range_Code", "Income_Code", "imd_decile"]
CONTINUOUS_FEATURES = ["Life_satisfaction", "Isolation_score", "Social_support_score", "green_space_pct", "park_distance_m", "crime_rate_per_1000"]


def classification_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0)
    }


class EpochMetrics(tf.keras.callbacks.Callback):
    def __init__(self, X_train, y_train, X_val, y_val):
        super().__init__()
        self.X_train, self.y_train = X_train, np.asarray(y_train)
        self.X_val, self.y_val = X_val, np.asarray(y_val)
        self.records = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_pred = self.model.predict(self.X_train, verbose=0).argmax(axis=1)
        val_pred = self.model.predict(self.X_val, verbose=0).argmax(axis=1)
        row = {
            "epoch": epoch + 1,
            "loss": logs.get("loss"),
            "val_loss": logs.get("val_loss"),
            "accuracy": logs.get("accuracy"),
            "val_accuracy": logs.get("val_accuracy"),
            "precision_weighted": precision_score(self.y_train, train_pred, average="weighted", zero_division=0),
            "val_precision_weighted": precision_score(self.y_val, val_pred, average="weighted", zero_division=0),
            "recall_weighted": recall_score(self.y_train, train_pred, average="weighted", zero_division=0),
            "val_recall_weighted": recall_score(self.y_val, val_pred, average="weighted", zero_division=0),
            "f1_weighted": f1_score(self.y_train, train_pred, average="weighted", zero_division=0),
            "val_f1_weighted": f1_score(self.y_val, val_pred, average="weighted", zero_division=0),
            "f1_macro": f1_score(self.y_train, train_pred, average="macro", zero_division=0),
            "val_f1_macro": f1_score(self.y_val, val_pred, average="macro", zero_division=0)
        }
        logs["f1_macro"], logs["val_f1_macro"] = row["f1_macro"], row["val_f1_macro"]
        self.records.append(row)
        print(f"precision: {row['precision_weighted']:.4f} - recall: {row['recall_weighted']:.4f} - f1: {row['f1_weighted']:.4f} - macro_f1: {row['f1_macro']:.4f} - val_precision: {row['val_precision_weighted']:.4f} - val_recall: {row['val_recall_weighted']:.4f} - val_f1: {row['val_f1_weighted']:.4f} - val_macro_f1: {row['val_f1_macro']:.4f}")


def prepare_data(target, path=None):
    config = TARGET_CONFIG[target]
    df = pd.read_csv(path or config["path"])
    df["target"] = pd.cut(df[config["score_col"]], bins=config["bins"], labels=[0, 1, 2, 3]).astype(int)
    print("\nTarget distribution:")
    print(df["target"].map(dict(enumerate(CLASS_NAMES))).value_counts())
    y = df["target"]
    X = df.drop(columns=["participant_id", config["score_col"], "target", "Lad_code_code", "msoa_code_code"])
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp)
    return X_train, X_val, X_test, y_train, y_val, y_test


def make_preprocessor():
    nominal_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    return ColumnTransformer([
        ("nominal", nominal_pipeline, NOMINAL_FEATURES),
        ("ordinal", numeric_pipeline, ORDINAL_FEATURES),
        ("continuous", numeric_pipeline, CONTINUOUS_FEATURES)
    ])


def build_tensorflow_mlp(input_dim, hidden_layers=(64, 32), dropout=0.2, learning_rate=0.001):
    model = tf.keras.Sequential([tf.keras.layers.Input(shape=(input_dim,))])
    for units in hidden_layers:
        model.add(tf.keras.layers.Dense(units, activation="relu"))
        model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(4, activation="softmax"))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_tensorflow_mlp(target, path=None, epochs=100, batch_size=32, learning_rate=0.001, hidden_layers=(64, 32), dropout=0.2, patience=10, use_class_weights=True, class_weight_power=0.5, save=True):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(target, path)
    preprocessor = make_preprocessor()
    X_train = preprocessor.fit_transform(X_train).astype("float32")
    X_val = preprocessor.transform(X_val).astype("float32")
    X_test = preprocessor.transform(X_test).astype("float32")
    model = build_tensorflow_mlp(X_train.shape[1], hidden_layers, dropout, learning_rate)
    model.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_f1_macro", mode="max", patience=patience, restore_best_weights=True, verbose=1)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1)
    epoch_metrics = EpochMetrics(X_train, y_train, X_val, y_val)
    class_weights = None
    if use_class_weights:
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train) ** class_weight_power
        class_weights = dict(zip(classes, weights))
        print("\nClass weights:", class_weights)

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=[epoch_metrics, early_stopping, reduce_lr], class_weight=class_weights, verbose=1)
    history_df = pd.DataFrame(epoch_metrics.records)
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    probabilities = model.predict(X_test, verbose=0)
    predictions = probabilities.argmax(axis=1)
    metrics = classification_metrics(y_test, predictions)
    metrics.update({"test_loss": test_loss, "test_accuracy": test_accuracy})
    metrics_df = pd.DataFrame(metrics.items(), columns=["metric", "value"])
    report_df = pd.DataFrame(classification_report(y_test, predictions, labels=[0, 1, 2, 3], target_names=CLASS_NAMES, output_dict=True, zero_division=0)).T
    prediction_df = pd.DataFrame({
        "actual": y_test.to_numpy(),
        "predicted": predictions,
        "actual_label": [CLASS_NAMES[i] for i in y_test],
        "predicted_label": [CLASS_NAMES[i] for i in predictions],
        "correct": y_test.to_numpy() == predictions
    })
    for i, name in enumerate(CLASS_NAMES):
        prediction_df[f"probability_{name.lower().replace(' ', '_')}"] = probabilities[:, i]

    print("\nTest metrics:")
    print(metrics_df.round(4).to_string(index=False))
    print("\nClassification report:")
    print(report_df.round(4))

    output_dir = PROJECT_ROOT / "outputs" / "training" / target
    model_dir = PROJECT_ROOT / "models" / target
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    loss_fig, loss_ax = plt.subplots(figsize=(9, 5))
    loss_ax.plot(history_df["epoch"], history_df["loss"], label="Training loss")
    loss_ax.plot(history_df["epoch"], history_df["val_loss"], label="Validation loss")
    loss_ax.set(xlabel="Epoch", ylabel="Loss", title=f"{target.title()} Security: Loss")
    loss_ax.legend()
    loss_ax.grid(alpha=0.3)
    loss_fig.tight_layout()

    accuracy_fig, accuracy_ax = plt.subplots(figsize=(9, 5))
    accuracy_ax.plot(history_df["epoch"], history_df["accuracy"], label="Training accuracy")
    accuracy_ax.plot(history_df["epoch"], history_df["val_accuracy"], label="Validation accuracy")
    accuracy_ax.set(xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1), title=f"{target.title()} Security: Accuracy")
    accuracy_ax.legend()
    accuracy_ax.grid(alpha=0.3)
    accuracy_fig.tight_layout()

    metric_fig, metric_ax = plt.subplots(figsize=(10, 5))
    metric_ax.plot(history_df["epoch"], history_df["val_accuracy"], label="Accuracy")
    metric_ax.plot(history_df["epoch"], history_df["val_precision_weighted"], label="Precision")
    metric_ax.plot(history_df["epoch"], history_df["val_recall_weighted"], label="Recall")
    metric_ax.plot(history_df["epoch"], history_df["val_f1_weighted"], label="Weighted F1")
    metric_ax.plot(history_df["epoch"], history_df["val_f1_macro"], label="Macro F1")
    metric_ax.set(xlabel="Epoch", ylabel="Score", ylim=(0, 1), title=f"{target.title()} Security: Validation Metrics")
    metric_ax.legend()
    metric_ax.grid(alpha=0.3)
    metric_fig.tight_layout()

    matrix = confusion_matrix(y_test, predictions, labels=[0, 1, 2, 3])
    confusion_fig, confusion_ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(matrix, display_labels=CLASS_NAMES).plot(ax=confusion_ax, cmap="Blues", values_format="d", colorbar=False)
    confusion_ax.set_title(f"{target.title()} Security: Confusion Matrix")
    confusion_fig.tight_layout()

    paths = {
        "loss_plot": output_dir / "tensorflow_loss_history.png",
        "accuracy_plot": output_dir / "tensorflow_accuracy_history.png",
        "metrics_plot": output_dir / "tensorflow_validation_metrics_history.png",
        "confusion_matrix_plot": output_dir / "tensorflow_confusion_matrix.png"
    }

    if save:
        model.save(model_dir / "tensorflow_mlp.keras")
        joblib.dump(preprocessor, model_dir / "tensorflow_preprocessor.joblib")
        history_df.to_csv(output_dir / "tensorflow_training_history.csv", index=False)
        metrics_df.to_csv(output_dir / "tensorflow_test_metrics.csv", index=False)
        report_df.to_csv(output_dir / "tensorflow_classification_report.csv")
        prediction_df.to_csv(output_dir / "tensorflow_test_predictions.csv", index=False)
        with open(output_dir / "tensorflow_test_metrics.json", "w") as file:
            json.dump(metrics, file, indent=2)
        loss_fig.savefig(paths["loss_plot"], dpi=300, bbox_inches="tight")
        accuracy_fig.savefig(paths["accuracy_plot"], dpi=300, bbox_inches="tight")
        metric_fig.savefig(paths["metrics_plot"], dpi=300, bbox_inches="tight")
        confusion_fig.savefig(paths["confusion_matrix_plot"], dpi=300, bbox_inches="tight")

    plt.show()
    return {
        "model": model,
        "preprocessor": preprocessor,
        "history": history_df,
        "metrics": metrics_df,
        "classification_report": report_df,
        "predictions": prediction_df,
        "confusion_matrix": matrix,
        "probabilities": probabilities,
        "y_test": y_test,
        "plot_paths": paths
    }
