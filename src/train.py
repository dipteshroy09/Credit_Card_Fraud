from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COLUMN = "Class"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def best_f1_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[float, float, float, float]:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    if len(thresholds) == 0:
        return 0.5, 0.0, 0.0, 0.0

    precisions = precisions[:-1]
    recalls = recalls[:-1]
    denom = precisions + recalls
    f1s = np.where(denom > 0, 2 * precisions * recalls / denom, 0.0)

    idx = int(np.argmax(f1s))
    return float(thresholds[idx]), float(precisions[idx]), float(recalls[idx]), float(f1s[idx])


def build_pipeline() -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[("scale", StandardScaler(), FEATURE_COLUMNS)],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="saga",
        random_state=42,
    )

    return Pipeline(steps=[("preprocess", preprocess), ("model", model)])


def train_and_save(data_path: Path, model_path: Path, metadata_path: Path) -> None:
    df = pd.read_csv(data_path)

    required = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].astype(int)

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y,
        random_state=42,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp,
        y_tmp,
        test_size=0.5,
        stratify=y_tmp,
        random_state=42,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    val_proba = pipeline.predict_proba(X_val)[:, 1]
    threshold, val_precision, val_recall, val_f1 = best_f1_threshold(y_val.to_numpy(), val_proba)

    final_pipeline = build_pipeline()
    X_train_full = pd.concat([X_train, X_val], axis=0)
    y_train_full = pd.concat([y_train, y_val], axis=0)
    final_pipeline.fit(X_train_full, y_train_full)

    test_proba = final_pipeline.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= threshold).astype(int)

    metrics = {
        "test_roc_auc": float(roc_auc_score(y_test, test_proba)),
        "test_pr_auc": float(average_precision_score(y_test, test_proba)),
        "test_precision": float(precision_score(y_test, test_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, test_pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, test_pred, zero_division=0)),
        "test_confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
        "test_samples": int(len(y_test)),
        "test_fraud_samples": int(y_test.sum()),
    }

    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "threshold": float(threshold),
        "threshold_selection": {
            "method": "max_f1_on_validation",
            "val_precision": float(val_precision),
            "val_recall": float(val_recall),
            "val_f1": float(val_f1),
        },
        "metrics": metrics,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, model_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Saved model to:", model_path)
    print("Saved metadata to:", metadata_path)
    print("Threshold:", metadata["threshold"])
    print("Test PR-AUC:", metrics["test_pr_auc"])
    print("Test Recall:", metrics["test_recall"])


def parse_args() -> argparse.Namespace:
    default_data = project_root() / "data" / "creditcard.csv"
    default_model = project_root() / "models" / "fraud_pipeline.joblib"
    default_meta = project_root() / "models" / "fraud_metadata.json"

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=default_data)
    parser.add_argument("--model-out", type=Path, default=default_model)
    parser.add_argument("--meta-out", type=Path, default=default_meta)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_and_save(args.data, args.model_out, args.meta_out)


if __name__ == "__main__":
    main()
