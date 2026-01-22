from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COLUMN = "Class"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_model_path() -> Path:
    return project_root() / "models" / "fraud_pipeline.joblib"


def default_metadata_path() -> Path:
    return project_root() / "models" / "fraud_metadata.json"


def load_model(model_path: str | Path | None = None) -> Any:
    path = Path(model_path) if model_path is not None else default_model_path()
    try:
        return joblib.load(path)
    except FileNotFoundError:
        raise
    except Exception as e:
        if isinstance(e, (AttributeError, ModuleNotFoundError, ImportError, TypeError, ValueError)):
            try:
                import sklearn

                skl_version = sklearn.__version__
            except Exception:
                skl_version = "unknown"

            raise RuntimeError(
                "Failed to load the saved model artifact. This is usually caused by a scikit-learn version mismatch "
                f"between training and inference (installed scikit-learn: {skl_version}). "
                "Recreate the artifact by running: python src/train.py"
            ) from e
        raise


def load_metadata(metadata_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(metadata_path) if metadata_path is not None else default_metadata_path()
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df[FEATURE_COLUMNS]


def predict_dataframe(model: Any, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    X = prepare_features(df)

    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)

    out = df.copy()
    out["fraud_probability"] = proba
    out["fraud_prediction"] = pred

    return out


def summarize_predictions(pred_df: pd.DataFrame) -> dict[str, float]:
    if "fraud_probability" not in pred_df.columns or "fraud_prediction" not in pred_df.columns:
        raise ValueError("Prediction dataframe must contain fraud_probability and fraud_prediction")

    total = float(len(pred_df))
    if total == 0:
        return {"total": 0.0, "predicted_fraud": 0.0, "predicted_fraud_rate": 0.0}

    predicted_fraud = float(np.asarray(pred_df["fraud_prediction"]).sum())
    return {
        "total": total,
        "predicted_fraud": predicted_fraud,
        "predicted_fraud_rate": predicted_fraud / total,
    }
