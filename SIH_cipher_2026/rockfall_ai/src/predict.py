"""Inference entry point for trained rockfall risk models."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from .feature_engineering import LEAKAGE_COLUMNS
from .preprocessing import clean_frame, read_table
from .utils import ROOT


def load_model(model_path: str | Path | None = None) -> Any:
    """Load the fitted model artifact used for inference."""
    path = Path(model_path) if model_path else ROOT / "models" / "best_model.pkl"
    with path.open("rb") as handle:
        return pickle.load(handle)


def prepare_inference_data(data: pd.DataFrame, model: Any) -> pd.DataFrame:
    """Apply the same lightweight feature transformations used by training."""
    data = clean_frame(data.copy())
    if "event_date" in data:
        data = data.rename(columns={"event_date": "date"})
    if "date" in data:
        date = pd.to_datetime(data["date"], errors="coerce")
        data["event_year"] = date.dt.year
        data["event_month"] = date.dt.month
        data["event_dayofyear"] = date.dt.dayofyear
        data = data.drop(columns="date")
    if {"latitude", "longitude"}.issubset(data.columns):
        data["latitude_longitude_interaction"] = data["latitude"] * data["longitude"]
    data = data.drop(columns=[c for c in data if c in LEAKAGE_COLUMNS], errors="ignore")
    # Absent fields become missing; the persisted preprocessing pipeline imputes them.
    return data.reindex(columns=model.feature_names_in_)


def score_records(records: list[dict[str, Any]], model: Any | None = None) -> pd.DataFrame:
    """Return binary model scores for one or more JSON-compatible records."""
    fitted_model = model or load_model()
    data = prepare_inference_data(pd.DataFrame(records), fitted_model)
    probability = fitted_model.predict_proba(data)[:, 1]
    return pd.DataFrame({"rockfall_probability": probability, "rockfall_risk": (probability >= 0.5).astype(int)})


def predict(input_path: str, output_path: str | None = None) -> pd.DataFrame:
    """Score CSV/XLSX records and save risk labels and probabilities."""
    data = read_table(Path(input_path))
    result = score_records(data.to_dict(orient="records"))
    result.to_csv(output_path or ROOT / "outputs" / "predictions.csv", index=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("input"); parser.add_argument("--output", default=None)
    args = parser.parse_args(); print(predict(args.input, args.output).head())
