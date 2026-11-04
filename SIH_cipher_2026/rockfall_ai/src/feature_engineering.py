"""Feature engineering for tabular rockfall-risk modelling."""
from __future__ import annotations

import numpy as np
import pandas as pd


LEAKAGE_COLUMNS = {"rockfall_risk", "landslide_category", "event_id", "source_link", "photo_link", "notes", "event_description", "event_title"}


def build_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build time, geographic, and previous-event features while removing target leakage."""
    data = frame.copy()
    if "date" in data:
        date = pd.to_datetime(data["date"], errors="coerce")
        data["event_year"] = date.dt.year
        data["event_month"] = date.dt.month
        data["event_dayofyear"] = date.dt.dayofyear
        data = data.drop(columns="date")
    if {"latitude", "longitude"}.issubset(data.columns):
        data["latitude_longitude_interaction"] = data["latitude"] * data["longitude"]
    if "fatality_count" in data:
        data["fatality_count"] = pd.to_numeric(data["fatality_count"], errors="coerce").clip(lower=0)
    y = data.pop("rockfall_risk").astype(int)
    drop = [column for column in data.columns if column in LEAKAGE_COLUMNS or column.endswith("_date") or column.endswith("_time")]
    return data.drop(columns=drop, errors="ignore"), y
