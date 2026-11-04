"""Dataset discovery, standardization, cleaning, and safe merging."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("rockfall_ai")
KEYS = ("date", "latitude", "longitude", "mine_id")


def standardize_name(name: object) -> str:
    """Convert an external column header into a predictable snake_case name."""
    value = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower())
    return re.sub(r"_+", "_", value).strip("_")


def read_table(path: Path) -> pd.DataFrame:
    """Read a CSV or XLSX, including Open-Meteo's two-header CSV format."""
    if path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    else:
        try:
            frame = pd.read_csv(path, low_memory=False)
        except pd.errors.ParserError:
            lines = path.read_text(encoding="utf-8-sig").splitlines()
            header = next((i for i, line in enumerate(lines) if line.lower().startswith("time,")), None)
            if header is None:
                raise
            frame = pd.read_csv(path, skiprows=header, low_memory=False)
    frame.columns = [standardize_name(c) for c in frame.columns]
    frame = frame.loc[:, ~frame.columns.str.match(r"^unnamed")]
    return frame


def classify_dataset(frame: pd.DataFrame) -> str:
    """Infer dataset role from meaningful columns rather than filenames."""
    columns = set(frame.columns)
    if {"event_date", "landslide_category"} & columns or "rockfall_risk" in columns:
        return "events"
    if "time" in columns or {"rain", "temperature_2m_c"} & columns:
        return "weather"
    if {"bulk_density", "depth", "resistivity"} & columns:
        return "geology"
    return "other"


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove empty/duplicate records, coerce dates, and replace invalid infinities."""
    data = frame.dropna(axis=0, how="all").drop_duplicates().copy()
    for col in data.columns:
        if col in {"date", "event_date", "time", "created_date", "submitted_date"} or col.endswith("_date"):
            data[col] = pd.to_datetime(data[col], errors="coerce")
    numeric = data.select_dtypes(include=np.number).columns
    data[numeric] = data[numeric].replace([np.inf, -np.inf], np.nan)
    for col in ("latitude", "longitude"):
        if col in data:
            data.loc[~data[col].between(-90 if col == "latitude" else -180, 90 if col == "latitude" else 180), col] = np.nan
    return data


def discover_datasets(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load supported files recursively; missing dataset roles remain empty."""
    loaded: dict[str, pd.DataFrame] = {}
    # The configured data directory may be a project parent. Never feed generated
    # artifacts back into training on a subsequent run.
    ignored_directories = {"models", "outputs", "reports", "notebooks", ".venv", "__pycache__"}
    ignored_prefixes = ("prediction", "model_comparison", "missing_value_report", "statistical_summary")
    for path in Path(data_dir).rglob("*"):
        if path.suffix.lower() not in {".csv", ".xlsx", ".xls"}:
            continue
        if any(part.lower() in ignored_directories for part in path.parts) or path.name.lower().startswith(ignored_prefixes):
            LOGGER.debug("Ignoring generated/non-input file: %s", path)
            continue
        try:
            frame = clean_frame(read_table(path))
            role = classify_dataset(frame)
            if role == "other":
                LOGGER.warning("Skipping unrecognized data file: %s", path.name)
                continue
            if role in loaded:
                loaded[role] = pd.concat([loaded[role], frame], ignore_index=True)
            else:
                loaded[role] = frame
            LOGGER.info("Loaded %s as %s (%d rows)", path.name, role, len(frame))
        except Exception as exc:
            LOGGER.warning("Could not read %s: %s", path.name, exc)
    return loaded


def _target_column(frame: pd.DataFrame) -> str | None:
    for column in frame.columns:
        if standardize_name(column) in {"rockfall_risk", "rockfall", "target", "label", "risk"}:
            return column
    return None


def prepare_events(events: pd.DataFrame) -> pd.DataFrame:
    """Create the binary target from an existing label or NASA category."""
    data = events.copy()
    target = _target_column(data)
    if target is not None:
        values = data[target].astype(str).str.strip().str.lower()
        data["rockfall_risk"] = values.isin({"1", "true", "yes", "rockfall", "rock_fall", "high"}).astype(int)
    elif "landslide_category" in data:
        data["rockfall_risk"] = data["landslide_category"].astype(str).str.lower().eq("rock_fall").astype(int)
    else:
        raise ValueError("No target found. Provide Rockfall_Risk or landslide_category.")
    if "event_date" in data:
        data = data.rename(columns={"event_date": "date"})
    return data


def merge_datasets(datasets: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list[str]]:
    """Merge records using aligned keys; use global geology summaries only without keys."""
    if "events" not in datasets:
        raise FileNotFoundError("No event/target dataset was discovered.")
    events = prepare_events(datasets["events"])
    assumptions: list[str] = []
    weather = datasets.get("weather")
    if weather is not None:
        weather = weather.rename(columns={"time": "date"})
        weather["date"] = pd.to_datetime(weather["date"], errors="coerce").dt.normalize()
        events["date"] = pd.to_datetime(events["date"], errors="coerce").dt.normalize()
        shared = [key for key in KEYS if key in events and key in weather]
        if shared and events[shared].dropna().merge(weather[shared].dropna().drop_duplicates(), on=shared).shape[0]:
            events = events.merge(weather.drop_duplicates(shared), on=shared, how="left", suffixes=("", "_weather"))
        else:
            assumptions.append("Weather was not merged: no overlapping date/location or Mine ID keys were present.")
    geology = datasets.get("geology")
    if geology is not None:
        shared = [key for key in ("mine_id", "latitude", "longitude") if key in events and key in geology]
        if shared:
            events = events.merge(geology.drop_duplicates(shared), on=shared, how="left", suffixes=("", "_geology"))
        else:
            numerical = geology.select_dtypes(include=np.number)
            for column in numerical.columns:
                events[f"geology_{column}_mean"] = numerical[column].mean()
                events[f"geology_{column}_std"] = numerical[column].std()
            assumptions.append("Geology had no spatial/Mine ID keys; profile mean and standard-deviation summaries were appended as site-level placeholders.")
    return events, assumptions
