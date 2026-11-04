"""Shared utilities for configuration, logging, and project tracking."""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def configure_logging(level: str = "INFO") -> logging.Logger:
    """Configure application logging once and return the package logger."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger("rockfall_ai")


def ensure_directories(config: dict[str, Any]) -> None:
    """Create runtime output directories."""
    for key in ("models_dir", "outputs_dir", "plots_dir", "reports_dir"):
        Path(config[key]).mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load JSON configuration, resolving relative paths from project root."""
    config_path = Path(path) if path else ROOT / "config.json"
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    for key in ("data_dir", "models_dir", "outputs_dir", "plots_dir", "reports_dir"):
        configured = Path(config[key])
        config[key] = str((ROOT / configured).resolve()) if not configured.is_absolute() else str(configured)
    return config


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write serializable output safely with consistent formatting."""
    Path(path).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def update_work_item(label: str, value: str | None = None) -> None:
    """Mark a WORK.md checkbox complete and optionally update the best-model block."""
    work_path = ROOT / "WORK.md"
    if not work_path.exists():
        return
    text = work_path.read_text(encoding="utf-8")
    text = text.replace(f"- [ ] {label}", f"- [x] {label}")
    if value is not None:
        text = re.sub(rf"(?m)^{re.escape(label)}:.*$", f"{label}: {value}", text)
    text = re.sub(r"Last Updated:.*", f"Last Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", text)
    work_path.write_text(text, encoding="utf-8")
