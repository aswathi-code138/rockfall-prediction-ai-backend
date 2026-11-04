"""FastAPI service for real-time rockfall-risk scoring."""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .predict import load_model, score_records
from .utils import ROOT, configure_logging
from database import create_tables, get_connection

LOGGER = logging.getLogger("rockfall_ai.api")
MODEL_PATH = ROOT / "models" / "best_model.pkl"


class PredictionRequest(BaseModel):
    """A batch of site/environment observations to score."""

    records: list[dict[str, Any]] = Field(min_length=1, max_length=1_000, description="Observation records keyed by feature name.")
    alert_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Probability at which an alert is raised.")


class PredictionItem(BaseModel):
    """Prediction and alert decision for one input observation."""

    rockfall_probability: float
    rockfall_risk: int
    alert: bool


class PredictionResponse(BaseModel):
    """Scoring response payload."""

    model_loaded: bool
    alert_threshold: float
    predictions: list[PredictionItem]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup while allowing health diagnostics if absent."""
    configure_logging()
    create_tables()
    try:
        app.state.model = load_model(MODEL_PATH)
        LOGGER.info("Loaded model from %s", MODEL_PATH)
    except FileNotFoundError:
        app.state.model = None
        LOGGER.warning("Model artifact not found at %s", MODEL_PATH)
    yield


app = FastAPI(
    title="Rockfall Prediction API",
    version="1.0.0",
    description="Real-time scoring service for the SIH AI-Based Rockfall Prediction and Alert System.",
    lifespan=lifespan,
)

origins = [origin.strip() for origin in os.getenv("ROCKFALL_CORS_ORIGINS", "*").split(",") if origin.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["*"])


@app.get("/health", tags=["Operations"])
def health(request: Request) -> dict[str, Any]:
    """Return server and saved-model readiness."""
    model_loaded = getattr(request.app.state, "model", None) is not None
    return {"status": "ok" if model_loaded else "degraded", "model_loaded": model_loaded, "model_path": str(MODEL_PATH)}


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    """Score one or more records and flag results above the alert threshold."""
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model artifact is unavailable. Run training before serving predictions.")
    try:
        results = score_records(payload.records, model)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid prediction records: {exc}") from exc
    items = [PredictionItem(rockfall_probability=float(row.rockfall_probability), rockfall_risk=int(row.rockfall_risk), alert=bool(row.rockfall_probability >= payload.alert_threshold)) for row in results.itertuples(index=False)]
    return PredictionResponse(model_loaded=True, alert_threshold=payload.alert_threshold, predictions=items)
