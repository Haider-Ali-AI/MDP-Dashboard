"""
=============================================================================
api.py  –  FastAPI Telemetry Ingestion Endpoint
=============================================================================
Provides a lightweight HTTP API that CI/CD pipelines, pre-commit hooks,
and developer tooling can POST code-module static metrics to.  Every
incoming record is:
  1. Validated by Pydantic.
  2. Scored by the production model (models/defect_model.pkl).
  3. Persisted into the SQLite engineering_telemetry table.
  4. Returned with an immediate defect risk assessment.

Running Locally
---------------
    uvicorn src.api:app --reload --port 8000

Example cURL
------------
    curl -X POST http://localhost:8000/log-telemetry/ \\
      -H "Content-Type: application/json" \\
      -d '{
        "loc_blank":0,"branch_count":5,"loc_code_and_comment":0,
        "loc_comments":2,"cyclomatic_complexity":4,"design_complexity":3,
        "essential_complexity":3,"loc_executable":25,"halstead_content":18.5,
        "halstead_difficulty":8.5,"halstead_effort":3250.0,
        "halstead_error_est":0.09,"halstead_length":55,"halstead_level":0.12,
        "halstead_prog_time":180.0,"halstead_volume":210.0,
        "num_operands":18,"num_operators":37,"num_unique_operands":12,
        "num_unique_operators":10,"loc_total":28,"defective":0,
        "source":"ci_hook"
      }'
=============================================================================
"""

import os
import sys
import logging
from typing import Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# ── Path bootstrap ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.database import init_db, seed_from_file, log_entry, get_stats, FEATURE_COLS

# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH    = "models/defect_model.pkl"
ARFF_SEED_PATH = "data/KC1.arff"

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MDP Defect Prediction API",
    description=(
        "Real-time software module telemetry ingestion and defect risk scoring. "
        "POST static code metrics from your CI/CD pipeline to get instant predictions."
    ),
    version="1.0.0",
)

# Allow cross-origin requests (useful for dashboard iframes or external tooling).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Startup: initialise DB + seed + load model
# ─────────────────────────────────────────────────────────────────────────────
_model_artefact = None   # Module-level cache – loaded once at startup.


@app.on_event("startup")
def _startup():
    global _model_artefact
    logger.info("API startup: initialising database …")
    init_db()
    if os.path.isfile(ARFF_SEED_PATH):
        seed_from_file(ARFF_SEED_PATH)

    if os.path.isfile(MODEL_PATH):
        _model_artefact = joblib.load(MODEL_PATH)
        logger.info("Model loaded from %s", MODEL_PATH)
    else:
        logger.warning(
            "Model not found at %s – /log-telemetry/ will store data "
            "but cannot score it until the model is trained.", MODEL_PATH
        )


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Input Schema
# ─────────────────────────────────────────────────────────────────────────────
class TelemetryRecord(BaseModel):
    """All 21 McCabe + Halstead feature columns, plus optional metadata."""
    # Feature columns
    loc_blank:               float = Field(0.0,   ge=0, description="Blank lines of code")
    branch_count:            float = Field(1.0,   ge=0, description="Branch count")
    loc_code_and_comment:    float = Field(0.0,   ge=0)
    loc_comments:            float = Field(0.0,   ge=0)
    cyclomatic_complexity:   float = Field(1.0,   ge=0, description="McCabe v(g)")
    design_complexity:       float = Field(1.0,   ge=0)
    essential_complexity:    float = Field(1.0,   ge=0)
    loc_executable:          float = Field(1.0,   ge=0)
    halstead_content:        float = Field(0.0,   ge=0)
    halstead_difficulty:     float = Field(0.0,   ge=0)
    halstead_effort:         float = Field(0.0,   ge=0)
    halstead_error_est:      float = Field(0.0,   ge=0)
    halstead_length:         float = Field(0.0,   ge=0)
    halstead_level:          float = Field(0.0,   ge=0)
    halstead_prog_time:      float = Field(0.0,   ge=0)
    halstead_volume:         float = Field(0.0,   ge=0)
    num_operands:            float = Field(0.0,   ge=0)
    num_operators:           float = Field(0.0,   ge=0)
    num_unique_operands:     float = Field(0.0,   ge=0)
    num_unique_operators:    float = Field(0.0,   ge=0)
    loc_total:               float = Field(1.0,   ge=0)
    # Metadata
    defective:               int   = Field(0, ge=0, le=1,
                                           description="Ground truth label (0=clean, 1=defective)")
    source:                  str   = Field("api",
                                           description="Origin tag: api | ci_hook | manual | historical")

    @field_validator("source")
    @classmethod
    def _valid_source(cls, v):
        allowed = {"api", "ci_hook", "manual", "historical"}
        if v not in allowed:
            raise ValueError(f"source must be one of {allowed}")
        return v


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _score(record: TelemetryRecord) -> Optional[float]:
    """Return model defect probability, or None if model not loaded."""
    if _model_artefact is None:
        return None
    feature_names = _model_artefact.get("feature_names", FEATURE_COLS)
    vec = np.array(
        [[getattr(record, f, 0.0) for f in feature_names]],
        dtype=np.float32,
    )
    proba = float(_model_artefact["model"].predict_proba(vec)[0][1])
    return round(proba, 4)


def _risk_label(proba: Optional[float], threshold: float = 0.07) -> str:
    if proba is None:
        return "UNKNOWN"
    return "HIGH" if proba >= threshold else "LOW"


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/log-telemetry/", summary="Ingest a module telemetry record")
def post_telemetry(record: TelemetryRecord):
    """
    Accept code module metrics from CI/CD pipelines or developer tools,
    persist them to the SQLite telemetry database, and return a real-time
    defect risk assessment.
    """
    try:
        proba    = _score(record)
        threshold = (_model_artefact or {}).get("threshold", 0.07)
        rowid    = log_entry(
            record.model_dump(exclude={"source"}),
            predicted_risk=proba,
            source=record.source,
        )
        return {
            "status":           "logged",
            "rowid":            rowid,
            "defect_probability": proba,
            "risk_level":       _risk_label(proba, threshold),
            "threshold":        threshold,
            "message": (
                "⚠️ HIGH RISK – refactoring recommended before deployment."
                if _risk_label(proba, threshold) == "HIGH"
                else "✅ LOW RISK – module passes the quality gate."
            ),
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected error in /log-telemetry/")
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")


@app.get("/db-stats/", summary="Live telemetry database statistics")
def get_db_stats():
    """Return summary statistics for the engineering_telemetry table."""
    try:
        return get_stats()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health/", summary="Health check")
def health():
    return {
        "status":       "ok",
        "model_loaded": _model_artefact is not None,
    }
