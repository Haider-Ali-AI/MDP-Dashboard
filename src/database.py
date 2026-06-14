"""
=============================================================================
database.py  –  SQLite Telemetry Layer for NASA MDP Defect Prediction
=============================================================================
Provides a persistent engineering_telemetry table that captures:
  • Historical records seeded from the original NASA ARFF file (first run).
  • New real-time records submitted via the FastAPI endpoint or the
    Streamlit manual log-entry form.

Schema mirrors all 21 McCabe + Halstead feature columns plus metadata:
  id              – Auto-increment primary key
  source          – 'historical' | 'api' | 'manual' | 'ci_hook'
  predicted_risk  – Model probability at time of logging (NULL for historical)
  logged_at       – UTC ISO-8601 timestamp
  [21 feature cols] + defective (0/1)

Thread Safety
-------------
Each public function creates its own connection so Streamlit's multi-
threaded execution does not share a single connection object.
=============================================================================
"""

import os
import sqlite3
import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Paths & Schema
# ─────────────────────────────────────────────────────────────────────────────
DB_PATH: str = os.path.join("data", "telemetry.db")
TABLE:   str = "engineering_telemetry"

# The 21 feature columns in the exact order the KC1 ARFF defines them.
FEATURE_COLS = [
    "loc_blank", "branch_count", "loc_code_and_comment", "loc_comments",
    "cyclomatic_complexity", "design_complexity", "essential_complexity",
    "loc_executable", "halstead_content", "halstead_difficulty",
    "halstead_effort", "halstead_error_est", "halstead_length",
    "halstead_level", "halstead_prog_time", "halstead_volume",
    "num_operands", "num_operators", "num_unique_operands",
    "num_unique_operators", "loc_total",
]

ALL_COLS = FEATURE_COLS + ["defective"]   # 22 domain columns

# DDL for the telemetry table.
_CREATE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source          TEXT    NOT NULL DEFAULT 'historical',
    predicted_risk  REAL,
    logged_at       TEXT    NOT NULL,
    {", ".join(f"{c}  REAL" for c in FEATURE_COLS)},
    defective       INTEGER NOT NULL DEFAULT 0
);
"""

_INDEX_SQL = f"""
CREATE INDEX IF NOT EXISTS idx_logged_at ON {TABLE}(logged_at);
"""


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _connect() -> sqlite3.Connection:
    """Return a new SQLite connection with row_factory set to dict-like rows."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _is_seeded(conn: sqlite3.Connection) -> bool:
    """Return True if the table already contains at least one historical row."""
    cur = conn.execute(
        f"SELECT COUNT(*) as n FROM {TABLE} WHERE source = 'historical'"
    )
    return cur.fetchone()["n"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """
    Create the SQLite database file and the engineering_telemetry table
    if they do not already exist.  Safe to call on every app startup.
    """
    conn = _connect()
    try:
        conn.execute(_CREATE_SQL)
        conn.execute(_INDEX_SQL)
        conn.commit()
        logger.info("Database initialised at: %s", DB_PATH)
    finally:
        conn.close()


def seed_from_file(arff_or_csv_path: str, force: bool = False) -> int:
    """
    Parse the NASA MDP ARFF / CSV and bulk-insert all rows as
    source='historical'.  Skips seeding if data already present unless
    force=True.

    Parameters
    ----------
    arff_or_csv_path : str
        Path to KC1.arff (or any supported MDP file).
    force : bool
        If True, re-seed even if rows already exist.

    Returns
    -------
    int – Number of rows inserted (0 if skipped).
    """
    conn = _connect()
    try:
        if not force and _is_seeded(conn):
            logger.info("DB already seeded – skipping (pass force=True to re-seed).")
            return 0

        # Re-use the ARFF parser from data_pipeline.
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.data_pipeline import run_pipeline

        _, _, _, _, _, df_full = run_pipeline(arff_or_csv_path)

        # Keep only the columns we care about; fill missing with 0.
        keep = [c for c in ALL_COLS if c in df_full.columns]
        df_seed = df_full[keep].copy()

        # Add metadata columns.
        now = datetime.now(timezone.utc).isoformat()
        df_seed["source"]         = "historical"
        df_seed["predicted_risk"] = None
        df_seed["logged_at"]      = now

        # Write to DB.
        df_seed.to_sql(TABLE, conn, if_exists="append", index=False)
        conn.commit()
        n = len(df_seed)
        logger.info("Seeded %d historical records into %s.", n, TABLE)
        return n

    except Exception as exc:
        logger.error("Seeding failed: %s", exc)
        conn.rollback()
        raise
    finally:
        conn.close()


def log_entry(
    record: dict,
    predicted_risk: Optional[float] = None,
    source: str = "api",
) -> int:
    """
    Insert one telemetry record into the database.

    Parameters
    ----------
    record         : dict  – Must contain all 22 domain columns.
    predicted_risk : float – Model defect probability (0.0 – 1.0).
    source         : str   – One of 'api', 'manual', 'ci_hook', 'historical'.

    Returns
    -------
    int – The rowid of the newly inserted row.
    """
    # Validate that all feature columns are present.
    missing = [c for c in ALL_COLS if c not in record]
    if missing:
        raise ValueError(f"Missing columns in record: {missing}")

    now = datetime.now(timezone.utc).isoformat()
    row = {
        "source":         source,
        "predicted_risk": float(predicted_risk) if predicted_risk is not None else None,
        "logged_at":      now,
        **{c: float(record[c]) for c in FEATURE_COLS},
        "defective":      int(record.get("defective", 0)),
    }

    cols    = ", ".join(row.keys())
    placeholders = ", ".join("?" * len(row))
    sql     = f"INSERT INTO {TABLE} ({cols}) VALUES ({placeholders})"

    conn = _connect()
    try:
        cur  = conn.execute(sql, list(row.values()))
        conn.commit()
        logger.info("Logged entry rowid=%d  risk=%.3f", cur.lastrowid,
                    predicted_risk or 0.0)
        return cur.lastrowid
    finally:
        conn.close()


def get_dataframe(source_filter: Optional[str] = None) -> pd.DataFrame:
    """
    Retrieve all (or filtered) records from the telemetry table.

    Parameters
    ----------
    source_filter : str | None
        If provided, only return rows where source = source_filter.

    Returns
    -------
    pd.DataFrame
    """
    conn = _connect()
    try:
        if source_filter:
            df = pd.read_sql_query(
                f"SELECT * FROM {TABLE} WHERE source = ?",
                conn, params=(source_filter,),
            )
        else:
            df = pd.read_sql_query(f"SELECT * FROM {TABLE}", conn)
        return df
    finally:
        conn.close()


def execute_query(sql: str) -> pd.DataFrame:
    """
    Execute a **read-only** SQL SELECT statement and return results as a
    DataFrame.  Raises ValueError on any non-SELECT statement to prevent
    accidental mutations from the LLM agent.

    Parameters
    ----------
    sql : str  – A SELECT statement.

    Returns
    -------
    pd.DataFrame
    """
    normalised = sql.strip().upper()
    if not normalised.startswith("SELECT"):
        raise ValueError(
            "Only SELECT statements are permitted via execute_query(). "
            f"Received: {sql[:80]}"
        )

    conn = _connect()
    try:
        df = pd.read_sql_query(sql, conn)
        return df
    finally:
        conn.close()


def get_stats() -> dict:
    """
    Return a quick summary dict for the API /db-stats/ endpoint.
    """
    conn = _connect()
    try:
        row = conn.execute(
            f"""SELECT
                COUNT(*)                           AS total_rows,
                SUM(defective)                     AS defective_count,
                ROUND(AVG(defective) * 100, 2)    AS defect_rate_pct,
                MAX(logged_at)                     AS last_logged_at
            FROM {TABLE}"""
        ).fetchone()
        return dict(row)
    finally:
        conn.close()
