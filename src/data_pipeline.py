"""
=============================================================================
data_pipeline.py  –  NASA MDP Software Defect Prediction
=============================================================================
Responsibilities
----------------
1. Load the raw NASA MDP dataset – supports both ARFF (Weka) and CSV formats.
   ARFF parser is built-in; no extra dependency needed.
2. Normalise column names to lower-case for consistency across all MDP files.
3. Clean the data: replace missing / null / infinite values in every numeric
   column with the column median (robust to outliers).
4. Encode the binary target variable:
     ARFF format  → 'Y' / 'N'   →  1 / 0
     CSV format   → 'true'/'false' / 'yes'/'no'  →  1 / 0
5. Perform a stratified 80/20 train-test split that preserves the class ratio.
6. Return the four split arrays plus the feature-name list so that
   train_model.py can consume them directly.

Actual KC1.arff Column Layout (after lower-casing)
---------------------------------------------------
  loc_blank, branch_count, loc_code_and_comment, loc_comments,
  cyclomatic_complexity, design_complexity, essential_complexity,
  loc_executable, halstead_content, halstead_difficulty, halstead_effort,
  halstead_error_est, halstead_length, halstead_level, halstead_prog_time,
  halstead_volume, num_operands, num_operators, num_unique_operands,
  num_unique_operators, loc_total,
  defective  ← TARGET  ('Y' / 'N')
=============================================================================
"""

import os
import sys
import re
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# The target column name AFTER lower-casing (covers both 'Defective' and 'defects').
TARGET_COLUMN: str = "defective"     # KC1/KC3/JM1 etc. use 'Defective'
FALLBACK_TARGET: str = "defects"     # Some CSV variants use 'defects'

# All string values that indicate a DEFECTIVE module (after lower-casing).
POSITIVE_LABELS: set = {"y", "yes", "true", "1"}

# Columns that are meta-data, not features (dropped before training).
NON_FEATURE_COLUMNS: list = [
    "defective", "defects", "module", "name", "filename", "version"
]

# Random seed for reproducibility.
RANDOM_STATE: int = 42

# Train / test split fraction.
TEST_SIZE: float = 0.20


# ---------------------------------------------------------------------------
# Helper: Parse ARFF File → pd.DataFrame
# ---------------------------------------------------------------------------
def _parse_arff(arff_path: str) -> pd.DataFrame:
    """
    Lightweight ARFF parser that does not require scipy or weka.

    Reads the @attribute header to discover column names and types, then
    reads the @data section and constructs a pandas DataFrame.

    Parameters
    ----------
    arff_path : str

    Returns
    -------
    pd.DataFrame
    """
    attribute_names = []
    attribute_types = []   # 'numeric' or 'nominal'
    data_rows = []
    in_data = False

    with open(arff_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("%"):
                continue                           # Skip comments / blank lines.

            if line.lower().startswith("@attribute"):
                # Pattern: @attribute <name> <type or {values}>
                parts = re.split(r"\s+", line, maxsplit=2)
                col_name = parts[1].strip("'\"")
                col_type_raw = parts[2].strip() if len(parts) > 2 else "numeric"
                if col_type_raw.startswith("{"):
                    attribute_types.append("nominal")
                else:
                    attribute_types.append("numeric")
                attribute_names.append(col_name)

            elif line.lower().startswith("@data"):
                in_data = True

            elif in_data:
                if not line or line.startswith("%"):
                    continue
                values = line.split(",")
                row = []
                for v, t in zip(values, attribute_types):
                    v = v.strip()
                    if t == "numeric":
                        try:
                            row.append(float(v) if v not in ("?", "") else np.nan)
                        except ValueError:
                            row.append(np.nan)
                    else:
                        row.append(v if v not in ("?", "") else np.nan)
                data_rows.append(row)

    df = pd.DataFrame(data_rows, columns=attribute_names)
    return df


# ---------------------------------------------------------------------------
# Helper: Load Dataset (ARFF or CSV)
# ---------------------------------------------------------------------------
def load_dataset(data_path: str) -> pd.DataFrame:
    """
    Load the NASA MDP dataset from *data_path*.

    Supports:
    • .arff  – parsed with the built-in lightweight parser.
    • .csv   – loaded with pandas read_csv.

    Parameters
    ----------
    data_path : str
        Absolute or relative path to the dataset file (.arff or .csv).

    Returns
    -------
    pd.DataFrame
        Raw DataFrame with lower-cased column names.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file extension is not supported.
    """
    if not os.path.isfile(data_path):
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'. "
            "Ensure the NASA MDP ARFF or CSV file exists at the given path."
        )

    ext = os.path.splitext(data_path)[1].lower()
    logger.info("Loading dataset from: %s  (format: %s)", data_path, ext)

    if ext == ".arff":
        df = _parse_arff(data_path)
    elif ext == ".csv":
        df = pd.read_csv(data_path)
    else:
        raise ValueError(
            f"Unsupported file extension '{ext}'. Expected .arff or .csv."
        )

    # Normalise: strip whitespace + lower-case all column names.
    df.columns = [c.strip().lower() for c in df.columns]

    logger.info("Loaded %d rows × %d columns.", df.shape[0], df.shape[1])
    logger.info("Columns: %s", list(df.columns))
    return df


# ---------------------------------------------------------------------------
# Helper: Resolve Target Column
# ---------------------------------------------------------------------------
def _resolve_target(df: pd.DataFrame) -> str:
    """
    Automatically find the target column name in the DataFrame.
    Tries TARGET_COLUMN first, then FALLBACK_TARGET.

    Raises KeyError if neither is found.
    """
    if TARGET_COLUMN in df.columns:
        return TARGET_COLUMN
    if FALLBACK_TARGET in df.columns:
        logger.info(
            "Target column '%s' not found; using fallback '%s'.",
            TARGET_COLUMN, FALLBACK_TARGET,
        )
        return FALLBACK_TARGET
    raise KeyError(
        f"No target column found. Looked for '{TARGET_COLUMN}' and "
        f"'{FALLBACK_TARGET}'. Available columns: {list(df.columns)}"
    )


# ---------------------------------------------------------------------------
# Helper: Clean / Impute Numeric Columns
# ---------------------------------------------------------------------------
def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every numeric column in *df*:
      • Replace ±Inf values with NaN.
      • Replace NaN values with the column's median.

    Median imputation is robust to the extreme Halstead / LOC outliers
    common in NASA MDP datasets.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    logger.info(
        "Imputing %d numeric column(s) with column medians …", len(numeric_cols)
    )

    for col in numeric_cols:
        n_inf = np.isinf(df[col]).sum()
        if n_inf > 0:
            logger.warning("  %s: replaced %d Inf value(s) with NaN.", col, n_inf)
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

        n_nan = df[col].isna().sum()
        if n_nan > 0:
            median_val = df[col].median()
            logger.info(
                "  %s: filled %d NaN(s) with median=%.4f.", col, n_nan, median_val
            )
            df[col] = df[col].fillna(median_val)

    return df


# ---------------------------------------------------------------------------
# Helper: Encode Target Variable
# ---------------------------------------------------------------------------
def encode_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Convert the string defect label into a binary integer:
      Y / yes / true / 1  →  1  (defective)
      N / no  / false / 0 →  0  (clean)

    Works for both ARFF ('Y'/'N') and CSV ('true'/'false') variants.
    """
    raw = df[target_col].copy().astype(str).str.strip().str.lower()
    df[target_col] = raw.apply(lambda x: 1 if x in POSITIVE_LABELS else 0)

    n_defective = int(df[target_col].sum())
    n_clean     = len(df) - n_defective
    defect_rate = 100.0 * n_defective / len(df)
    logger.info(
        "Target encoded → Defective: %d | Clean: %d | Defect Rate: %.2f%%",
        n_defective, n_clean, defect_rate,
    )
    return df


# ---------------------------------------------------------------------------
# Helper: Select Features
# ---------------------------------------------------------------------------
def select_features(df: pd.DataFrame, target_col: str) -> list:
    """
    Return the ordered list of numeric feature column names, excluding the
    target and any known non-feature meta-columns.
    """
    drop_cols = set(NON_FEATURE_COLUMNS)
    features = [
        c for c in df.columns
        if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])
    ]
    logger.info("Selected %d feature(s): %s", len(features), features)
    return features


# ---------------------------------------------------------------------------
# Main Pipeline Function
# ---------------------------------------------------------------------------
def run_pipeline(data_path: str):
    """
    Execute the full preprocessing pipeline.

    Parameters
    ----------
    data_path : str
        Path to the NASA MDP ARFF or CSV file.

    Returns
    -------
    X_train, X_test : np.ndarray
    y_train, y_test : np.ndarray
    feature_names   : list[str]
    df_full         : pd.DataFrame   (cleaned, with binary target)
    """
    # ── Step 1: Load ──────────────────────────────────────────────────────
    df = load_dataset(data_path)

    # ── Step 2: Resolve target column ────────────────────────────────────
    target_col = _resolve_target(df)

    # ── Step 3: Impute numeric columns ────────────────────────────────────
    df = clean_numeric_columns(df)

    # ── Step 4: Encode target ─────────────────────────────────────────────
    df = encode_target(df, target_col)

    # Normalise target column name to 'defective' for downstream consistency.
    if target_col != "defective":
        df = df.rename(columns={target_col: "defective"})

    # ── Step 5: Select features ───────────────────────────────────────────
    feature_names = select_features(df, "defective")

    if not feature_names:
        raise ValueError(
            "No numeric feature columns found after preprocessing. "
            "Check that the file contains standard NASA MDP metric columns."
        )

    X = df[feature_names].values.astype(np.float32)
    y = df["defective"].values.astype(np.int32)

    # ── Step 6: Stratified 80/20 split ───────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    logger.info(
        "Split → Train: %d rows | Test: %d rows",
        len(X_train), len(X_test),
    )
    logger.info(
        "Train defect rate: %.2f%% | Test defect rate: %.2f%%",
        100.0 * y_train.mean(),
        100.0 * y_test.mean(),
    )

    return X_train, X_test, y_train, y_test, feature_names, df


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data_file = (
        sys.argv[1] if len(sys.argv) > 1
        else "data/NASADefectDataset-master/OriginalData/MDP/KC1.arff"
    )

    X_train, X_test, y_train, y_test, features, df_full = run_pipeline(data_file)

    print("\n── Pipeline complete ──")
    print(f"  Features : {features}")
    print(f"  X_train  : {X_train.shape}")
    print(f"  X_test   : {X_test.shape}")
    print(f"  y_train  : {y_train.shape}  (defective={y_train.sum()})")
    print(f"  y_test   : {y_test.shape}   (defective={y_test.sum()})")
