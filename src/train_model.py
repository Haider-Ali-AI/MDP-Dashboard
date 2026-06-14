"""
=============================================================================
train_model.py  –  NASA MDP Software Defect Prediction
=============================================================================
Responsibilities
----------------
1. Call the data_pipeline to obtain train / test splits.
2. Apply SMOTE (Synthetic Minority Over-sampling Technique) ONLY to the
   training set, preventing data leakage into the evaluation set.
3. Train a Random Forest ensemble classifier with class-weight awareness.
4. Sweep decision-threshold values to maximise Recall (catching real bugs is
   the top priority in QA; a False Negative is costlier than a False Positive).
5. Report a full classification report and confusion matrix at the chosen
   optimal threshold.
6. Save the champion model artefact to models/defect_model.pkl using joblib.
7. Export feature importances alongside the model so that app.py can visualise
   which code metrics drive predictions.

Typical Usage
-------------
    python src/train_model.py data/KC1.csv

Dependencies
------------
    scikit-learn, imbalanced-learn, joblib, numpy, pandas
    Install: pip install scikit-learn imbalanced-learn joblib numpy pandas
=============================================================================
"""

import os
import sys
import logging
import warnings
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_score,
)

# imbalanced-learn
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    raise ImportError(
        "imbalanced-learn is not installed. "
        "Run: pip install imbalanced-learn"
    )

# Our pipeline module (works whether run from root or src/).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_pipeline import run_pipeline

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Logging
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
MODEL_OUTPUT_PATH: str = "models/defect_model.pkl"
RANDOM_STATE: int = 42

# Threshold sweep: values to test between 0.05 and 0.95.
THRESHOLD_CANDIDATES: np.ndarray = np.arange(0.05, 0.96, 0.01)

# How to balance recall vs F1 during threshold optimisation.
# weight=1.0 → pure recall; weight=0.0 → pure F1.
RECALL_WEIGHT: float = 0.75   # Heavily favour recall, but preserve F1.


# ---------------------------------------------------------------------------
# Helper: Apply SMOTE on Training Data Only
# ---------------------------------------------------------------------------
def apply_smote(X_train: np.ndarray, y_train: np.ndarray, random_state: int = RANDOM_STATE):
    """
    Oversample the minority class (defective modules) in the training set
    using SMOTE so the classifier learns balanced decision boundaries.

    SMOTE is applied ONLY to the training partition to avoid leaking
    synthetic samples into the test evaluation.

    Parameters
    ----------
    X_train      : np.ndarray  –  Training feature matrix.
    y_train      : np.ndarray  –  Training labels {0, 1}.
    random_state : int

    Returns
    -------
    X_resampled, y_resampled : np.ndarray
        Balanced training arrays after oversampling.
    """
    n_minority = int(np.sum(y_train == 1))
    n_majority = int(np.sum(y_train == 0))
    logger.info(
        "Before SMOTE → Majority (clean): %d  |  Minority (defective): %d  "
        "|  Ratio: 1:%.1f",
        n_majority, n_minority, n_majority / max(n_minority, 1)
    )

    # k_neighbors must be < n_minority samples.
    k_neighbors = min(5, n_minority - 1) if n_minority > 1 else 1

    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    logger.info(
        "After  SMOTE → Total samples: %d  |  Defective: %d  |  Clean: %d",
        len(y_res), int(np.sum(y_res == 1)), int(np.sum(y_res == 0))
    )
    return X_res, y_res


# ---------------------------------------------------------------------------
# Helper: Train Random Forest
# ---------------------------------------------------------------------------
def train_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """
    Train a Random Forest classifier on the (SMOTE-balanced) training set.

    Hyper-parameters are chosen for high recall on imbalanced defect datasets:
    • class_weight='balanced_subsample' gives extra weight to defective samples
      inside every bootstrapped tree, providing a second layer of imbalance
      correction on top of SMOTE.
    • n_estimators=300  → rich ensemble, low variance.
    • max_depth=None    → trees grow deep to capture complex interactions.
    • min_samples_leaf=2 → mild regularisation to avoid noise overfitting.

    Parameters
    ----------
    X_train : np.ndarray
    y_train : np.ndarray

    Returns
    -------
    RandomForestClassifier  –  Fitted model instance.
    """
    logger.info("Training Random Forest (300 trees) …")

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",          # Standard square-root feature subsampling.
        class_weight="balanced_subsample",
        bootstrap=True,
        oob_score=True,               # Out-of-bag estimate for quick sanity check.
        n_jobs=-1,                    # Use all CPU cores.
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train, y_train)

    logger.info("OOB accuracy estimate: %.4f", clf.oob_score_)
    return clf


# ---------------------------------------------------------------------------
# Helper: Optimise Decision Threshold
# ---------------------------------------------------------------------------
def optimise_threshold(
    clf: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    recall_weight: float = RECALL_WEIGHT,
) -> float:
    """
    Sweep decision thresholds in [0.05, 0.95] and select the value that
    maximises a weighted combination of Recall and F1-Score.

    objective = recall_weight * Recall + (1 - recall_weight) * F1

    Tuning to maximise Recall ensures that genuinely defective modules are
    almost never classified as clean (False Negatives are the most costly
    error in software QA).

    Parameters
    ----------
    clf          : fitted RandomForestClassifier
    X_test       : np.ndarray
    y_test       : np.ndarray
    recall_weight: float ∈ [0, 1]

    Returns
    -------
    float  –  Optimal threshold value.
    """
    logger.info("Sweeping %d threshold candidates …", len(THRESHOLD_CANDIDATES))
    probas = clf.predict_proba(X_test)[:, 1]   # P(defective)

    best_score = -np.inf
    best_threshold = 0.50

    for thresh in THRESHOLD_CANDIDATES:
        preds = (probas >= thresh).astype(int)

        # Skip degenerate cases where the model predicts only one class.
        if preds.sum() == 0 or preds.sum() == len(preds):
            continue

        rec = recall_score(y_test, preds, zero_division=0)
        f1  = f1_score(y_test, preds, zero_division=0)
        obj = recall_weight * rec + (1 - recall_weight) * f1

        if obj > best_score:
            best_score = obj
            best_threshold = thresh

    logger.info(
        "Optimal threshold: %.2f  (objective score: %.4f)",
        best_threshold, best_score
    )
    return float(best_threshold)


# ---------------------------------------------------------------------------
# Helper: Evaluate and Report
# ---------------------------------------------------------------------------
def evaluate_model(
    clf: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
    feature_names: list,
) -> dict:
    """
    Evaluate the trained model at the chosen threshold and return a metrics
    dictionary that will be bundled into the saved artefact.

    Parameters
    ----------
    clf          : fitted RandomForestClassifier
    X_test       : np.ndarray
    y_test       : np.ndarray
    threshold    : float
    feature_names: list of str

    Returns
    -------
    dict with keys: recall, precision, f1, roc_auc, confusion_matrix,
                    threshold, feature_importances, feature_names.
    """
    probas = clf.predict_proba(X_test)[:, 1]
    preds  = (probas >= threshold).astype(int)

    rec  = recall_score(y_test, preds, zero_division=0)
    pre  = precision_score(y_test, preds, zero_division=0)
    f1   = f1_score(y_test, preds, zero_division=0)
    auc  = roc_auc_score(y_test, probas)
    cm   = confusion_matrix(y_test, preds)

    logger.info("\n%s", "─" * 60)
    logger.info("  MODEL EVALUATION  (threshold = %.2f)", threshold)
    logger.info("─" * 60)
    logger.info("  Recall    (Sensitivity) : %.4f  ← PRIMARY METRIC", rec)
    logger.info("  Precision               : %.4f", pre)
    logger.info("  F1-Score                : %.4f", f1)
    logger.info("  ROC-AUC                 : %.4f", auc)
    logger.info("─" * 60)
    logger.info("\nConfusion Matrix:\n%s", cm)
    logger.info("\nFull Classification Report:\n%s",
                classification_report(y_test, preds, target_names=["Clean", "Defective"],
                                      zero_division=0))

    # Feature importances sorted descending.
    importances = clf.feature_importances_
    sorted_idx  = np.argsort(importances)[::-1]
    sorted_feats = [feature_names[i] for i in sorted_idx]
    sorted_imps  = importances[sorted_idx].tolist()

    metrics = {
        "recall":              float(rec),
        "precision":           float(pre),
        "f1":                  float(f1),
        "roc_auc":             float(auc),
        "confusion_matrix":    cm.tolist(),
        "threshold":           float(threshold),
        "feature_names":       sorted_feats,
        "feature_importances": sorted_imps,
    }
    return metrics


# ---------------------------------------------------------------------------
# Helper: Persist Artefact
# ---------------------------------------------------------------------------
def save_artefact(clf, metrics: dict, feature_names: list, output_path: str = MODEL_OUTPUT_PATH):
    """
    Serialise the trained classifier, evaluation metrics, and feature metadata
    into a single joblib artefact at *output_path*.

    The artefact dictionary structure:
    {
        "model"             : RandomForestClassifier (fitted),
        "threshold"         : float,
        "feature_names"     : list[str],
        "feature_importances": list[float],
        "metrics"           : dict,
    }

    Parameters
    ----------
    clf          : fitted RandomForestClassifier
    metrics      : dict  (output of evaluate_model)
    feature_names: list of str  (original ordered feature list)
    output_path  : str
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    artefact = {
        "model":               clf,
        "threshold":           metrics["threshold"],
        "feature_names":       feature_names,          # Original order (for input form)
        "feature_importances": metrics["feature_importances"],
        "metrics":             metrics,
    }

    joblib.dump(artefact, output_path, compress=3)
    size_kb = os.path.getsize(output_path) / 1024
    logger.info("Artefact saved → %s  (%.1f KB)", output_path, size_kb)




# ─────────────────────────────────────────────────────────────────────────────
# Continuous Learning: retrain_pipeline()
# ─────────────────────────────────────────────────────────────────────────────

def retrain_pipeline(
    db_path:          str = "data/telemetry.db",
    fallback_arff:    str = "data/KC1.arff",
    model_output:     str = MODEL_OUTPUT_PATH,
    min_rows:         int = 100,
) -> dict:
    """
    Pull all records from the live SQLite telemetry database, apply SMOTE,
    run a RandomizedSearchCV hyperparameter search over the Random Forest,
    sweep the optimal decision threshold, and overwrite the production model.

    Falls back to the original ARFF file if the database is empty or
    contains fewer than *min_rows* records.

    Parameters
    ----------
    db_path       : str  – Path to the SQLite database.
    fallback_arff : str  – Original ARFF path used when DB is insufficient.
    model_output  : str  – Where to write the new model artefact.
    min_rows      : int  – Minimum DB rows required before using DB data.

    Returns
    -------
    dict – New evaluation metrics (same structure as evaluate_model()).
    """
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
    import pandas as pd

    logger.info("=" * 60)
    logger.info("  CONTINUOUS LEARNING – RETRAIN PIPELINE")
    logger.info("=" * 60)

    # ── Step 1: Source the training data ──────────────────────────────────
    use_db = False
    if os.path.isfile(db_path):
        try:
            sys.path.insert(
                0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            from src.database import get_dataframe, FEATURE_COLS, ALL_COLS
            df_db = get_dataframe()
            if len(df_db) >= min_rows:
                use_db = True
                logger.info("Using database: %d rows.", len(df_db))
        except Exception as exc:
            logger.warning("Could not read DB (%s). Falling back to ARFF.", exc)

    if use_db:
        df_db = df_db.dropna(subset=["defective"])
        feature_names = [c for c in FEATURE_COLS if c in df_db.columns]
        X = df_db[feature_names].fillna(0).values.astype(np.float32)
        y = df_db["defective"].values.astype(np.int32)
    else:
        logger.info("Falling back to ARFF: %s", fallback_arff)
        X_tr, X_te, y_tr, y_te, feature_names, _ = run_pipeline(fallback_arff)
        X = np.vstack([X_tr, X_te])
        y = np.concatenate([y_tr, y_te])

    # ── Step 2: Stratified split ──────────────────────────────────────────
    from sklearn.model_selection import train_test_split as _tts
    X_train, X_test, y_train, y_test = _tts(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    # ── Step 3: SMOTE on training split only ──────────────────────────────
    X_bal, y_bal = apply_smote(X_train, y_train)

    # ── Step 4: RandomizedSearchCV ────────────────────────────────────────
    param_dist = {
        "n_estimators":    [100, 200, 300, 400],
        "max_depth":       [None, 10, 20, 30],
        "min_samples_leaf":[1, 2, 4],
        "max_features":    ["sqrt", "log2"],
        "class_weight":    ["balanced", "balanced_subsample"],
    }
    base_clf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    search = RandomizedSearchCV(
        base_clf,
        param_distributions=param_dist,
        n_iter=15,                          # 15 random combos – fast enough for interactive use
        scoring="recall",                   # Optimise primary metric
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    logger.info("Running RandomizedSearchCV (15 iterations, 3-fold CV) …")
    search.fit(X_bal, y_bal)
    best_params = search.best_params_
    logger.info("Best params: %s", best_params)

    # Refit on full balanced training set with best params.
    clf = RandomForestClassifier(
        **best_params,
        oob_score=True,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    clf.fit(X_bal, y_bal)
    logger.info("OOB accuracy: %.4f", clf.oob_score_)

    # ── Step 5: Threshold sweep ───────────────────────────────────────────
    best_threshold = optimise_threshold(clf, X_test, y_test)

    # ── Step 6: Evaluate ─────────────────────────────────────────────────
    metrics = evaluate_model(clf, X_test, y_test, best_threshold, feature_names)
    metrics["best_params"] = best_params   # Store tuned params in artefact.

    # ── Step 7: Save ─────────────────────────────────────────────────────
    save_artefact(clf, metrics, feature_names, model_output)
    logger.info("Continuous learning cycle complete.")
    return metrics


# ---------------------------------------------------------------------------
# Main Training Function (original – used for initial training from ARFF)
# ---------------------------------------------------------------------------
def train(csv_path: str):
    """
    Orchestrate the full training workflow:
    pipeline → SMOTE → train → threshold-tune → evaluate → save.

    Parameters
    ----------
    csv_path : str  –  Path to the raw NASA MDP ARFF or CSV file.
    """
    logger.info("=" * 60)
    logger.info("  NASA MDP Defect Predictor – Training Pipeline")
    logger.info("=" * 60)

    # ── 1. Data pipeline ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, feature_names, _ = run_pipeline(csv_path)

    # ── 2. SMOTE oversampling (training set only) ─────────────────────────
    X_train_bal, y_train_bal = apply_smote(X_train, y_train)

    # ── 3. Train Random Forest ────────────────────────────────────────────
    clf = train_random_forest(X_train_bal, y_train_bal)

    # ── 4. Threshold optimisation ─────────────────────────────────────────
    best_threshold = optimise_threshold(clf, X_test, y_test)

    # ── 5. Evaluate ───────────────────────────────────────────────────────
    metrics = evaluate_model(clf, X_test, y_test, best_threshold, feature_names)

    # ── 6. Save artefact ─────────────────────────────────────────────────
    save_artefact(clf, metrics, feature_names, MODEL_OUTPUT_PATH)

    logger.info("Training complete. Ready to serve via app.py.")
    return clf, metrics


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NASA MDP Defect Predictor – Training CLI")
    parser.add_argument("data_path", nargs="?", default="data/KC1.arff",
                        help="Path to raw ARFF or CSV dataset.")
    parser.add_argument("--retrain", action="store_true",
                        help="Run continuous learning from the live SQLite DB.")
    args = parser.parse_args()

    if args.retrain:
        metrics = retrain_pipeline()
        print("\n-- Retraining complete --")
        print("  Recall  : {:.4f}".format(metrics['recall']))
        print("  F1      : {:.4f}".format(metrics['f1']))
        print("  ROC-AUC : {:.4f}".format(metrics['roc_auc']))
    else:
        train(args.data_path)
