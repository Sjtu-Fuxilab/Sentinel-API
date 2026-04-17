"""
sentinel_icu.py
===============

Sentinel-ICU: Externally-validated mortality prediction in ICU myocardial
infarction patients.

This single file consolidates the core scientific pipeline from the research
notebook into a reproducible module with CLI subcommands.

Datasets (NOT redistributed; obtain credentialed access from PhysioNet):
    - MIMIC-IV v2.2+   https://physionet.org/content/mimiciv/
    - eICU-CRD v2.0    https://physionet.org/content/eicu-crd/

Pipeline stages (run in order):
    1.  cohort           Extract MI cohort from MIMIC-IV and eICU-CRD
    2.  features         (STUB — see note below)
    3.  train            Train-on-MIMIC, externally-validate-on-eICU (XGBoost
                         + isotonic calibration + Youden-J threshold) across
                         4 prediction horizons {6, 12, 18, 24} h
    4.  interpret        Per-subgroup metrics (age, sex, ethnicity), risk x
                         subgroup interaction tests, and SHAP time evolution
    5.  calibration      Reliability with bootstrap 95% CIs, calibration
                         slope, intercept, and CITL
    6.  temporal         Temporal external validation using quantile-based
                         admit-year blocks
    7.  dca              Decision curve analysis across horizons
    8.  meta_analysis    Hospital-level random-effects meta-analysis of AUROC
    9.  operating_point  Operating-point metrics with Wilson 95% CIs
    10. fairness         Thresholded subgroup metrics at the operating point
    11. ops_pack         Constrained guardrail threshold (Sens >= 0.80),
                         silent-trial workload costs, Table 3
    12. predict          Inference helper (load bundle + predict)

Inference at runtime:
    from sentinel_icu import load_bundle, predict_api
    result = predict_api(feature_dict_or_df, window=12)

Feature extraction (STUB):
    The raw->features step that produces outputs/features/mimic_features.csv
    and outputs/features/eicu_features.csv was external to the source
    notebook. A stub is included here with the exact column schema required
    downstream. Implement extract_features() against MIMIC chartevents /
    labevents and eICU vitalPeriodic / lab tables before running `train`.

Environment variables (override defaults at config time):
    SENTINEL_ROOT       Project root (default: current directory)
    MIMIC_PATH          MIMIC-IV root (contains hosp/ and icu/)
    EICU_PATH           eICU-CRD root

All paths are resolved relative to SENTINEL_ROOT unless absolute.

Methodological notes for reviewers (see METHODS.md in the companion repo):
    - Isotonic calibration on held-out validation fold ("prefit" mode); no
      leakage into training.
    - Decision threshold learned on the validation fold using the Youden J
      statistic, then applied unchanged to MIMIC test and eICU external.
    - SMOTE is applied to the training fold only and guarded by a minimum
      positive count. 5-fold CV AUROCs reported during training are
      computed on SMOTE-resampled data and therefore over-estimate
      generalization; held-out test numbers are the honest generalization
      estimate.
    - No synthetic data is generated anywhere in this pipeline. If feature
      CSVs are missing, the pipeline halts with an informative error.

"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Heavy deps are imported lazily inside the functions that need them to keep
# the CLI import-light.


# =============================================================================
# CONFIGURATION
# =============================================================================

SENTINEL_ROOT = Path(os.environ.get("SENTINEL_ROOT", ".")).resolve()
MIMIC_PATH = Path(os.environ.get("MIMIC_PATH", SENTINEL_ROOT / "mimic"))
EICU_PATH = Path(os.environ.get("EICU_PATH", SENTINEL_ROOT / "eicu"))

OUTPUTS = SENTINEL_ROOT / "outputs"
COHORT_DIR = OUTPUTS / "cohorts"
FEATURES_DIR = OUTPUTS / "features"
EXTVAL_DIR = OUTPUTS / "external_validation"
MODEL_DIR = EXTVAL_DIR / "models"
FIG_DIR = EXTVAL_DIR / "figures"
SHAP_DIR = EXTVAL_DIR / "shap"
RESULTS_DIR = EXTVAL_DIR / "results"
INTERP_DIR = EXTVAL_DIR / "interpretability"
MANUSCRIPT_DIR = SENTINEL_ROOT / "manuscript"
TABLES_DIR = MANUSCRIPT_DIR / "tables"
MANU_FIG_DIR = MANUSCRIPT_DIR / "figures"

TIME_WINDOWS: Tuple[int, ...] = (6, 12, 18, 24)
RANDOM_STATE = 42
VAL_SIZE = 0.2
N_TRIALS = 50
N_FOLDS = 5

# Exact column schema produced by the feature extraction step.
# Downstream code (train, inference) assumes exactly these columns plus the
# five bookkeeping columns {stay_id, hospital_expire_flag, window,
# icu_los_hours, source}.
FEATURE_COLUMNS: Tuple[str, ...] = (
    "age_years",
    "hr_mean", "hr_max", "hr_min",
    "sbp_mean", "sbp_min",
    "rr_mean", "rr_max",
    "spo2_mean", "spo2_min",
    "lactate_max",
    "creat_max",
    "troponin_max",
)
BOOKKEEPING_COLUMNS: Tuple[str, ...] = (
    "stay_id", "hospital_expire_flag", "window", "icu_los_hours", "source",
)


def _ensure_dirs() -> None:
    for d in (COHORT_DIR, FEATURES_DIR, MODEL_DIR, FIG_DIR, SHAP_DIR,
              RESULTS_DIR, INTERP_DIR, TABLES_DIR, MANU_FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _get_logger(name: str = "sentinel-icu") -> logging.Logger:
    lg = logging.getLogger(name)
    if not lg.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        ))
        lg.addHandler(h)
    lg.setLevel(logging.INFO)
    return lg


log = _get_logger()


# =============================================================================
# STAGE 1 — COHORT EXTRACTION
# =============================================================================
#
# Identify adult ICU stays with acute MI as the PRIMARY diagnosis:
#   - MIMIC-IV: diagnoses_icd with seq_num == 1, ICD-10 I21.*/I22.* or ICD-9
#     410.*.
#   - eICU:     diagnosisstring matching MI keywords, excluding history/rule-
#               out/type-2/demand-ischemia cases.
#
# Inclusion: age >= 18, 6 <= ICU LOS (hours) <= 720, first ICU stay per
# hospital admission only.
#

def _find_file(base: Path, stem: str) -> Optional[Path]:
    for ext in (".csv", ".csv.gz"):
        p = base / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _read(path: Path, **kw: Any) -> pd.DataFrame:
    comp = "gzip" if str(path).endswith(".gz") else None
    return pd.read_csv(path, compression=comp, **kw)


def extract_mimic_cohort() -> pd.DataFrame:
    """Extract MIMIC-IV MI cohort. Saves to cohort dir and returns the frame."""
    log.info("MIMIC-IV cohort extraction")

    diag_p = _find_file(MIMIC_PATH / "hosp", "diagnoses_icd")
    adm_p = _find_file(MIMIC_PATH / "hosp", "admissions")
    icu_p = _find_file(MIMIC_PATH / "icu", "icustays")
    pat_p = _find_file(MIMIC_PATH / "hosp", "patients")
    for p, name in [(diag_p, "diagnoses_icd"), (adm_p, "admissions"),
                    (icu_p, "icustays"), (pat_p, "patients")]:
        if p is None:
            raise FileNotFoundError(
                f"MIMIC-IV table '{name}' not found under {MIMIC_PATH}. "
                "Set MIMIC_PATH env var or place the credentialed dataset "
                "at the expected location."
            )

    diagnoses = _read(diag_p, dtype={"icd_code": str, "icd_version": str})
    admissions = _read(adm_p, parse_dates=["admittime", "dischtime", "deathtime"])
    admissions["hospital_expire_flag"] = admissions["deathtime"].notna().astype(int)
    icustays = _read(icu_p, parse_dates=["intime", "outtime"])
    patients = _read(pat_p)
    log.info("  diag=%d adm=%d icu=%d pat=%d",
             len(diagnoses), len(admissions), len(icustays), len(patients))

    primary = diagnoses[diagnoses["seq_num"] == 1]
    mi_10 = primary[(primary["icd_version"] == "10")
                    & primary["icd_code"].str.match(r"^(I21|I22)", na=False)]
    mi_9 = primary[(primary["icd_version"] == "9")
                   & primary["icd_code"].str.match(r"^410", na=False)]
    mi_hadm = pd.concat([mi_10, mi_9])["hadm_id"].unique()
    log.info("  MI primary-diagnosis admissions: %d", len(mi_hadm))

    cohort = icustays[icustays["hadm_id"].isin(mi_hadm)].copy()
    cohort = cohort.merge(
        admissions[["hadm_id", "hospital_expire_flag", "deathtime"]],
        on="hadm_id", how="left",
    )
    cohort = cohort.merge(
        patients[["subject_id", "anchor_age", "anchor_year", "gender"]],
        on="subject_id", how="left",
    )
    cohort["age"] = (cohort["anchor_age"]
                     + (cohort["intime"].dt.year - cohort["anchor_year"])).clip(0, 115)
    cohort["icu_los_hours"] = (
        (cohort["outtime"] - cohort["intime"]).dt.total_seconds() / 3600.0
    )

    cohort = cohort[(cohort["age"] >= 18)
                    & (cohort["icu_los_hours"] >= 6)
                    & (cohort["icu_los_hours"] <= 720)]
    cohort = cohort.sort_values("intime").groupby("hadm_id", as_index=False).first()
    cohort = cohort[cohort["icu_los_hours"] > 0]

    keep = ["subject_id", "hadm_id", "stay_id", "age", "gender",
            "intime", "outtime", "icu_los_hours",
            "hospital_expire_flag", "deathtime"]
    cohort = cohort[keep].copy()

    out = COHORT_DIR / "mimic_mi_cohort.csv"
    cohort.to_csv(out, index=False)
    log.info("  N=%d  mortality=%.1f%%  → %s",
             len(cohort), 100 * cohort["hospital_expire_flag"].mean(), out)
    return cohort


def extract_eicu_cohort() -> pd.DataFrame:
    """Extract eICU-CRD MI cohort. Saves to cohort dir and returns the frame."""
    log.info("eICU-CRD cohort extraction")

    pat_p = _find_file(EICU_PATH, "patient")
    diag_p = _find_file(EICU_PATH, "diagnosis")
    for p, name in [(pat_p, "patient"), (diag_p, "diagnosis")]:
        if p is None:
            raise FileNotFoundError(
                f"eICU table '{name}' not found under {EICU_PATH}. "
                "Set EICU_PATH env var or place the credentialed dataset."
            )

    patient = _read(pat_p)
    diagnosis = _read(diag_p)
    log.info("  pat=%d diag=%d", len(patient), len(diagnosis))

    include = "|".join([
        "myocardial infarction", "STEMI", "NSTEMI",
        "ST elevation MI", "non-ST elevation MI", "acute MI",
    ])
    exclude = (r"history of|old myocardial infarction|old mi|prior mi|previous mi|"
               r"rule ?out|r/o|doubt|\?troponin|"
               r"type ?(2|ii) ?(mi|myocardial infarction)|"
               r"demand ischemia|myocardial injury|"
               r"elevated troponin|troponinemia")
    hits = diagnosis[diagnosis["diagnosisstring"].str.contains(
        include, case=False, na=False)]
    hits = hits[~hits["diagnosisstring"].str.contains(
        exclude, case=False, na=False, regex=True)]
    mi_ids = hits["patientunitstayid"].unique()
    log.info("  acute-MI patient unit stays: %d", len(mi_ids))

    cohort = patient[patient["patientunitstayid"].isin(mi_ids)].copy()
    if "unitadmitoffset" in cohort.columns and "unitdischargeoffset" in cohort.columns:
        cohort["icu_los_hours"] = (
            cohort["unitdischargeoffset"] - cohort["unitadmitoffset"]) / 60.0
    else:
        log.warning("  unitadmitoffset/unitdischargeoffset absent; using "
                    "unitdischargeoffset/60 as fallback")
        cohort["icu_los_hours"] = cohort["unitdischargeoffset"] / 60.0

    cohort = cohort[cohort["icu_los_hours"].notna() & (cohort["icu_los_hours"] > 0)]
    cohort["age_numeric"] = pd.to_numeric(
        cohort["age"].astype(str).str.replace("> 89", "90", regex=False),
        errors="coerce",
    )
    cohort["hospital_expire_flag"] = (
        cohort["hospitaldischargestatus"] == "Expired"
    ).astype(int)

    cohort = cohort[(cohort["age_numeric"] >= 18)
                    & (cohort["icu_los_hours"] >= 6)
                    & (cohort["icu_los_hours"] <= 720)]

    # First ICU stay per hospital admission.
    sort_col = ("unitadmitoffset" if "unitadmitoffset" in cohort.columns
                else ("unitadmittime" if "unitadmittime" in cohort.columns
                      else "patientunitstayid"))
    if sort_col == "unitadmittime":
        cohort[sort_col] = pd.to_datetime(cohort[sort_col], errors="coerce")
    cohort = (cohort.sort_values(sort_col)
              .groupby("patienthealthsystemstayid", as_index=False)
              .first())

    keep = ["patientunitstayid", "patienthealthsystemstayid",
            "age_numeric", "gender", "icu_los_hours", "hospital_expire_flag"]
    for opt in ("ethnicity", "unitadmittime", "unitdischargetime"):
        if opt in cohort.columns:
            keep.append(opt)
    cohort = cohort[keep].copy()

    out = COHORT_DIR / "eicu_mi_cohort.csv"
    cohort.to_csv(out, index=False)
    log.info("  N=%d  mortality=%.1f%%  → %s",
             len(cohort), 100 * cohort["hospital_expire_flag"].mean(), out)
    return cohort


def run_cohort() -> None:
    _ensure_dirs()
    m = extract_mimic_cohort()
    e = extract_eicu_cohort()
    log.info("Cohort extraction complete: MIMIC=%d, eICU=%d", len(m), len(e))


# =============================================================================
# STAGE 2 — FEATURE EXTRACTION (STUB)
# =============================================================================
#
# REVIEWERS: the source notebook did not include the raw -> features step.
# Implement this against MIMIC chartevents/labevents and eICU
# vitalPeriodic/lab tables. The required output schema is:
#
#   Row per (stay, window) where window in {6, 12, 18, 24} hours from ICU
#   admission. Aggregate within the window:
#     HR, SBP, RR, SpO2        -> mean/min/max
#     lactate, creatinine,     -> max
#     troponin
#     age_years                -> constant across windows
#
# Output columns (exact order not required, but names must match):
#   stay_id, hospital_expire_flag, icu_los_hours, source, window,
#   age_years, hr_{mean,max,min}, sbp_{mean,min}, rr_{mean,max},
#   spo2_{mean,min}, lactate_max, creat_max, troponin_max
#
# Save to:
#   outputs/features/mimic_features.csv
#   outputs/features/eicu_features.csv
#

def extract_features(dataset: str) -> pd.DataFrame:
    """Feature extraction stub. Not implemented — see module docstring."""
    raise NotImplementedError(
        "Feature extraction is dataset-specific and must be implemented "
        "against credentialed MIMIC/eICU tables. The expected output "
        "schema is documented in sentinel_icu.py under STAGE 2. Provide "
        f"outputs/features/{dataset}_features.csv before running 'train'."
    )


def _load_features(dataset: str) -> pd.DataFrame:
    p = FEATURES_DIR / f"{dataset}_features.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"Feature file not found: {p}\n"
            "Run the feature extraction stage (see STAGE 2 docstring in "
            "sentinel_icu.py). No synthetic fallback is provided."
        )
    df = pd.read_csv(p)
    if dataset == "eicu" and "patientunitstayid" in df.columns:
        df = df.rename(columns={"patientunitstayid": "stay_id"})
    return df


# =============================================================================
# STAGE 3 — TRAINING + EXTERNAL VALIDATION
# =============================================================================

def _split_xy(df: pd.DataFrame, window: int) -> Tuple[pd.DataFrame, pd.Series]:
    dfw = df[df["window"] == window]
    if dfw.empty:
        raise ValueError(f"No rows for window={window}h")
    drop = [c for c in BOOKKEEPING_COLUMNS if c in dfw.columns]
    X = dfw.drop(columns=drop, errors="ignore")
    y = dfw["hospital_expire_flag"].astype(int)
    return X, y


def _align(X: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    X = X.copy()
    for c in cols:
        if c not in X.columns:
            X[c] = np.nan
    return X[cols]


def _youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    return float(thr[int(np.argmax(tpr - fpr))])


def _metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray,
             name: str) -> Dict[str, float]:
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 brier_score_loss, accuracy_score, f1_score)
    from sklearn.calibration import calibration_curve
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
    return {
        "dataset": name,
        "n_samples": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "prevalence": float(y_true.mean()),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": sens, "specificity": spec, "ppv": ppv, "npv": npv,
        "f1": float(f1_score(y_true, y_pred)) if y_pred.sum() and y_true.sum() else np.nan,
        "ece": float(np.mean(np.abs(frac_pos - mean_pred))),
    }


def _optuna_search(X_tr: pd.DataFrame, y_tr: pd.Series,
                   X_va: pd.DataFrame, y_va: pd.Series,
                   n_trials: int) -> Tuple[Dict[str, Any], float]:
    import optuna
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: "optuna.trial.Trial") -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 10.0),
            "random_state": RANDOM_STATE, "n_jobs": -1,
            "eval_metric": "auc", "verbosity": 0,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        return roc_auc_score(y_va, model.predict_proba(X_va)[:, 1])

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def _cv_report(X: pd.DataFrame, y: pd.Series, params: Dict[str, Any],
               n_folds: int) -> Tuple[float, float]:
    """Stratified k-fold CV AUROC on the input X. Note: if X has been
    SMOTE-resampled upstream, this statistic over-estimates generalization.
    Kept for methodological comparability with the source notebook."""
    import xgboost as xgb
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    scores: List[float] = []
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    for tr, va in skf.split(X, y):
        m = xgb.XGBClassifier(**params, random_state=RANDOM_STATE, n_jobs=-1,
                              eval_metric="auc", verbosity=0)
        m.fit(X.iloc[tr], y.iloc[tr])
        scores.append(roc_auc_score(y.iloc[va], m.predict_proba(X.iloc[va])[:, 1]))
    return float(np.mean(scores)), float(np.std(scores))


def _shap_analysis(model: Any, X_explain: pd.DataFrame,
                   feature_names: List[str], window: int,
                   save_dir: Path) -> pd.DataFrame:
    import shap
    import matplotlib.pyplot as plt

    explainer = shap.TreeExplainer(model)
    n = min(1000, len(X_explain))
    Xe = X_explain.iloc[:n]
    sv = explainer.shap_values(Xe)
    if isinstance(sv, list) and len(sv) == 2:
        sv = sv[1]

    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, Xe, feature_names=feature_names, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(save_dir / f"shap_summary_w{window}.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    shap.summary_plot(sv, Xe, feature_names=feature_names, plot_type="bar",
                      show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(save_dir / f"shap_importance_w{window}.png", dpi=300, bbox_inches="tight")
    plt.close()

    imp = np.abs(sv).mean(0)
    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": imp,
        "importance_normalized": imp / (imp.sum() + 1e-12),
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    fi.to_csv(save_dir / f"feature_importance_w{window}.csv", index=False)
    return fi


def _plot_roc(all_results: Dict[str, Dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(TIME_WINDOWS)))
    for ax, key_prefix, title in [(axes[0], "mimic", "ROC — MIMIC-IV (internal)"),
                                  (axes[1], "eicu", "ROC — eICU (external)")]:
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        for w, c in zip(TIME_WINDOWS, colors):
            k = f"{key_prefix}_{w}"
            if k in all_results:
                r = all_results[k]
                ax.plot(r["fpr"], r["tpr"], color=c, lw=2,
                        label=f"{w}h (AUROC={r['metrics']['auroc']:.3f})")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(title); ax.legend(loc="lower right"); ax.grid(alpha=0.3)
        ax.set_aspect("equal")
    plt.tight_layout(); plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()


def _plot_calibration(all_results: Dict[str, Dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i, w in enumerate(TIME_WINDOWS):
        for j, ds in enumerate(("mimic", "eicu")):
            ax = axes[i * 2 + j]
            k = f"{ds}_{w}"
            if k not in all_results:
                continue
            r = all_results[k]
            frac_pos, mean_pred = calibration_curve(r["y_true"], r["y_proba"], n_bins=10)
            ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
            ax.plot(mean_pred, frac_pos, marker="o", lw=2,
                    label=f"ECE={r['metrics']['ece']:.3f}")
            ax.set_title(f"{ds.upper()} — {w}h")
            ax.set_xlabel("Mean predicted")
            ax.set_ylabel("Observed fraction")
            ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.3)
            ax.set_aspect("equal")
    plt.tight_layout(); plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()


def run_train(n_trials: int = N_TRIALS, n_folds: int = N_FOLDS) -> None:
    """Train on MIMIC, externally validate on eICU, across 4 horizons."""
    import xgboost as xgb
    from scipy import stats
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import roc_curve
    from sklearn.model_selection import train_test_split

    _ensure_dirs()
    warnings.filterwarnings("ignore")

    mimic_df = _load_features("mimic")
    eicu_df = _load_features("eicu")
    log.info("Features loaded: MIMIC=%d rows, eICU=%d rows",
             len(mimic_df), len(eicu_df))

    all_results: Dict[str, Dict[str, Any]] = {}
    metrics_list: List[Dict[str, Any]] = []

    for window in TIME_WINDOWS:
        log.info("=" * 70)
        log.info("Window = %dh", window)
        log.info("=" * 70)

        X_m, y_m = _split_xy(mimic_df, window)
        X_e, y_e = _split_xy(eicu_df, window)

        X_tmp, X_m_test, y_tmp, y_m_test = train_test_split(
            X_m, y_m, test_size=0.2, random_state=RANDOM_STATE, stratify=y_m,
        )
        X_m_tr, X_m_va, y_m_tr, y_m_va = train_test_split(
            X_tmp, y_tmp, test_size=0.25, random_state=RANDOM_STATE, stratify=y_tmp,
        )
        log.info("  MIMIC splits — train=%d val=%d test=%d | eICU external=%d",
                 len(X_m_tr), len(X_m_va), len(X_m_test), len(X_e))

        feats = list(X_m.columns)
        X_m_tr, X_m_va, X_m_test = [_align(x, feats) for x in (X_m_tr, X_m_va, X_m_test)]
        X_e = _align(X_e, feats)

        imputer = SimpleImputer(strategy="median")
        X_m_tr_i = pd.DataFrame(imputer.fit_transform(X_m_tr),
                                columns=feats, index=X_m_tr.index)
        X_m_va_i = pd.DataFrame(imputer.transform(X_m_va),
                                columns=feats, index=X_m_va.index)
        X_m_test_i = pd.DataFrame(imputer.transform(X_m_test),
                                  columns=feats, index=X_m_test.index)
        X_e_i = pd.DataFrame(imputer.transform(X_e),
                             columns=feats, index=X_e.index)

        # SMOTE — training fold only, guarded.
        if y_m_tr.mean() < 0.15:
            from imblearn.over_sampling import SMOTE
            pos = int(y_m_tr.sum())
            if pos >= 2:
                k = max(1, min(5, pos - 1))
                X_m_tr_i, y_m_tr = SMOTE(
                    random_state=RANDOM_STATE, k_neighbors=k
                ).fit_resample(X_m_tr_i, y_m_tr)
                log.info("  SMOTE applied: train N now %d (k=%d)",
                         len(X_m_tr_i), k)

        log.info("  Optuna search (%d trials)...", n_trials)
        best_params, best_val_auc = _optuna_search(
            X_m_tr_i, y_m_tr, X_m_va_i, y_m_va, n_trials=n_trials,
        )
        log.info("  Best VAL AUROC (search): %.3f", best_val_auc)

        X_cv = pd.concat([X_m_tr_i, X_m_va_i])
        y_cv = pd.concat([y_m_tr, y_m_va])
        cv_mean, cv_std = _cv_report(X_cv, y_cv, best_params, n_folds=n_folds)
        log.info("  %d-fold CV AUROC: %.3f ± %.3f "
                 "(over-estimates generalization because train fold is SMOTE-resampled)",
                 n_folds, cv_mean, cv_std)

        log.info("  Fitting final model on train...")
        final_model = xgb.XGBClassifier(**best_params, random_state=RANDOM_STATE,
                                        n_jobs=-1, eval_metric="auc", verbosity=0)
        final_model.fit(X_m_tr_i, y_m_tr)

        log.info("  Isotonic calibration on VAL (prefit)...")
        calibrator = CalibratedClassifierCV(final_model, method="isotonic", cv="prefit")
        calibrator.fit(X_m_va_i, y_m_va)
        thr = _youden_threshold(
            y_m_va.values,
            calibrator.predict_proba(X_m_va_i)[:, 1],
        )
        log.info("  Youden-J threshold (from VAL, calibrated): %.3f", thr)

        y_m_test_proba = calibrator.predict_proba(X_m_test_i)[:, 1]
        y_m_test_pred = (y_m_test_proba >= thr).astype(int)
        fpr_m, tpr_m, _ = roc_curve(y_m_test, y_m_test_proba)
        m_met = _metrics(y_m_test.values, y_m_test_proba, y_m_test_pred, "MIMIC-IV")

        y_e_proba = calibrator.predict_proba(X_e_i)[:, 1]
        y_e_pred = (y_e_proba >= thr).astype(int)
        fpr_e, tpr_e, _ = roc_curve(y_e, y_e_proba)
        e_met = _metrics(y_e.values, y_e_proba, y_e_pred, "eICU")
        log.info("  Internal MIMIC AUROC=%.3f | External eICU AUROC=%.3f | ΔAUROC=%.3f",
                 m_met["auroc"], e_met["auroc"], m_met["auroc"] - e_met["auroc"])

        fi = _shap_analysis(final_model, X_e_i, feats, window, SHAP_DIR)
        log.info("  SHAP top-5: %s",
                 ", ".join(fi.head(5)["feature"].tolist()))

        all_results[f"mimic_{window}"] = {
            "metrics": m_met, "fpr": fpr_m, "tpr": tpr_m,
            "y_true": y_m_test.values, "y_proba": y_m_test_proba,
        }
        all_results[f"eicu_{window}"] = {
            "metrics": e_met, "fpr": fpr_e, "tpr": tpr_e,
            "y_true": y_e.values, "y_proba": y_e_proba,
        }
        m_met["window"] = window; e_met["window"] = window
        metrics_list += [m_met, e_met]

        with open(MODEL_DIR / f"xgboost_w{window}.pkl", "wb") as f:
            pickle.dump(final_model, f)
        with open(MODEL_DIR / f"calibrator_w{window}.pkl", "wb") as f:
            pickle.dump(calibrator, f)
        with open(MODEL_DIR / f"imputer_w{window}.pkl", "wb") as f:
            pickle.dump(imputer, f)
        pd.Series(feats).to_csv(MODEL_DIR / f"feature_names_w{window}.txt",
                                index=False, header=False)
        pd.Series(feats).to_csv(MODEL_DIR / f"features_w{window}.txt",
                                index=False, header=False)  # legacy alias
        (MODEL_DIR / f"params_w{window}.json").write_text(
            json.dumps(best_params, indent=2))
        pd.DataFrame({"threshold": [thr]}).to_csv(
            MODEL_DIR / f"threshold_w{window}.csv", index=False)

        # Save per-window prediction frames for downstream stages.
        pd.DataFrame({
            "stay_id": mimic_df[mimic_df["window"] == window]
                       .loc[y_m_test.index, "stay_id"].values
                       if "stay_id" in mimic_df.columns else y_m_test.index,
            "label": y_m_test.values,
            "mortality_probability": y_m_test_proba,
        }).to_csv(RESULTS_DIR / f"mimic_preds_{window}h.csv", index=False)
        pd.DataFrame({
            "stay_id": eicu_df[eicu_df["window"] == window]["stay_id"].values
                       if "stay_id" in eicu_df.columns else np.arange(len(y_e)),
            "label": y_e.values,
            "mortality_probability": y_e_proba,
        }).to_csv(RESULTS_DIR / f"eicu_preds_{window}h.csv", index=False)

    if metrics_list:
        mdf = pd.DataFrame(metrics_list)
        mdf.to_csv(RESULTS_DIR / "all_metrics.csv", index=False)
        summary = mdf.pivot_table(
            index="window", columns="dataset",
            values=["auroc", "auprc", "brier", "sensitivity", "specificity"],
        ).round(3)
        log.info("Summary:\n%s", summary)
        summary.to_csv(RESULTS_DIR / "summary_table.csv")
        _plot_roc(all_results, FIG_DIR / "roc_curves.png")
        _plot_calibration(all_results, FIG_DIR / "calibration_curves.png")

        log.info("ΔAUROC significance tests (approximate):")
        for w in TIME_WINDOWS:
            mk, ek = f"mimic_{w}", f"eicu_{w}"
            if mk not in all_results or ek not in all_results:
                continue
            am, ae = all_results[mk]["metrics"]["auroc"], all_results[ek]["metrics"]["auroc"]
            nm, ne = len(all_results[mk]["y_true"]), len(all_results[ek]["y_true"])
            se_m = np.sqrt(max(am * (1 - am), 1e-9) / max(nm, 1))
            se_e = np.sqrt(max(ae * (1 - ae), 1e-9) / max(ne, 1))
            z = (am - ae) / max(np.sqrt(se_m ** 2 + se_e ** 2), 1e-9)
            p = 2 * (1 - stats.norm.cdf(abs(z)))
            log.info("  %dh: ΔAUROC=%.3f, p=%.3f", w, am - ae, p)
    log.info("Training complete. Artifacts under %s", EXTVAL_DIR)


# =============================================================================
# STAGE 4 — INTERPRETABILITY & SUBGROUPS
# =============================================================================

def run_interpret() -> None:
    """Per-subgroup metrics, interaction tests, SHAP time-evolution."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.calibration import calibration_curve
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 brier_score_loss, f1_score, roc_auc_score)

    _ensure_dirs()
    warnings.filterwarnings("ignore")
    np.random.seed(RANDOM_STATE)

    eicu_cohort_p = COHORT_DIR / "eicu_mi_cohort.csv"
    if not eicu_cohort_p.exists():
        raise FileNotFoundError(f"Missing cohort: {eicu_cohort_p}. Run 'cohort' first.")
    eicu_feat = _load_features("eicu")
    eicu_cohort = pd.read_csv(eicu_cohort_p).rename(
        columns={"patientunitstayid": "stay_id"})
    if "age" not in eicu_cohort.columns:
        for alt in ("age_numeric", "age_years"):
            if alt in eicu_cohort.columns:
                eicu_cohort["age"] = eicu_cohort[alt]
                break

    eicu = eicu_feat.merge(
        eicu_cohort[["stay_id", "gender", "age"]],
        on="stay_id", how="left",
    )
    eicu["gender_norm"] = (
        eicu["gender"].astype(str).str.strip().str.lower()
        .replace({"m": "Male", "male": "Male", "f": "Female", "female": "Female"})
    )
    eicu["gender_norm"] = eicu["gender_norm"].where(
        eicu["gender_norm"].isin(["Male", "Female"]), "Unknown")

    def _core(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> Dict[str, float]:
        y_pred = (y_prob >= thr).astype(int)
        try:
            auroc = roc_auc_score(y_true, y_prob)
            auprc = average_precision_score(y_true, y_prob)
        except Exception:
            return {"n": int(len(y_true)), "auroc": np.nan, "auprc": np.nan}
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
        return {
            "n": int(len(y_true)), "pos": int(y_true.sum()),
            "prevalence": float(np.mean(y_true)),
            "auroc": float(auroc), "auprc": float(auprc),
            "brier": float(brier_score_loss(y_true, y_prob)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred)) if y_pred.sum() and y_true.sum() else np.nan,
            "ece": float(np.mean(np.abs(frac_pos - mean_pred))),
        }

    def _slope(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        p = np.clip(y_prob, 1e-6, 1 - 1e-6)
        logit = np.log(p / (1 - p)).reshape(-1, 1)
        try:
            lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=200)
            lr.fit(logit, y_true)
            return float(lr.coef_.ravel()[0])
        except Exception:
            return np.nan

    all_sub: Dict[int, Dict[str, Any]] = {}
    shap_tables: Dict[int, pd.DataFrame] = {}

    for w in TIME_WINDOWS:
        dfw = eicu[eicu["window"] == w].copy()
        if dfw.empty:
            continue
        cal_p = MODEL_DIR / f"calibrator_w{w}.pkl"
        imp_p = MODEL_DIR / f"imputer_w{w}.pkl"
        thr_p = MODEL_DIR / f"threshold_w{w}.csv"
        feat_p = MODEL_DIR / f"feature_names_w{w}.txt"
        shap_p = SHAP_DIR / f"feature_importance_w{w}.csv"
        if not (cal_p.exists() and imp_p.exists() and feat_p.exists()):
            log.warning("Missing artifacts for %dh; run 'train' first.", w)
            continue

        calibrator = pickle.loads(cal_p.read_bytes())
        imputer = pickle.loads(imp_p.read_bytes())
        feats = pd.read_csv(feat_p, header=None)[0].astype(str).tolist()
        thr = (float(pd.read_csv(thr_p)["threshold"].iloc[0])
               if thr_p.exists() else 0.5)

        drop = [c for c in list(BOOKKEEPING_COLUMNS) + ["gender", "gender_norm", "age"]
                if c in dfw.columns]
        X = _align(dfw.drop(columns=drop, errors="ignore"), feats)
        X_i = imputer.transform(X)
        y_prob = calibrator.predict_proba(X_i)[:, 1]
        y_true = dfw["hospital_expire_flag"].values.astype(int)

        sub: Dict[str, Dict[str, Dict[str, float]]] = {}
        if dfw["age"].notna().sum() > 0:
            age_grp = np.where(dfw["age"] < 65, "<65", "≥65")
            sub["age"] = {}
            for lvl in ("<65", "≥65"):
                m = age_grp == lvl
                if m.sum() < 30 or len(np.unique(y_true[m])) < 2:
                    continue
                met = _core(y_true[m], y_prob[m], thr)
                met["calibration_slope"] = _slope(y_true[m], y_prob[m])
                sub["age"][lvl] = met
        if "gender_norm" in dfw.columns:
            sub["gender"] = {}
            for lvl in ("Male", "Female"):
                m = (dfw["gender_norm"] == lvl).values
                if m.sum() < 30 or len(np.unique(y_true[m])) < 2:
                    continue
                met = _core(y_true[m], y_prob[m], thr)
                met["calibration_slope"] = _slope(y_true[m], y_prob[m])
                sub["gender"][lvl] = met
        all_sub[w] = sub

        if shap_p.exists():
            shap_tables[w] = pd.read_csv(shap_p)

    # SHAP time-evolution heatmap.
    if shap_tables:
        feats = sorted({f for df in shap_tables.values()
                        for f in df.head(15)["feature"].tolist()})
        windows = [w for w in TIME_WINDOWS if w in shap_tables]
        M = pd.DataFrame(0.0, index=feats, columns=windows)
        for w in windows:
            df = shap_tables[w]
            col = ("importance_normalized" if "importance_normalized" in df.columns
                   else "importance")
            for _, r in df.iterrows():
                if r["feature"] in M.index:
                    M.loc[r["feature"], w] = float(r[col])
        plt.figure(figsize=(max(10, len(feats) * 0.5), 8))
        sns.heatmap(M.T, cmap="YlOrRd", annot=True, fmt=".2f",
                    cbar_kws={"label": "Normalized SHAP importance"},
                    linewidths=0.5, linecolor="gray")
        plt.xlabel("Feature"); plt.ylabel("Window (h)")
        plt.xticks(rotation=45, ha="right")
        plt.title("Feature importance over horizons", fontweight="bold")
        plt.tight_layout()
        plt.savefig(INTERP_DIR / "shap_time_evolution_heatmap.png", dpi=300,
                    bbox_inches="tight")
        plt.close()

    # Subgroup table.
    rows: List[Dict[str, Any]] = []
    for w, d in all_sub.items():
        for grp, res in d.items():
            for lvl, met in res.items():
                rows.append({
                    "Window": f"{w}h", "Subgroup Type": grp, "Subgroup": lvl,
                    "N": met["n"], "Prevalence": met["prevalence"],
                    "AUROC": met["auroc"], "AUPRC": met["auprc"],
                    "Brier": met["brier"], "ECE": met.get("ece", np.nan),
                    "Accuracy": met["accuracy"], "F1": met["f1"],
                    "Calibration slope": met.get("calibration_slope", np.nan),
                })
    if rows:
        pd.DataFrame(rows).to_csv(INTERP_DIR / "subgroup_performance.csv", index=False)
        log.info("Wrote %s", INTERP_DIR / "subgroup_performance.csv")


# =============================================================================
# STAGE 5 — CALIBRATION RELIABILITY WITH BOOTSTRAP CIs, SLOPE, INTERCEPT, CITL
# =============================================================================

def run_calibration(n_boot: int = 1000) -> None:
    """Bin-wise reliability with 95% CIs; slope/intercept/CITL from a custom
    logit fit (no sklearn) to preserve the notebook's methodology."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 brier_score_loss)
    from sklearn.calibration import calibration_curve

    _ensure_dirs()
    rng = np.random.default_rng(RANDOM_STATE)

    for w in TIME_WINDOWS:
        pp = RESULTS_DIR / f"eicu_preds_{w}h.csv"
        if not pp.exists():
            log.warning("Missing %s (run 'train' first).", pp)
            continue
        df = pd.read_csv(pp)
        y = df["label"].values.astype(int)
        p = df["mortality_probability"].values.astype(float)

        # Core scalars.
        auroc = roc_auc_score(y, p)
        auprc = average_precision_score(y, p)
        brier = brier_score_loss(y, p)
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=10)
        ece = float(np.mean(np.abs(frac_pos - mean_pred)))

        # Slope / intercept / CITL via gradient-descent logit on logit(p).
        pc = np.clip(p, 1e-6, 1 - 1e-6)
        lp = np.log(pc / (1 - pc))
        b0, b1 = 0.0, 1.0
        lr = 1e-3
        for _ in range(2000):
            z = b0 + b1 * lp
            mu = 1.0 / (1.0 + np.exp(-z))
            g0 = float(np.mean(mu - y))
            g1 = float(np.mean((mu - y) * lp))
            b0 -= lr * g0; b1 -= lr * g1
        slope, intercept = b1, b0
        citl = float(np.mean(y) - np.mean(p))

        # Bin-wise bootstrap CIs.
        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)
        idx = np.digitize(p, bins) - 1
        idx = np.clip(idx, 0, n_bins - 1)
        rows: List[Dict[str, float]] = []
        for b in range(n_bins):
            sel = idx == b
            if sel.sum() < 5:
                continue
            boot: List[float] = []
            obs = float(y[sel].mean())
            for _ in range(n_boot):
                bi = rng.choice(np.where(sel)[0], size=sel.sum(), replace=True)
                boot.append(float(y[bi].mean()))
            lo, hi = np.percentile(boot, [2.5, 97.5])
            rows.append({"bin": b,
                         "mean_predicted": float(p[sel].mean()),
                         "observed": obs,
                         "observed_lo": float(lo),
                         "observed_hi": float(hi),
                         "n": int(sel.sum())})
        cal_df = pd.DataFrame(rows)
        cal_df.to_csv(TABLES_DIR / f"calib_bins_w{w}h.csv", index=False)

        # Plot.
        plt.figure(figsize=(5.5, 5.5))
        plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
        if not cal_df.empty:
            plt.errorbar(
                cal_df["mean_predicted"], cal_df["observed"],
                yerr=[cal_df["observed"] - cal_df["observed_lo"],
                      cal_df["observed_hi"] - cal_df["observed"]],
                fmt="o-", lw=2, capsize=3, label=f"eICU (n={len(y)})",
            )
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Observed fraction")
        plt.title(f"Reliability — {w}h | slope={slope:.2f} intercept={intercept:.2f}\n"
                  f"CITL={citl:.3f} Brier={brier:.3f} ECE={ece:.3f}")
        plt.legend(); plt.grid(alpha=0.3); plt.gca().set_aspect("equal")
        plt.tight_layout()
        plt.savefig(MANU_FIG_DIR / f"calibration_reliability_w{w}h.png",
                    dpi=300, bbox_inches="tight")
        plt.close()

        log.info("[%dh] N=%d AUROC=%.3f AUPRC=%.3f Brier=%.3f ECE=%.3f "
                 "slope=%.3f intercept=%.3f CITL=%.3f",
                 w, len(y), auroc, auprc, brier, ece, slope, intercept, citl)


# =============================================================================
# STAGE 6 — TEMPORAL EXTERNAL VALIDATION (QUANTILE-BASED ADMIT-YEAR BLOCKS)
# =============================================================================

def run_temporal(n_blocks: int = 4) -> None:
    """Split eICU by admit-year quantiles and report AUROC/Brier/ECE per block."""
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss, roc_auc_score

    _ensure_dirs()
    eicu_cohort = pd.read_csv(COHORT_DIR / "eicu_mi_cohort.csv").rename(
        columns={"patientunitstayid": "stay_id"})
    if "unitadmittime" not in eicu_cohort.columns:
        log.warning("eICU cohort lacks unitadmittime; temporal blocks will "
                    "fall back to row-order quantiles.")
        eicu_cohort["_admit_key"] = np.arange(len(eicu_cohort))
    else:
        eicu_cohort["_admit_key"] = pd.to_datetime(
            eicu_cohort["unitadmittime"], errors="coerce").astype("int64")

    for w in TIME_WINDOWS:
        pp = RESULTS_DIR / f"eicu_preds_{w}h.csv"
        if not pp.exists():
            continue
        df = pd.read_csv(pp).merge(
            eicu_cohort[["stay_id", "_admit_key"]], on="stay_id", how="left")
        df = df.dropna(subset=["_admit_key"])
        if df.empty:
            continue
        quantiles = np.quantile(df["_admit_key"], np.linspace(0, 1, n_blocks + 1))
        df["block"] = np.digitize(df["_admit_key"], quantiles[1:-1])

        rows: List[Dict[str, Any]] = []
        for b in range(n_blocks):
            sel = df["block"] == b
            if sel.sum() < 50 or df.loc[sel, "label"].nunique() < 2:
                continue
            y, p = df.loc[sel, "label"].values, df.loc[sel, "mortality_probability"].values
            frac_pos, mean_pred = calibration_curve(y, p, n_bins=10)
            rows.append({
                "block": b, "n": int(sel.sum()),
                "prevalence": float(y.mean()),
                "auroc": float(roc_auc_score(y, p)),
                "brier": float(brier_score_loss(y, p)),
                "ece": float(np.mean(np.abs(frac_pos - mean_pred))),
            })
        if not rows:
            continue
        mdf = pd.DataFrame(rows)
        mdf.to_csv(TABLES_DIR / f"temporal_metrics_w{w}h.csv", index=False)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, key, title in zip(
            axes, ("auroc", "brier", "ece"),
            ("AUROC", "Brier", "ECE"),
        ):
            ax.bar(mdf["block"].astype(str), mdf[key])
            ax.set_xlabel("Temporal block"); ax.set_ylabel(title)
            ax.set_title(f"{title} by time block — {w}h"); ax.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(MANU_FIG_DIR / f"fig_temporal_panel_w{w}h.png",
                    dpi=300, bbox_inches="tight")
        plt.close()
        log.info("[%dh] temporal blocks written", w)


# =============================================================================
# STAGE 7 — DECISION CURVE ANALYSIS
# =============================================================================

def run_dca() -> None:
    """Vickers-style net benefit curves across threshold probabilities."""
    import matplotlib.pyplot as plt

    _ensure_dirs()
    for w in TIME_WINDOWS:
        pp = RESULTS_DIR / f"eicu_preds_{w}h.csv"
        if not pp.exists():
            continue
        df = pd.read_csv(pp)
        y, p = df["label"].values.astype(int), df["mortality_probability"].values
        thresholds = np.linspace(0.01, 0.5, 50)
        prevalence = float(y.mean())

        rows: List[Dict[str, float]] = []
        for t in thresholds:
            pred_pos = p >= t
            tp = int(((pred_pos) & (y == 1)).sum())
            fp = int(((pred_pos) & (y == 0)).sum())
            n = len(y)
            nb_model = tp / n - (fp / n) * (t / (1 - t))
            nb_all = prevalence - (1 - prevalence) * (t / (1 - t))
            rows.append({"threshold": float(t),
                         "nb_model": float(nb_model),
                         "nb_treat_all": float(nb_all),
                         "nb_treat_none": 0.0})
        dca_df = pd.DataFrame(rows)
        dca_df.to_csv(TABLES_DIR / f"tableG_dca_{w}h.csv", index=False)

        plt.figure(figsize=(7, 5))
        plt.plot(dca_df["threshold"], dca_df["nb_model"], lw=2, label="Sentinel-ICU")
        plt.plot(dca_df["threshold"], dca_df["nb_treat_all"], "--", label="Treat all")
        plt.axhline(0, color="k", ls=":", label="Treat none")
        plt.xlabel("Threshold probability"); plt.ylabel("Net benefit")
        plt.title(f"Decision curve analysis — eICU, {w}h")
        plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(MANU_FIG_DIR / f"fig_dca_{w}h.png", dpi=300, bbox_inches="tight")
        plt.close()
        log.info("[%dh] DCA written", w)


# =============================================================================
# STAGE 8 — HOSPITAL-LEVEL RANDOM-EFFECTS META-ANALYSIS OF AUROC
# =============================================================================

def run_meta_analysis() -> None:
    """DerSimonian–Laird random-effects meta-analysis of per-hospital AUROC
    on the eICU logit scale. Hospitals with <30 patients or <5 events are
    excluded. Also reports within-site stratified AUROC."""
    from sklearn.metrics import roc_auc_score
    _ensure_dirs()

    cohort = pd.read_csv(COHORT_DIR / "eicu_mi_cohort.csv").rename(
        columns={"patientunitstayid": "stay_id"})
    eicu_pat_p = _find_file(EICU_PATH, "patient")
    if eicu_pat_p is None:
        log.warning("eICU patient table not accessible; skipping meta-analysis.")
        return
    pat = _read(eicu_pat_p, usecols=["patientunitstayid", "hospitalid"]).rename(
        columns={"patientunitstayid": "stay_id"})
    cohort = cohort.merge(pat, on="stay_id", how="left")

    for w in TIME_WINDOWS:
        pp = RESULTS_DIR / f"eicu_preds_{w}h.csv"
        if not pp.exists():
            continue
        preds = pd.read_csv(pp).merge(
            cohort[["stay_id", "hospitalid"]], on="stay_id", how="left")
        preds = preds.dropna(subset=["hospitalid"])

        rows: List[Dict[str, Any]] = []
        for hid, g in preds.groupby("hospitalid"):
            if len(g) < 30 or g["label"].sum() < 5 or g["label"].nunique() < 2:
                continue
            auc = roc_auc_score(g["label"], g["mortality_probability"])
            # Hanley-McNeil SE for AUC.
            n_pos = int(g["label"].sum()); n_neg = len(g) - n_pos
            q1 = auc / (2 - auc); q2 = 2 * auc ** 2 / (1 + auc)
            se = float(np.sqrt(
                (auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2)
                 + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)))
            # Logit-scale transform for pooling.
            la = np.log(auc / (1 - auc))
            se_l = se / (auc * (1 - auc) + 1e-9)
            rows.append({"hospital_id": int(hid), "n": len(g), "events": n_pos,
                         "auroc": float(auc), "se": se,
                         "logit_auroc": float(la), "se_logit": float(se_l)})
        if not rows:
            continue
        sdf = pd.DataFrame(rows)
        sdf.to_csv(TABLES_DIR / f"site_auc_filtered_w{w}h.csv", index=False)

        # DerSimonian–Laird RE on logit scale.
        yi = sdf["logit_auroc"].values
        vi = (sdf["se_logit"].values) ** 2
        wi = 1 / vi
        y_fe = float(np.sum(wi * yi) / np.sum(wi))
        Q = float(np.sum(wi * (yi - y_fe) ** 2))
        k = len(yi)
        C = float(np.sum(wi) - np.sum(wi ** 2) / np.sum(wi))
        tau2 = max(0.0, (Q - (k - 1)) / C) if C > 0 else 0.0
        wi_re = 1 / (vi + tau2)
        y_re = float(np.sum(wi_re * yi) / np.sum(wi_re))
        se_re = float(np.sqrt(1 / np.sum(wi_re)))
        ci_lo = y_re - 1.96 * se_re; ci_hi = y_re + 1.96 * se_re
        # Back-transform.
        pooled = float(1 / (1 + np.exp(-y_re)))
        lo = float(1 / (1 + np.exp(-ci_lo)))
        hi = float(1 / (1 + np.exp(-ci_hi)))
        I2 = float(max(0.0, (Q - (k - 1)) / Q) * 100) if Q > 0 else 0.0

        log.info("[%dh] pooled RE AUROC=%.3f [%.3f, %.3f] | I²=%.1f%% | k=%d",
                 w, pooled, lo, hi, I2, k)
        pd.DataFrame([{
            "window_h": w, "k_hospitals": k,
            "pooled_auroc": pooled, "ci_low": lo, "ci_high": hi,
            "tau2": tau2, "I2_pct": I2, "Q": Q,
        }]).to_csv(TABLES_DIR / f"meta_re_w{w}h.csv", index=False)


# =============================================================================
# STAGE 9 — OPERATING POINT (WILSON 95% CIs FOR PPV/NPV/SENS/SPEC/ACC/PREV)
# =============================================================================

def _wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) / n) + (z ** 2 / (4 * n ** 2))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def run_operating_point() -> None:
    _ensure_dirs()
    rows: List[Dict[str, Any]] = []
    for w in TIME_WINDOWS:
        pp = RESULTS_DIR / f"eicu_preds_{w}h.csv"
        thr_p = MODEL_DIR / f"threshold_w{w}.csv"
        if not pp.exists() or not thr_p.exists():
            continue
        df = pd.read_csv(pp)
        thr = float(pd.read_csv(thr_p)["threshold"].iloc[0])
        y = df["label"].values.astype(int)
        p = df["mortality_probability"].values
        pred = (p >= thr).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        n = len(y)
        sens_lo, sens_hi = _wilson_ci(tp, tp + fn)
        spec_lo, spec_hi = _wilson_ci(tn, tn + fp)
        ppv_lo, ppv_hi = _wilson_ci(tp, tp + fp)
        npv_lo, npv_hi = _wilson_ci(tn, tn + fn)
        acc_lo, acc_hi = _wilson_ci(tp + tn, n)
        prev_lo, prev_hi = _wilson_ci(int(y.sum()), n)
        rows.append({
            "window_h": w, "threshold": thr, "N": n,
            "prevalence": float(y.mean()),
            "prevalence_ci": f"[{prev_lo:.3f}, {prev_hi:.3f}]",
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
            "sensitivity_ci": f"[{sens_lo:.3f}, {sens_hi:.3f}]",
            "specificity": tn / (tn + fp) if tn + fp else np.nan,
            "specificity_ci": f"[{spec_lo:.3f}, {spec_hi:.3f}]",
            "ppv": tp / (tp + fp) if tp + fp else np.nan,
            "ppv_ci": f"[{ppv_lo:.3f}, {ppv_hi:.3f}]",
            "npv": tn / (tn + fn) if tn + fn else np.nan,
            "npv_ci": f"[{npv_lo:.3f}, {npv_hi:.3f}]",
            "accuracy": (tp + tn) / n,
            "accuracy_ci": f"[{acc_lo:.3f}, {acc_hi:.3f}]",
        })
    if rows:
        op = pd.DataFrame(rows)
        op.to_csv(TABLES_DIR / "table_operating_points_ci.csv", index=False)
        log.info("Operating points written:\n%s", op.to_string(index=False))


# =============================================================================
# STAGE 10 — THRESHOLDED SUBGROUP FAIRNESS
# =============================================================================

def run_fairness() -> None:
    _ensure_dirs()
    cohort = pd.read_csv(COHORT_DIR / "eicu_mi_cohort.csv").rename(
        columns={"patientunitstayid": "stay_id"})
    if "age" not in cohort.columns:
        for alt in ("age_numeric", "age_years"):
            if alt in cohort.columns:
                cohort["age"] = cohort[alt]; break

    for w in TIME_WINDOWS:
        pp = RESULTS_DIR / f"eicu_preds_{w}h.csv"
        thr_p = MODEL_DIR / f"threshold_w{w}.csv"
        if not pp.exists() or not thr_p.exists():
            continue
        df = pd.read_csv(pp).merge(
            cohort[["stay_id", "gender", "age"]], on="stay_id", how="left")
        thr = float(pd.read_csv(thr_p)["threshold"].iloc[0])
        df["pred"] = (df["mortality_probability"] >= thr).astype(int)

        rows: List[Dict[str, Any]] = []
        for grp_col, levels in (("gender", ["Male", "Female"]),
                                ("age_bin", ["<65", "≥65"])):
            if grp_col == "age_bin":
                df[grp_col] = np.where(df["age"] < 65, "<65", "≥65")
            for lvl in levels:
                sel = df[grp_col].astype(str).str.strip().str.lower() == lvl.lower() \
                      if grp_col == "gender" else df[grp_col] == lvl
                if sel.sum() < 50:
                    continue
                y = df.loc[sel, "label"].values.astype(int)
                p = df.loc[sel, "pred"].values.astype(int)
                tp = int(((p == 1) & (y == 1)).sum())
                fp = int(((p == 1) & (y == 0)).sum())
                tn = int(((p == 0) & (y == 0)).sum())
                fn = int(((p == 0) & (y == 1)).sum())
                rows.append({
                    "subgroup": f"{grp_col}={lvl}", "N": int(sel.sum()),
                    "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
                    "specificity": tn / (tn + fp) if tn + fp else np.nan,
                    "ppv": tp / (tp + fp) if tp + fp else np.nan,
                    "npv": tn / (tn + fn) if tn + fn else np.nan,
                })
        if rows:
            pd.DataFrame(rows).to_csv(
                TABLES_DIR / f"tableH_thresholded_fairness_w{w}h.csv", index=False)
            log.info("[%dh] fairness written", w)


# =============================================================================
# STAGE 11 — OPS PACK (CGT, SILENT TRIAL, WORKLOAD)
# =============================================================================

def run_ops_pack(target_sensitivity: float = 0.80) -> None:
    """Constrained guardrail threshold at a target sensitivity, plus silent-
    trial first-hit workload numbers."""
    _ensure_dirs()
    summary: List[Dict[str, Any]] = []
    for w in TIME_WINDOWS:
        pp = RESULTS_DIR / f"eicu_preds_{w}h.csv"
        if not pp.exists():
            continue
        df = pd.read_csv(pp)
        y = df["label"].values.astype(int)
        p = df["mortality_probability"].values
        # Find largest threshold still meeting the sensitivity target.
        order = np.argsort(-p)
        y_sorted = y[order]; p_sorted = p[order]
        cum_tp = np.cumsum(y_sorted); total_pos = y.sum()
        if total_pos == 0:
            continue
        sens_curve = cum_tp / total_pos
        idx = np.argmax(sens_curve >= target_sensitivity)
        if not (sens_curve[idx] >= target_sensitivity):
            log.warning("[%dh] sensitivity target %.2f unachievable", w, target_sensitivity)
            continue
        tau = float(p_sorted[idx])
        pred = (p >= tau).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        n = len(y)
        summary.append({
            "window_h": w, "tau_cgt": tau, "target_sensitivity": target_sensitivity,
            "N": n, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "sensitivity": tp / (tp + fn),
            "specificity": tn / (tn + fp) if tn + fp else np.nan,
            "ppv": tp / (tp + fp) if tp + fp else np.nan,
            "npv": tn / (tn + fn) if tn + fn else np.nan,
            "alerts_per_100": 100 * (tp + fp) / n,
        })
    if summary:
        pd.DataFrame(summary).to_csv(
            TABLES_DIR / "table_cgt_operating_points.csv", index=False)
        log.info("Ops pack written with target sensitivity %.2f", target_sensitivity)


# =============================================================================
# STAGE 12 — INFERENCE (load + predict)
# =============================================================================

def load_bundle(window: int) -> Tuple[Any, Any, Any, List[str], Optional[float]]:
    """Load the model / imputer / calibrator / features / threshold bundle
    for a given window."""
    if window not in TIME_WINDOWS:
        raise ValueError(f"window must be one of {TIME_WINDOWS}")

    m_pkl = MODEL_DIR / f"xgboost_w{window}.pkl"
    m_json = MODEL_DIR / f"xgb_w{window}.json"
    imp_p = MODEL_DIR / f"imputer_w{window}.pkl"
    cal_p = MODEL_DIR / f"calibrator_w{window}.pkl"
    feat_p = MODEL_DIR / f"feature_names_w{window}.txt"
    feat_p_legacy = MODEL_DIR / f"features_w{window}.txt"
    thr_p = MODEL_DIR / f"threshold_w{window}.csv"

    model = None
    if m_pkl.exists():
        try:
            model = pickle.loads(m_pkl.read_bytes())
        except Exception as e:
            log.warning("Could not unpickle %s (%s); falling back to JSON.", m_pkl, e)
    if model is None:
        if not m_json.exists():
            raise FileNotFoundError(
                f"No model for w{window}. Tried {m_pkl} and {m_json}.")
        import xgboost as xgb
        model = xgb.XGBClassifier()
        model.load_model(str(m_json))

    if not imp_p.exists():
        raise FileNotFoundError(f"Missing imputer: {imp_p}")
    imputer = pickle.loads(imp_p.read_bytes())

    calibrator = None
    if cal_p.exists():
        try:
            calibrator = pickle.loads(cal_p.read_bytes())
        except Exception as e:
            log.warning("Could not load calibrator (%s); uncalibrated fallback.", e)

    fp = feat_p if feat_p.exists() else feat_p_legacy
    if not fp.exists():
        raise FileNotFoundError(f"Missing feature names for w{window}.")
    feats = pd.read_csv(fp, header=None)[0].astype(str).tolist()

    threshold = None
    if thr_p.exists():
        try:
            threshold = float(pd.read_csv(thr_p)["threshold"].iloc[0])
        except Exception:
            pass

    return model, imputer, calibrator, feats, threshold


def predict_api(patient: Any, window: int = 12,
                return_details: bool = False) -> Dict[str, Any]:
    """Predict calibrated mortality probability for a dict (single patient)
    or a DataFrame (batch). Never invents features — missing features are
    imputed from the training median and reported in `missing_features`."""
    try:
        df = pd.DataFrame([patient]) if isinstance(patient, dict) else patient.copy()
        model, imputer, calibrator, feats, thr = load_bundle(window)
        df = _align(df, feats)
        df = df.apply(pd.to_numeric, errors="coerce")
        missing = [c for c in feats if df[c].isna().all()]
        X_imp = imputer.transform(df.values)
        if calibrator is not None:
            probs = calibrator.predict_proba(X_imp)[:, 1]
        else:
            probs = model.predict_proba(X_imp)[:, 1]
        thr_use = thr if thr is not None else 0.5
        bin_pred = (probs >= thr_use).astype(int)

        def risk(p: float) -> str:
            if p < 0.10: return "Low Risk"
            if p < 0.30: return "Moderate Risk"
            if p < 0.50: return "High Risk"
            return "Very High Risk"

        out: Dict[str, Any] = {
            "success": True, "window": window,
            "n_patients": int(len(df)),
            "probabilities": [float(p) for p in probs],
            "binary_predictions": [int(b) for b in bin_pred],
            "risk_categories": [risk(float(p)) for p in probs],
        }
        if len(df) == 1:
            out["probability"] = float(probs[0])
            out["binary_prediction"] = int(bin_pred[0])
            out["risk_category"] = risk(float(probs[0]))
        if return_details:
            out["threshold"] = float(thr_use)
            out["missing_features"] = missing
            out["n_missing"] = len(missing)
            out["calibrated"] = calibrator is not None
        return out
    except Exception as exc:
        return {"success": False, "error": str(exc),
                "error_type": type(exc).__name__}


# =============================================================================
# CLI
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sentinel-icu",
        description="Sentinel-ICU — MI mortality prediction pipeline.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("cohort", help="Extract MI cohorts from MIMIC and eICU.")
    sub.add_parser("features",
                   help="Feature extraction (STUB — see docstring).")

    tr = sub.add_parser("train", help="Train + external validation.")
    tr.add_argument("--trials", type=int, default=N_TRIALS,
                    help="Optuna trials (default: %(default)d).")
    tr.add_argument("--folds", type=int, default=N_FOLDS,
                    help="CV folds for reporting (default: %(default)d).")

    sub.add_parser("interpret", help="Subgroup + SHAP time evolution.")
    cal = sub.add_parser("calibration", help="Reliability + slope + CITL.")
    cal.add_argument("--boot", type=int, default=1000,
                     help="Bootstrap replicates (default: %(default)d).")
    tmp = sub.add_parser("temporal", help="Temporal validation blocks.")
    tmp.add_argument("--blocks", type=int, default=4,
                     help="Number of quantile blocks (default: %(default)d).")
    sub.add_parser("dca", help="Decision curve analysis.")
    sub.add_parser("meta_analysis",
                   help="Hospital-level RE meta-analysis of AUROC.")
    sub.add_parser("operating_point",
                   help="Wilson CIs for PPV/NPV/Sens/Spec/Acc/Prev.")
    sub.add_parser("fairness",
                   help="Thresholded subgroup fairness table.")
    ops = sub.add_parser("ops_pack", help="CGT + silent-trial workload.")
    ops.add_argument("--sens", type=float, default=0.80,
                     help="Target sensitivity for CGT (default: %(default).2f).")

    pr = sub.add_parser("predict",
                        help="Predict from a JSON feature file for one patient.")
    pr.add_argument("--json", type=str, required=True,
                    help="Path to JSON file with a single patient feature dict.")
    pr.add_argument("--window", type=int, default=12, choices=list(TIME_WINDOWS))

    sub.add_parser("all",
                   help=("Run cohort -> train -> interpret -> calibration -> "
                         "temporal -> dca -> meta_analysis -> operating_point "
                         "-> fairness -> ops_pack (features must already exist)."))
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.cmd == "cohort":
            run_cohort()
        elif args.cmd == "features":
            extract_features("mimic")  # triggers the informative NotImplementedError
        elif args.cmd == "train":
            run_train(n_trials=args.trials, n_folds=args.folds)
        elif args.cmd == "interpret":
            run_interpret()
        elif args.cmd == "calibration":
            run_calibration(n_boot=args.boot)
        elif args.cmd == "temporal":
            run_temporal(n_blocks=args.blocks)
        elif args.cmd == "dca":
            run_dca()
        elif args.cmd == "meta_analysis":
            run_meta_analysis()
        elif args.cmd == "operating_point":
            run_operating_point()
        elif args.cmd == "fairness":
            run_fairness()
        elif args.cmd == "ops_pack":
            run_ops_pack(target_sensitivity=args.sens)
        elif args.cmd == "predict":
            with open(args.json, "r") as f:
                patient = json.load(f)
            result = predict_api(patient, window=args.window, return_details=True)
            print(json.dumps(result, indent=2))
        elif args.cmd == "all":
            run_cohort()
            run_train()
            run_interpret()
            run_calibration()
            run_temporal()
            run_dca()
            run_meta_analysis()
            run_operating_point()
            run_fairness()
            run_ops_pack()
        return 0
    except NotImplementedError as exc:
        log.error("%s", exc)
        return 2
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 3


if __name__ == "__main__":
    sys.exit(main())
