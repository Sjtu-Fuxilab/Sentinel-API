# Sentinel-ICU

Code for the retrospective external validation study *Staffable ICU Mortality Early-Warning System with Calibrated Multi-Centre Validation and Alert-Budget Guardrails*.

An XGBoost ICU mortality model with isotonic calibration, validated externally at 6, 12, 18, and 24 h post-admission landmarks. Outputs are framed as a staffable alert program: guardrail thresholds mapped to alert volume per 100 admissions, clinician-review workload, net monetary benefit, and silent-trial lead time.

## Key results (external validation, eICU-CRD)

| Horizon | AUROC | AUPRC | ECE   | Brier |
|:-------:|:-----:|:-----:|:-----:|:-----:|
| 6 h     | 0.857 | 0.546 | 0.023 | 0.076 |
| 12 h    | 0.864 | 0.535 | 0.046 | 0.081 |
| 18 h    | 0.835 | 0.453 | —     | 0.086 |
| 24 h    | 0.825 | 0.432 | —     | 0.091 |

Pooled random-effects AUROC at 12 h: **0.869 (95% CI 0.857–0.881)**. At the 12-h guardrail threshold τCGT = 0.086: 30.1 alerts / 100 admissions, PPV 0.322 (95% CI 0.310–0.335), median silent-trial lead time 138.6 h (IQR 54.6–286.3). Calibration slopes (0.234–0.347) indicate compressed log-odds — local recalibration is recommended before activation.

## Data

Credentialed access required:

- MIMIC-IV v2.2 — <https://physionet.org/content/mimiciv/> (DOI: 10.13026/6mm1-ek67)
- eICU-CRD v2.0 — <https://physionet.org/content/eicu-crd/> (DOI: 10.13026/C2WM1R)

No patient data is redistributed in this repository.

## Installation

Python ≥ 3.10.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export SENTINEL_ROOT=/path/to/project
export MIMIC_PATH=/path/to/mimic-iv
export EICU_PATH=/path/to/eicu-crd
```

## Usage

```bash
python sentinel_icu.py cohort      # 1. build cohorts
python sentinel_icu.py features    # 2. feature extraction (user-supplied)
python sentinel_icu.py train       # 3. MIMIC → eICU external validation
python sentinel_icu.py all         # 4. downstream analyses
```

The `all` command runs: `interpret` (subgroup + SHAP), `calibration` (reliability, slope, intercept, CITL), `temporal` (quantile-block validation), `dca` (decision-curve analysis), `meta_analysis` (hospital-level DerSimonian–Laird pooling), `operating_point` (Wilson CIs at τYJ), `fairness` (subgroup metrics at τCGT), and `ops_pack` (CGT at target sensitivity, workload and NMB readouts).

Inference:

```python
from sentinel_icu import predict_api
predict_api(patient_dict, window=12, return_details=True)
```

## Feature extraction

`extract_features()` is a stub. The raw-to-features step must be implemented against MIMIC-IV `chartevents`/`labevents` and eICU-CRD `vitalPeriodic`/`lab` tables. The required output schema (one row per `stay_id × window`, with `window ∈ {6, 12, 18, 24}`) is reported in the `NotImplementedError` the stub raises. No synthetic fallback is provided — the pipeline halts if feature CSVs are missing.

## Outputs

- `outputs/cohorts/` — MIMIC and eICU cohorts
- `outputs/external_validation/{models,results,figures,shap,interpretability}/` — trained artifacts and per-horizon diagnostics
- `manuscript/{tables,figures}/` — reliability plots, temporal panels, DCA, meta-analysis, operating-point Wilson CIs, CGT table

## Caveats

- **CV AUROCs are computed on SMOTE-resampled training folds** and therefore overstate generalization. The MIMIC test and eICU external numbers are the honest estimates.
- **Isotonic calibration can saturate at zero** in the lowest predicted-probability bin. This is intrinsic to isotonic regression; Platt or beta calibration can be swapped in at the `run_train` stage.
- **Local recalibration before activation** is recommended (slope < 1 in external data).


## Funding

Interdisciplinary Program of Shanghai Jiao Tong University, China (YG2025QNA31).

## License

MIT (see `LICENSE`). MIMIC-IV and eICU-CRD remain governed by their PhysioNet data-use agreements.

## Contact
Prof. Wei Qin (wqin@sjtu.edu.cn)

Sanwal Ahmad Zafar — <sanwalzafar@sjtu.edu.cn> · Corresponding author: Prof. Wei Qin — <wqin@sjtu.edu.cn>.
