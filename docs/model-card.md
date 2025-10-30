# Model Card — Sentinel-API (Draft)

**Version:** v0.1.2  \
**Date:** 2025-10-30  \
**Status:** Pre-publication (manuscript in preparation)

## Overview
Sentinel-API is a FastAPI-based service that provides **in-hospital mortality risk prediction** for ICU patients (e.g., MI cohort). This repository hosts the **serving code, API schema, and reproducibility assets**; it **does not** contain PHI or raw clinical datasets.

## Intended Use
- **Primary**: Research, benchmarking, and reproducible demonstrations by credentialed users with access to MIMIC-IV / eICU.
- **Out of scope**: Direct clinical deployment or use for bedside decision-making without formal validation, approval, and governance.

## Datasets
- **Development & evaluation**: MIMIC-IV, eICU Collaborative Research Database (research access required).
- **Data handling**: No PHI or raw patient data are committed to this repository. Only aggregated metrics/figures may be published.
- **Access**: Users must obtain credentials independently and follow dataset usage policies.

## Methods (high level)
- Features: time-aggregated vitals/labs/demographics.
- Model family: gradient-boosted decision trees (XGBoost) with hyperparameter optimization (Optuna).
- Validation: k-fold CV; separate validation by time windows (e.g., 6h/12h/24h/48h) where applicable.
- Interpretability: SHAP (global & local explanations), reported with caution.

## Performance (to be finalized)
- Add AUROC/AUPRC, calibration, sensitivity/specificity for each dataset/window.
- Include CIs/uncertainty. Provide exact splits and seeds once finalized.

## Ethical Considerations
- **Bias & fairness**: Evaluate across demographic subgroups where feasible; report disparities if found.
- **Clinical risk**: Model outputs are **not** clinical advice. Include guardrails in downstream tools.
- **Reproducibility**: Lockfile and container image digest are provided; `/version` endpoint exposes tag & git SHA.

## Limitations
- Dataset shift, missingness, and documentation artifacts may degrade performance.
- External validation is required before any deployment.

## How to cite
- See `CITATION.cff` in the repository. A DOI will be added post-preprint/publication.

## Contact
For questions or collaborations: open an issue or see contact in `README.md`.
