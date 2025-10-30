# Data Availability

This repository **does not** include raw clinical data or PHI.

- **MIMIC-IV** and **eICU** require credentialed access from their respective maintainers.
- Users must obtain dataset access independently and comply with all terms.
- The repository contains **code**, **API**, and **reproducibility assets** only.

## Reproduction outline
1. Obtain access to MIMIC-IV and/or eICU.
2. Prepare cohort extraction as described in `api/data/cohort_extraction.py`.
3. Install from `requirements.txt` (or use `requirements-lock.txt` for exact versions).
4. Train/evaluate models as documented; generate figures without exporting PHI.

If you discover any data exposure risks, please file a confidential security report (see `SECURITY.md`).
