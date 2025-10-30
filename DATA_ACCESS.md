# Data Access

This repository does not contain clinical datasets. To reproduce results:
- Obtain access to **MIMIC-IV** and/or **eICU** following their credentialing.
- Set env variables (no PHI in repo):
  - `SENTINEL_DATA=path/to/data`
  - `SENTINEL_MODELS=path/to/models`
See `configs/default.yaml` and `api/utils/paths.py`.
