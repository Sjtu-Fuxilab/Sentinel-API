# Sentinel-API

<!-- badges: start -->
[![Docker](https://img.shields.io/badge/container-GHCR-blue)](https://ghcr.io/sjtu-fuxilab/sentinel-api)
![CI](https://github.com/Sjtu-Fuxilab/Sentinel-API/actions/workflows/ci.yml/badge.svg?branch=main)
![CodeQL](https://github.com/Sjtu-Fuxilab/Sentinel-API/actions/workflows/codeql.yml/badge.svg?branch=main)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
<!-- badges: end -->


🏥 Machine Learning API for ICU Mortality Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview
Sentinel-API is a REST API for predicting in-hospital mortality among ICU patients (e.g., MI). Models target MIMIC-IV / eICU.

## ✨ Features
- Real-time risk prediction
- 6h/12h/24h/48h windows
- XGBoost + Optuna
- SHAP interpretability
- FastAPI with validation & docs

## 🚀 Quick Start

```bash
git clone https://github.com/Sjtu-Fuxilab/Sentinel-API.git
cd Sentinel-API
pip install -r requirements.txt
```

```bash
python api/main.py
# or:
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Docs: `http://localhost:8000/docs`

## 📁 Structure
```
Sentinel-API/
├── api/
│   ├── main.py
│   ├── models/
│   ├── data/
│   └── utils/
├── notebooks/
├── requirements.txt
└── README.md
```

## 👥 Authors
Sanwal Ahmad Zafar, Assoc. Prof. Wei Qin · · Fuxilab, Shanghai Jiao Tong University, China. 
## 🐳 Container (GHCR)

```bash
docker pull ghcr.io/sjtu-fuxilab/sentinel-api:latest
docker run --rm -p 8000:8000 ghcr.io/sjtu-fuxilab/sentinel-api:latest
```


## Figures & Demo

[▶ sentinel-api-compressed.mp4](docs/videos/sentinel-api-compressed.mp4)
<p><video src="docs/videos/sentinel-api-compressed.mp4" controls width="720"></video></p>

## Reproducibility

- **Exact Python environment:** see [`requirements-lock.txt`](requirements-lock.txt).
- **Container image:** published to GHCR (`ghcr.io/sjtu-fuxilab/sentinel-api`) per release tag.
- **Traceability:** the API exposes `GET /version` returning the `tag` and `git sha`.
- **How to cite:** repository includes a `CITATION.cff`; a DOI will be added after the preprint/publication is available.

