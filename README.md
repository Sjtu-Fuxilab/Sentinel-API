# Sentinel-API

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
Sanwal Ahmad Zafar · Shanghai Jiao Tong University · Fuxilab
