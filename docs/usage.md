# Usage

## Serve API
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

## Predict (example)
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"age": 65, "gender": "M", "hr_mean": 85, "sbp_mean": 120}'
```
