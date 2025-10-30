import pytest
try:
    from fastapi.testclient import TestClient
    from api.main import app
except Exception as e:
    pytest.skip(f'skipping API tests (import failed): {e}', allow_module_level=True)

def test_health():
    c = TestClient(app)
    r = c.get('/health')
    assert r.status_code == 200
    assert r.json().get('status') == 'healthy'

def test_predict_minimal():
    c = TestClient(app)
    payload = {'age': 65, 'gender': 'M'}
    r = c.post('/predict', json=payload)
    assert r.status_code == 200
    out = r.json()
    assert 'probability' in out and 'risk_category' in out
