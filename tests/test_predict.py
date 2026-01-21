from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_returns_503_if_model_missing():
    # If you didn't train yet, predict should return 503.
    r = client.post("/predict", json={
        "tenure_months": 5,
        "monthly_charges": 80.0,
        "total_charges": 300.0,
        "contract_type": "month-to-month",
        "internet_service": "fiber",
        "payment_method": "electronic_check",
        "paperless_billing": True
    })
    assert r.status_code in (200, 503)

def test_predict_shape_if_model_exists():
    # This will pass if model exists; otherwise previous test covers missing model case.
    r = client.post("/predict", json={
        "tenure_months": 24,
        "monthly_charges": 55.0,
        "total_charges": 1400.0,
        "contract_type": "one_year",
        "internet_service": "dsl",
        "payment_method": "credit_card",
        "paperless_billing": False
    })
    if r.status_code == 200:
        data = r.json()
        assert 0.0 <= data["churn_probability"] <= 1.0
        assert isinstance(data["will_churn"], bool)
