from fastapi.testclient import TestClient

from src.app.main import app, model

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_model_is_loaded():
    assert model is not None


def test_predict_valid_input():
    data = {
        "tenure_months": 12,
        "monthly_charges": 75.5,
        "support_calls": 2,
        "contract": "Month-to-month",
        "internet_service": "Fiber optic"
    }

    response = client.post("/predict", json=data)

    assert response.status_code == 200
    assert response.json()["prediction"] in [0, 1]


def test_predict_invalid_input():
    data = {
        "tenure_months": -5,
        "monthly_charges": 75.5,
        "support_calls": 2,
        "contract": "Month-to-month",
        "internet_service": "Fiber optic"
    }

    response = client.post("/predict", json=data)

    assert response.status_code == 422


def test_prediction_is_deterministic():
    data = {
        "tenure_months": 12,
        "monthly_charges": 75.5,
        "support_calls": 2,
        "contract": "Month-to-month",
        "internet_service": "Fiber optic"
    }

    first = client.post("/predict", json=data).json()
    second = client.post("/predict", json=data).json()

    assert first == second
