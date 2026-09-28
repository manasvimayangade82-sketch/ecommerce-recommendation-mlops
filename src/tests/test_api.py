from fastapi.testclient import TestClient
from api.app import app

app.router.on_startup.clear()

import api.app as api_app

api_app.model_info = {
    "selected_model": "Test Model"
}

api_app.model = {}

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "E-Commerce Recommendation API"
    assert response.json()["status"] == "running"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True