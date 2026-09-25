from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_search_validation_error():
    # Validates Pydantic min_length=1
    response = client.post("/search", json={"query": "", "limit": 5})
    assert response.status_code == 422

def test_search_success():
    # Requires Qdrant to be running locally via Docker
    response = client.post("/search", json={"query": "data engineer", "limit": 2})
    assert response.status_code == 200
    data = response.json()
    
    assert "results" in data
    assert len(data["results"]) <= 2
    if len(data["results"]) > 0:
        assert "score" in data["results"][0]
        assert "text" in data["results"][0]