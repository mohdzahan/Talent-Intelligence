from fastapi.testclient import TestClient
from api.main import app

# TestClient mimics a live server without actually needing to bind to a port
client = TestClient(app)

def test_embed_success():
    response = client.post("/embed", json={"texts": ["Machine learning is fascinating."]})
    assert response.status_code == 200
    
    data = response.json()
    assert "embeddings" in data
    assert len(data["embeddings"]) == 1
    # all-MiniLM-L6-v2 outputs a 384-dimensional vector. 
    # BGE-M3 will output 1024, so this test will break and require updating in Week 4.
    assert len(data["embeddings"][0]) == 384 

def test_embed_validation_error_empty_list():
    # Pydantic min_length=1 should catch this before the ML model sees it
    response = client.post("/embed", json={"texts": []})
    assert response.status_code == 422 

def test_similarity_identical_strings():
    response = client.post("/similarity", json={
        "text1": "Data engineering pipeline",
        "text2": "Data engineering pipeline"
    })
    assert response.status_code == 200
    
    data = response.json()
    # Identical vectors should have a cosine similarity of exactly 1.0
    # Using round() to handle microscopic floating-point math inaccuracies
    assert round(data["score"], 4) == 1.0000

def test_similarity_opposite_strings():
    # These strings should yield a relatively low similarity score
    response = client.post("/similarity", json={
        "text1": "I love building software systems.",
        "text2": "The weather is terribly rainy today."
    })
    assert response.status_code == 200
    assert response.json()["score"] < 0.3