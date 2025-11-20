from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_predict_sentiment_positive():
    response = client.post("/predict", json={"text": "I loved this movie, it was fantastic!"})
    print("Positive response:", response.json())
    assert response.status_code == 200
    assert response.json()["sentiment"] == "positive"

def test_predict_sentiment_negative():
    response = client.post("/predict", json={"text": "This was the worst movie ever."})
    print("Negative response:", response.json())
    assert response.status_code == 200
    assert response.json()["sentiment"] == "negative"

def test_empty_text():
    response = client.post("/predict", json={"text": ""})
    print("Empty text response:", response.status_code, response.json() if response.content else None)
    assert response.status_code == 400


if __name__ == "__main__":
    test_predict_sentiment_positive()
    test_predict_sentiment_negative()
    test_empty_text()