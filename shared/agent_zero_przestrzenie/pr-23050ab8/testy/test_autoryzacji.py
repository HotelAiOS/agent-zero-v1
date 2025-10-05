import pytest
from fastapi.testclient import TestClient
from endpointy_autoryzacji import app

# Klient testowy
client = TestClient(app)

def test_logowanie_sukces():
    """Test pomyślnego logowania"""
    response = client.post(
        "/logowanie",
        data={"username": "admin", "password": "tajne"}
    )
    assert response.status_code == 200
    assert "access_token" in response.json()

def test_glowny_endpoint():
    """Test głównego endpointu"""
    response = client.get("/")
    assert response.status_code == 200
    dane = response.json()
    assert "wiadomosc" in dane

def test_endpoint_chroniony_bez_tokena():
    """Test chronionego endpointu bez tokena"""
    response = client.get("/chroniony")
    assert response.status_code == 401

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
