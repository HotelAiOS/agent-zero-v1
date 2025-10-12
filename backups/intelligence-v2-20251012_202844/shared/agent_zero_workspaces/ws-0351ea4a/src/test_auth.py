import pytest
from fastapi.testclient import TestClient
from auth_endpoints import app

client = TestClient(app)

def test_login_success():
    response = client.post(
        "/login",
        data={"username": "admin", "password": "secret"}
    )
    assert response.status_code == 200
    assert "access_token" in response.json()

def test_login_failure():
    response = client.post(
        "/login", 
        data={"username": "wrong", "password": "wrong"}
    )
    assert response.status_code == 401

def test_protected_route():
    # First login
    login_response = client.post(
        "/login",
        data={"username": "admin", "password": "secret"}
    )
    token = login_response.json()["access_token"]
    
    # Access protected route
    response = client.get(
        "/protected",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert "message" in response.json()

if __name__ == "__main__":
    pytest.main([__file__])
