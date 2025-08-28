import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import tempfile
import os
from unittest.mock import Mock, patch
import io

from app.main import app
from app.database import get_db, Base
from app.database.models import User, Video, ProcessingJob
from app.config import settings

SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

@pytest.fixture
def test_user():
    return {
        "username": "testuser",
        "email": "test@example.com", 
        "password": "testpassword123"
    }

@pytest.fixture
def auth_headers(test_user):
    response = client.post("/api/v1/auth/register", json=test_user)
    assert response.status_code == 201
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}

class TestAuthentication:
    def test_register_user(self, test_user):
        response = client.post("/api/v1/auth/register", json=test_user)
        assert response.status_code == 201
        assert "access_token" in response.json()
        assert response.json()["token_type"] == "bearer"
    
    def test_register_duplicate_email(self, test_user):
        client.post("/api/v1/auth/register", json=test_user)
        
        response = client.post("/api/v1/auth/register", json=test_user)
        assert response.status_code == 400
        assert "already registered" in response.json()["detail"]
    
    def test_login_valid_credentials(self, test_user):
        client.post("/api/v1/auth/register", json=test_user)
        
        response = client.post(
            "/api/v1/auth/login",
            data={"username": test_user["username"], "password": test_user["password"]}
        )
        assert response.status_code == 200
        assert "access_token" in response.json()
    
    def test_login_invalid_credentials(self):
        response = client.post(
            "/api/v1/auth/login",
            data={"username": "nonexistent", "password": "wrongpassword"}
        )
        assert response.status_code == 401
    
    def test_get_current_user(self, auth_headers):
        response = client.get("/api/v1/auth/me", headers=auth_headers)
        assert response.status_code == 200
        assert "username" in response.json()
    
    def test_access_protected_route_without_token(self):
        response = client.get("/api/v1/auth/me")
        assert response.status_code == 401

class TestVideoEndpoints:
    def test_upload_video_valid(self, auth_headers):
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            tmp_file.write(b"fake video content")
            tmp_file.flush()
            
            with open(tmp_file.name, 'rb') as f:
                response = client.post(
                    "/api/v1/videos/upload",
                    files={"file": ("test.mp4", f, "video/mp4")},
                    data={"category": "activity"},
                    headers=auth_headers
                )
            
            os.unlink(tmp_file.name)
        
        assert response.status_code == 201
        assert "id" in response.json()
        assert response.json()["original_filename"] == "test.mp4"
    
    def test_upload_video_invalid_format(self, auth_headers):
        fake_file = io.BytesIO(b"fake content")
        response = client.post(
            "/api/v1/videos/upload",
            files={"file": ("test.txt", fake_file, "text/plain")},
            headers=auth_headers
        )
        assert response.status_code == 400
        assert "Unsupported file format" in response.json()["detail"]
    
    def test_upload_video_without_auth(self):
        fake_file = io.BytesIO(b"fake video")
        response = client.post(
            "/api/v1/videos/upload",
            files={"file": ("test.mp4", fake_file, "video/mp4")}
        )
        assert response.status_code == 401
    
    def test_get_videos(self, auth_headers):
        response = client.get("/api/v1/videos/", headers=auth_headers)
        assert response.status_code == 200
        assert isinstance(response.json(), list)
    
    def test_get_video_by_id(self, auth_headers):
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            tmp_file.write(b"fake video content")
            tmp_file.flush()
            
            with open(tmp_file.name, 'rb') as f:
                upload_response = client.post(
                    "/api/v1/videos/upload",
                    files={"file": ("test.mp4", f, "video/mp4")},
                    headers=auth_headers
                )
            
            os.unlink(tmp_file.name)
        
        video_id = upload_response.json()["id"]
        
        response = client.get(f"/api/v1/videos/{video_id}", headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["id"] == video_id
    
    def test_get_nonexistent_video(self, auth_headers):
        response = client.get("/api/v1/videos/99999", headers=auth_headers)
        assert response.status_code == 404
    
    def test_update_video(self, auth_headers):
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            tmp_file.write(b"fake video content")
            tmp_file.flush()
            
            with open(tmp_file.name, 'rb') as f:
                upload_response = client.post(
                    "/api/v1/videos/upload",
                    files={"file": ("test.mp4", f, "video/mp4")},
                    headers=auth_headers
                )
            
            os.unlink(tmp_file.name)
        
        video_id = upload_response.json()["id"]
        
        response = client.put(
            f"/api/v1/videos/{video_id}",
            json={"category": "sports"},
            headers=auth_headers
        )
        assert response.status_code == 200
        assert response.json()["category"] == "sports"
    
    def test_delete_video(self, auth_headers):
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            tmp_file.write(b"fake video content")
            tmp_file.flush()
            
            with open(tmp_file.name, 'rb') as f:
                upload_response = client.post(
                    "/api/v1/videos/upload",
                    files={"file": ("test.mp4", f, "video/mp4")},
                    headers=auth_headers
                )
            
            os.unlink(tmp_file.name)
        
        video_id = upload_response.json()["id"]
        
        response = client.delete(f"/api/v1/videos/{video_id}", headers=auth_headers)
        assert response.status_code == 204

class TestProcessingEndpoints:
    @pytest.fixture
    def sample_video(self, auth_headers):
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            tmp_file.write(b"fake video content")
            tmp_file.flush()
            
            with open(tmp_file.name, 'rb') as f:
                response = client.post(
                    "/api/v1/videos/upload",
                    files={"file": ("test.mp4", f, "video/mp4")},
                    headers=auth_headers
                )
            
            os.unlink(tmp_file.name)
        
        return response.json()["id"]
    
    @patch('app.tasks.video_processing.process_video_task.delay')
    def test_start_text_detection(self, mock_task, sample_video, auth_headers):
        mock_task.return_value = Mock(id="test-task-id")
        
        response = client.post(
            f"/api/v1/processing/videos/{sample_video}/detect",
            params={"confidence_threshold": 0.6, "use_transformer": True},
            headers=auth_headers
        )
        assert response.status_code == 200
        assert "celery_task_id" in response.json()
        mock_task.assert_called_once()
    
    def test_start_detection_nonexistent_video(self, auth_headers):
        response = client.post(
            "/api/v1/processing/videos/99999/detect",
            headers=auth_headers
        )
        assert response.status_code == 404
    
    @patch('app.tasks.video_processing.process_video_task.delay')
    def test_get_job_status(self, mock_task, sample_video, auth_headers):
        mock_task.return_value = Mock(id="test-task-id")
        
        start_response = client.post(
            f"/api/v1/processing/videos/{sample_video}/detect",
            headers=auth_headers
        )
        job_id = start_response.json()["id"]
        
        response = client.get(
            f"/api/v1/processing/jobs/{job_id}/status",
            headers=auth_headers
        )
        assert response.status_code == 200
        assert "status" in response.json()
    
    def test_get_results_no_processing(self, sample_video, auth_headers):
        response = client.get(
            f"/api/v1/processing/videos/{sample_video}/results",
            headers=auth_headers
        )
        assert response.status_code == 404

class TestHealthEndpoints:
    def test_root_endpoint(self):
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["service"] == settings.app_name
    
    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert "status" in response.json()
        assert "version" in response.json()

class TestErrorHandling:
    def test_404_error(self):
        response = client.get("/nonexistent-endpoint")
        assert response.status_code == 404
    
    def test_method_not_allowed(self):
        response = client.put("/")
        assert response.status_code == 405
    
    def test_validation_error(self):
        response = client.post(
            "/api/v1/auth/register",
            json={"username": "", "email": "invalid-email", "password": ""}
        )
        assert response.status_code == 422

class TestRateLimiting:
    def test_auth_rate_limiting(self):
        login_data = {"username": "test", "password": "wrong"}
        
        responses = []
        for _ in range(15):  # Exceed the limit of 10 per minute
            response = client.post("/api/v1/auth/login", data=login_data)
            responses.append(response.status_code)
        
        assert 429 in responses  # Too Many Requests

@pytest.mark.parametrize("endpoint", [
    "/api/v1/videos/",
    "/api/v1/auth/me"
])
def test_protected_endpoints_require_auth(endpoint):
    response = client.get(endpoint)
    assert response.status_code == 401

@pytest.mark.parametrize("method,endpoint", [
    ("GET", "/api/v1/videos/"),
    ("POST", "/api/v1/videos/upload"),
    ("GET", "/health"),
    ("GET", "/")
])
def test_cors_headers(method, endpoint):
    headers = {"Origin": "http://localhost:3000"}
    
    if method == "POST":
        response = client.post(endpoint, headers=headers)
    else:
        response = client.get(endpoint, headers=headers)
    
    assert "Access-Control-Allow-Origin" in response.headers