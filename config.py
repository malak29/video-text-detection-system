from pydantic import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    app_name: str = "Video Text Detection API"
    debug: bool = False
    version: str = "1.0.0"
    
    database_url: str
    redis_url: str = "redis://localhost:6379/0"
    
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: str = "us-east-1"
    s3_bucket_name: Optional[str] = None
    
    max_file_size: int = 500 * 1024 * 1024
    max_video_duration: int = 300
    supported_formats: list = ["mp4", "avi", "mov", "mkv"]
    
    model_path: str = "./models"
    temp_dir: str = "./temp"
    output_dir: str = "./output"
    
    celery_broker_url: str
    celery_result_backend: str
    
    log_level: str = "INFO"
    enable_metrics: bool = True
    metrics_port: int = 9090
    
    gpu_enabled: bool = True
    batch_size: int = 32
    confidence_threshold: float = 0.5
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()