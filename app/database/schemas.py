from pydantic import BaseModel, EmailStr, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class VideoCategory(str, Enum):
    ACTIVITY = "activity"
    DRIVING = "driving"
    GAME = "game"
    SPORTS = "sports"
    STREET_INDOOR = "street_indoor"
    STREET_OUTDOOR = "street_outdoor"
    OTHER = "other"

class UserBase(BaseModel):
    email: EmailStr
    username: str
    is_active: Optional[bool] = True

class UserCreate(UserBase):
    password: str

class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    username: Optional[str] = None
    is_active: Optional[bool] = None

class User(UserBase):
    id: int
    is_superuser: bool
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class VideoBase(BaseModel):
    filename: str
    category: Optional[VideoCategory] = None

class VideoCreate(VideoBase):
    original_filename: str
    file_path: str
    file_size: int

class VideoUpdate(BaseModel):
    category: Optional[VideoCategory] = None
    duration: Optional[float] = None
    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None

class Video(VideoBase):
    id: int
    original_filename: str
    file_size: int
    duration: Optional[float] = None
    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    owner_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class FrameBase(BaseModel):
    frame_number: int
    timestamp: float
    width: int
    height: int

class FrameCreate(FrameBase):
    video_id: int
    file_path: str

class Frame(FrameBase):
    id: int
    video_id: int
    file_path: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class TextDetectionBase(BaseModel):
    text_content: str
    confidence: float
    bbox_x1: int
    bbox_y1: int
    bbox_x2: int
    bbox_y2: int
    language: Optional[str] = None
    category: Optional[str] = None

class TextDetectionCreate(TextDetectionBase):
    frame_id: int
    model_name: str
    model_version: str

class TextDetection(TextDetectionBase):
    id: int
    frame_id: int
    model_name: str
    model_version: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class ProcessingJobBase(BaseModel):
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0

class ProcessingJobCreate(BaseModel):
    video_id: int
    celery_task_id: str

class ProcessingJobUpdate(BaseModel):
    status: Optional[TaskStatus] = None
    progress: Optional[float] = None
    total_frames: Optional[int] = None
    processed_frames: Optional[int] = None
    result_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

class ProcessingJob(ProcessingJobBase):
    id: int
    celery_task_id: str
    video_id: int
    total_frames: Optional[int] = None
    processed_frames: int
    result_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime
    
    class Config:
        from_attributes = True

class ModelVersionBase(BaseModel):
    name: str
    version: str
    model_type: str
    is_active: bool = False

class ModelVersionCreate(ModelVersionBase):
    file_path: str
    config: Optional[Dict[str, Any]] = None

class ModelVersionUpdate(BaseModel):
    is_active: Optional[bool] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None

class ModelVersion(ModelVersionBase):
    id: int
    file_path: str
    config: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class VideoWithDetections(Video):
    frames: List[Frame] = []
    processing_jobs: List[ProcessingJob] = []

class FrameWithDetections(Frame):
    text_detections: List[TextDetection] = []

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None