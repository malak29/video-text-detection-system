from sqlalchemy import Column, Integer, String, DateTime, Float, Text, Boolean, ForeignKey, JSON, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum
from datetime import datetime

Base = declarative_base()

class TaskStatus(enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class VideoCategory(enum.Enum):
    ACTIVITY = "activity"
    DRIVING = "driving"
    GAME = "game"
    SPORTS = "sports"
    STREET_INDOOR = "street_indoor"
    STREET_OUTDOOR = "street_outdoor"
    OTHER = "other"

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    videos = relationship("Video", back_populates="owner")

class Video(Base):
    __tablename__ = "videos"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    duration = Column(Float, nullable=True)
    fps = Column(Float, nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    category = Column(Enum(VideoCategory), nullable=True)
    
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    owner = relationship("User", back_populates="videos")
    frames = relationship("Frame", back_populates="video", cascade="all, delete-orphan")
    processing_jobs = relationship("ProcessingJob", back_populates="video")

class Frame(Base):
    __tablename__ = "frames"
    
    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False)
    frame_number = Column(Integer, nullable=False)
    timestamp = Column(Float, nullable=False)
    file_path = Column(String, nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    video = relationship("Video", back_populates="frames")
    text_detections = relationship("TextDetection", back_populates="frame", cascade="all, delete-orphan")

class TextDetection(Base):
    __tablename__ = "text_detections"
    
    id = Column(Integer, primary_key=True, index=True)
    frame_id = Column(Integer, ForeignKey("frames.id"), nullable=False)
    text_content = Column(Text, nullable=False)
    confidence = Column(Float, nullable=False)
    
    bbox_x1 = Column(Integer, nullable=False)
    bbox_y1 = Column(Integer, nullable=False)
    bbox_x2 = Column(Integer, nullable=False)
    bbox_y2 = Column(Integer, nullable=False)
    
    language = Column(String, nullable=True)
    category = Column(String, nullable=True)
    
    model_name = Column(String, nullable=False)
    model_version = Column(String, nullable=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    frame = relationship("Frame", back_populates="text_detections")

class ProcessingJob(Base):
    __tablename__ = "processing_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    celery_task_id = Column(String, unique=True, nullable=False, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False)
    status = Column(Enum(TaskStatus), default=TaskStatus.PENDING)
    
    progress = Column(Float, default=0.0)
    total_frames = Column(Integer, nullable=True)
    processed_frames = Column(Integer, default=0)
    
    result_data = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    video = relationship("Video", back_populates="processing_jobs")

class ModelVersion(Base):
    __tablename__ = "model_versions"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    version = Column(String, nullable=False)
    model_type = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    config = Column(JSON, nullable=True)
    
    is_active = Column(Boolean, default=False)
    performance_metrics = Column(JSON, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())