from sqlalchemy.orm import Session, selectinload
from sqlalchemy import and_, desc, func
from typing import List, Optional
from passlib.context import CryptContext
from . import models, schemas

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

class UserCRUD:
    @staticmethod
    def get(db: Session, user_id: int) -> Optional[models.User]:
        return db.query(models.User).filter(models.User.id == user_id).first()
    
    @staticmethod
    def get_by_email(db: Session, email: str) -> Optional[models.User]:
        return db.query(models.User).filter(models.User.email == email).first()
    
    @staticmethod
    def get_by_username(db: Session, username: str) -> Optional[models.User]:
        return db.query(models.User).filter(models.User.username == username).first()
    
    @staticmethod
    def create(db: Session, user: schemas.UserCreate) -> models.User:
        hashed_password = get_password_hash(user.password)
        db_user = models.User(
            email=user.email,
            username=user.username,
            hashed_password=hashed_password
        )
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return db_user
    
    @staticmethod
    def authenticate(db: Session, username: str, password: str) -> Optional[models.User]:
        user = UserCRUD.get_by_username(db, username)
        if not user or not verify_password(password, user.hashed_password):
            return None
        return user

class VideoCRUD:
    @staticmethod
    def create(db: Session, video: schemas.VideoCreate, owner_id: int) -> models.Video:
        db_video = models.Video(**video.dict(), owner_id=owner_id)
        db.add(db_video)
        db.commit()
        db.refresh(db_video)
        return db_video
    
    @staticmethod
    def get(db: Session, video_id: int) -> Optional[models.Video]:
        return db.query(models.Video).filter(models.Video.id == video_id).first()
    
    @staticmethod
    def get_by_user(db: Session, user_id: int, skip: int = 0, limit: int = 100) -> List[models.Video]:
        return db.query(models.Video).filter(models.Video.owner_id == user_id).offset(skip).limit(limit).all()
    
    @staticmethod
    def update(db: Session, video_id: int, video_update: schemas.VideoUpdate) -> Optional[models.Video]:
        db_video = db.query(models.Video).filter(models.Video.id == video_id).first()
        if db_video:
            for key, value in video_update.dict(exclude_unset=True).items():
                setattr(db_video, key, value)
            db.commit()
            db.refresh(db_video)
        return db_video
    
    @staticmethod
    def delete(db: Session, video_id: int) -> bool:
        db_video = db.query(models.Video).filter(models.Video.id == video_id).first()
        if db_video:
            db.delete(db_video)
            db.commit()
            return True
        return False

class FrameCRUD:
    @staticmethod
    def create_bulk(db: Session, frames: List[schemas.FrameCreate]) -> List[models.Frame]:
        db_frames = [models.Frame(**frame.dict()) for frame in frames]
        db.add_all(db_frames)
        db.commit()
        return db_frames
    
    @staticmethod
    def get_by_video(db: Session, video_id: int) -> List[models.Frame]:
        return db.query(models.Frame).filter(models.Frame.video_id == video_id).order_by(models.Frame.frame_number).all()

class TextDetectionCRUD:
    @staticmethod
    def create_bulk(db: Session, detections: List[schemas.TextDetectionCreate]) -> List[models.TextDetection]:
        db_detections = [models.TextDetection(**detection.dict()) for detection in detections]
        db.add_all(db_detections)
        db.commit()
        return db_detections
    
    @staticmethod
    def get_by_frame(db: Session, frame_id: int) -> List[models.TextDetection]:
        return db.query(models.TextDetection).filter(models.TextDetection.frame_id == frame_id).all()
    
    @staticmethod
    def get_by_video(db: Session, video_id: int) -> List[models.TextDetection]:
        return db.query(models.TextDetection).join(models.Frame).filter(models.Frame.video_id == video_id).all()

class ProcessingJobCRUD:
    @staticmethod
    def create(db: Session, job: schemas.ProcessingJobCreate) -> models.ProcessingJob:
        db_job = models.ProcessingJob(**job.dict())
        db.add(db_job)
        db.commit()
        db.refresh(db_job)
        return db_job
    
    @staticmethod
    def get_by_task_id(db: Session, task_id: str) -> Optional[models.ProcessingJob]:
        return db.query(models.ProcessingJob).filter(models.ProcessingJob.celery_task_id == task_id).first()
    
    @staticmethod
    def update(db: Session, task_id: str, job_update: schemas.ProcessingJobUpdate) -> Optional[models.ProcessingJob]:
        db_job = ProcessingJobCRUD.get_by_task_id(db, task_id)
        if db_job:
            for key, value in job_update.dict(exclude_unset=True).items():
                setattr(db_job, key, value)
            db.commit()
            db.refresh(db_job)
        return db_job

class ModelVersionCRUD:
    @staticmethod
    def create(db: Session, model: schemas.ModelVersionCreate) -> models.ModelVersion:
        db_model = models.ModelVersion(**model.dict())
        db.add(db_model)
        db.commit()
        db.refresh(db_model)
        return db_model
    
    @staticmethod
    def get_active(db: Session, model_type: str) -> Optional[models.ModelVersion]:
        return db.query(models.ModelVersion).filter(
            and_(models.ModelVersion.model_type == model_type, models.ModelVersion.is_active == True)
        ).first()
    
    @staticmethod
    def set_active(db: Session, model_id: int) -> Optional[models.ModelVersion]:
        db.query(models.ModelVersion).filter(models.ModelVersion.is_active == True).update({"is_active": False})
        db_model = db.query(models.ModelVersion).filter(models.ModelVersion.id == model_id).first()
        if db_model:
            db_model.is_active = True
            db.commit()
            db.refresh(db_model)
        return db_model