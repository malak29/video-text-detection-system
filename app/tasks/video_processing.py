from celery import Celery
from celery.signals import task_prerun, task_postrun, task_failure
from sqlalchemy.orm import Session
from typing import Dict, Any
import logging
import os
import tempfile
from datetime import datetime

from ..config import settings
from ..database import SessionLocal, VideoCRUD, ProcessingJobCRUD, ProcessingJobUpdate, FrameCRUD, TextDetectionCRUD, FrameCreate, TextDetectionCreate
from ..ml.inference.pipeline import VideoTextPipeline
from ..services.storage_service import StorageService

logger = logging.getLogger(__name__)

celery_app = Celery('video_text_detection')
celery_app.config_from_object({
    'broker_url': settings.celery_broker_url,
    'result_backend': settings.celery_result_backend,
    'task_serializer': 'json',
    'accept_content': ['json'],
    'result_serializer': 'json',
    'timezone': 'UTC',
    'enable_utc': True,
    'task_track_started': True,
    'task_time_limit': 3600,
    'worker_prefetch_multiplier': 1,
    'worker_max_tasks_per_child': 10
})

storage_service = StorageService()
pipeline = VideoTextPipeline(
    use_transformer_ocr=True,
    confidence_threshold=settings.confidence_threshold,
    batch_size=settings.batch_size
)

@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **kwds):
    logger.info(f"Task {task_id} started: {task.name}")
    
    if task.name == 'app.tasks.video_processing.process_video_task':
        video_id = args[0] if args else None
        if video_id:
            db = SessionLocal()
            try:
                job_update = ProcessingJobUpdate(
                    status='processing',
                    started_at=datetime.utcnow()
                )
                ProcessingJobCRUD.update(db=db, task_id=task_id, job_update=job_update)
            finally:
                db.close()

@task_postrun.connect  
def task_postrun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, retval=None, state=None, **kwds):
    logger.info(f"Task {task_id} finished: {task.name}, state: {state}")

@task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, traceback=None, einfo=None, **kwds):
    logger.error(f"Task {task_id} failed: {exception}")
    
    db = SessionLocal()
    try:
        job_update = ProcessingJobUpdate(
            status='failed',
            completed_at=datetime.utcnow(),
            error_message=str(exception)
        )
        ProcessingJobCRUD.update(db=db, task_id=task_id, job_update=job_update)
    finally:
        db.close()

@celery_app.task(bind=True, name='process_video_task')
def process_video_task(self, video_id: int, config: Dict[str, Any]):
    db = SessionLocal()
    local_video_path = None
    
    try:
        video = VideoCRUD.get(db=db, video_id=video_id)
        if not video:
            raise ValueError(f"Video {video_id} not found")
        
        logger.info(f"Starting video processing for video {video_id}")
        
        if video.file_path.startswith('s3://'):
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                local_video_path = tmp_file.name
                
            retrieved_path = storage_service.retrieve_video(video.file_path, local_video_path)
            if not retrieved_path:
                raise ValueError("Failed to retrieve video from storage")
            
            video_path = retrieved_path
        else:
            video_path = video.file_path
        
        if not os.path.exists(video_path):
            raise ValueError(f"Video file not found: {video_path}")
        
        pipeline.confidence_threshold = config.get('confidence_threshold', 0.5)
        pipeline.batch_size = config.get('batch_size', 16)
        
        async def progress_callback(progress, processed_frames, total_frames):
            job_update = ProcessingJobUpdate(
                progress=progress * 100,
                processed_frames=processed_frames,
                total_frames=total_frames
            )
            ProcessingJobCRUD.update(db=db, task_id=self.request.id, job_update=job_update)
            
            self.update_state(
                state='PROGRESS',
                meta={
                    'progress': progress * 100,
                    'processed_frames': processed_frames,
                    'total_frames': total_frames
                }
            )
        
        import asyncio
        results = asyncio.run(pipeline.process_video(
            video_path=video_path,
            output_dir=settings.output_dir,
            progress_callback=progress_callback
        ))
        
        if results['status'] == 'success':
            save_results_to_database(db, video_id, results)
            
            job_update = ProcessingJobUpdate(
                status='completed',
                completed_at=datetime.utcnow(),
                progress=100.0,
                result_data=results
            )
            ProcessingJobCRUD.update(db=db, task_id=self.request.id, job_update=job_update)
            
            logger.info(f"Video processing completed for video {video_id}")
            
            return {
                'status': 'success',
                'video_id': video_id,
                'results': results['summary'],
                'total_detections': results['summary']['total_detections']
            }
        else:
            raise ValueError(f"Processing failed: {results.get('error', 'Unknown error')}")
    
    except Exception as e:
        logger.error(f"Video processing failed for video {video_id}: {str(e)}")
        
        job_update = ProcessingJobUpdate(
            status='failed',
            completed_at=datetime.utcnow(),
            error_message=str(e)
        )
        ProcessingJobCRUD.update(db=db, task_id=self.request.id, job_update=job_update)
        
        raise e
    
    finally:
        if local_video_path and os.path.exists(local_video_path):
            os.unlink(local_video_path)
        
        db.close()

def save_results_to_database(db: Session, video_id: int, results: Dict[str, Any]):
    try:
        frame_creates = []
        detection_creates = []
        
        for frame_result in results['results']:
            frame_number = frame_result['frame_number']
            timestamp = frame_result['timestamp']
            
            frame_create = FrameCreate(
                video_id=video_id,
                frame_number=frame_number,
                timestamp=timestamp,
                file_path=f"frame_{frame_number:04d}.jpg",
                width=results['video_info'].get('width', 640),
                height=results['video_info'].get('height', 480)
            )
            frame_creates.append(frame_create)
        
        created_frames = FrameCRUD.create_bulk(db=db, frames=frame_creates)
        frame_mapping = {frame.frame_number: frame.id for frame in created_frames}
        
        for frame_result in results['results']:
            frame_number = frame_result['frame_number']
            frame_id = frame_mapping[frame_number]
            
            for detection in frame_result['detections']:
                detection_create = TextDetectionCreate(
                    frame_id=frame_id,
                    text_content=detection['text'],
                    confidence=detection['detection_confidence'],
                    bbox_x1=detection['bbox'][0],
                    bbox_y1=detection['bbox'][1], 
                    bbox_x2=detection['bbox'][2],
                    bbox_y2=detection['bbox'][3],
                    model_name='DBNet-CRNN',
                    model_version='1.0.0'
                )
                detection_creates.append(detection_create)
        
        if detection_creates:
            TextDetectionCRUD.create_bulk(db=db, detections=detection_creates)
        
        logger.info(f"Saved {len(created_frames)} frames and {len(detection_creates)} detections to database")
        
    except Exception as e:
        logger.error(f"Failed to save results to database: {e}")
        raise

@celery_app.task(name='cleanup_temp_files')
def cleanup_temp_files_task():
    try:
        result = storage_service.cleanup_temp_files(max_age_hours=24)
        logger.info(f"Temp cleanup completed: {result}")
        return result
    except Exception as e:
        logger.error(f"Temp cleanup failed: {e}")
        raise

@celery_app.task(name='health_check_task')
def health_check_task():
    return {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'worker_id': os.getenv('HOSTNAME', 'unknown')
    }