from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
import json

from ...database import get_db, VideoCRUD, ProcessingJobCRUD, ProcessingJob, ProcessingJobCreate, User
from ...config import settings
from ..endpoints.auth import get_current_active_user
from ...services.processing_service import ProcessingService
from ...tasks.video_processing import process_video_task

router = APIRouter(prefix="/processing", tags=["processing"])

processing_service = ProcessingService()

@router.post("/videos/{video_id}/detect", response_model=ProcessingJob)
async def start_text_detection(
    video_id: int,
    confidence_threshold: Optional[float] = None,
    use_transformer: bool = True,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    video = VideoCRUD.get(db=db, video_id=video_id)
    
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    if video.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    existing_job = db.query(ProcessingJob).filter(
        ProcessingJob.video_id == video_id,
        ProcessingJob.status.in_(["pending", "processing"])
    ).first()
    
    if existing_job:
        raise HTTPException(
            status_code=409, 
            detail="Video is already being processed"
        )
    
    task_config = {
        'confidence_threshold': confidence_threshold or settings.confidence_threshold,
        'use_transformer': use_transformer,
        'batch_size': settings.batch_size
    }
    
    task = process_video_task.delay(video_id, task_config)
    
    job_create = ProcessingJobCreate(
        video_id=video_id,
        celery_task_id=task.id
    )
    
    job = ProcessingJobCRUD.create(db=db, job=job_create)
    
    return job

@router.get("/jobs/{job_id}", response_model=ProcessingJob)
async def get_processing_job(
    job_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    video = VideoCRUD.get(db=db, video_id=job.video_id)
    if not video or video.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    return job

@router.get("/jobs/{job_id}/status")
async def get_job_status(
    job_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    video = VideoCRUD.get(db=db, video_id=job.video_id)
    if not video or video.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    task_result = processing_service.get_task_status(job.celery_task_id)
    
    return {
        'job_id': job.id,
        'status': job.status,
        'progress': job.progress,
        'processed_frames': job.processed_frames,
        'total_frames': job.total_frames,
        'celery_status': task_result.get('status'),
        'celery_info': task_result.get('info', {}),
        'started_at': job.started_at,
        'completed_at': job.completed_at,
        'error_message': job.error_message
    }

@router.post("/jobs/{job_id}/cancel")
async def cancel_processing_job(
    job_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    video = VideoCRUD.get(db=db, video_id=job.video_id)
    if not video or video.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    if job.status not in ["pending", "processing"]:
        raise HTTPException(
            status_code=409, 
            detail=f"Cannot cancel job with status: {job.status}"
        )
    
    success = processing_service.cancel_task(job.celery_task_id)
    
    if success:
        job.status = "cancelled"
        db.commit()
        return {"message": "Job cancelled successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to cancel job")

@router.get("/videos/{video_id}/results")
async def get_video_results(
    video_id: int,
    format: str = "json",
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    video = VideoCRUD.get(db=db, video_id=video_id)
    
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    if video.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    completed_job = db.query(ProcessingJob).filter(
        ProcessingJob.video_id == video_id,
        ProcessingJob.status == "completed"
    ).order_by(ProcessingJob.completed_at.desc()).first()
    
    if not completed_job or not completed_job.result_data:
        raise HTTPException(status_code=404, detail="No completed processing results found")
    
    if format == "csv":
        csv_content = await processing_service.export_results_csv(completed_job.result_data)
        return {"format": "csv", "content": csv_content}
    elif format == "xml":
        xml_content = await processing_service.export_results_xml(completed_job.result_data)
        return {"format": "xml", "content": xml_content}
    else:
        return {
            "format": "json",
            "results": completed_job.result_data,
            "summary": completed_job.result_data.get('summary', {})
        }

@router.get("/videos/{video_id}/annotated")
async def get_annotated_video(
    video_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    video = VideoCRUD.get(db=db, video_id=video_id)
    
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    if video.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    completed_job = db.query(ProcessingJob).filter(
        ProcessingJob.video_id == video_id,
        ProcessingJob.status == "completed"
    ).order_by(ProcessingJob.completed_at.desc()).first()
    
    if not completed_job:
        raise HTTPException(status_code=404, detail="No completed processing found")
    
    annotated_video_path = await processing_service.create_annotated_video(
        video.file_path, 
        completed_job.result_data
    )
    
    if not annotated_video_path or not os.path.exists(annotated_video_path):
        raise HTTPException(status_code=404, detail="Annotated video not available")
    
    return FileResponse(
        path=annotated_video_path,
        filename=f"annotated_{video.original_filename}",
        media_type='video/mp4'
    )