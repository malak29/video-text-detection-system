from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List, Optional
import aiofiles
from pathlib import Path
import uuid
import os
import shutil

from ...database import get_db, VideoCRUD, User, Video, VideoCreate, VideoUpdate, VideoWithDetections
from ...config import settings
from ..endpoints.auth import get_current_active_user
from ...services.video_service import VideoService
from ...services.storage_service import StorageService

router = APIRouter(prefix="/videos", tags=["videos"])

video_service = VideoService()
storage_service = StorageService()

@router.post("/upload", response_model=Video, status_code=status.HTTP_201_CREATED)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    category: Optional[str] = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in [f".{fmt}" for fmt in settings.supported_formats]:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format. Supported: {settings.supported_formats}"
        )
    
    if file.size and file.size > settings.max_file_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {settings.max_file_size} bytes"
        )
    
    try:
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = Path(settings.temp_dir) / unique_filename
        
        os.makedirs(settings.temp_dir, exist_ok=True)
        
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        video_info = await video_service.get_video_metadata(str(file_path))
        
        if video_info.get('duration', 0) > settings.max_video_duration:
            os.remove(file_path)
            raise HTTPException(
                status_code=413,
                detail=f"Video too long. Maximum duration: {settings.max_video_duration} seconds"
            )
        
        final_path = await storage_service.store_video(str(file_path), unique_filename)
        
        video_create = VideoCreate(
            filename=unique_filename,
            original_filename=file.filename,
            file_path=final_path,
            file_size=file.size or len(content),
            category=category
        )
        
        db_video = VideoCRUD.create(db=db, video=video_create, owner_id=current_user.id)
        
        if video_info:
            video_update = VideoUpdate(
                duration=video_info.get('duration'),
                fps=video_info.get('fps'),
                width=video_info.get('width'),
                height=video_info.get('height')
            )
            VideoCRUD.update(db=db, video_id=db_video.id, video_update=video_update)
            db.refresh(db_video)
        
        background_tasks.add_task(os.remove, file_path)
        
        return db_video
        
    except HTTPException:
        raise
    except Exception as e:
        if file_path.exists():
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.get("/", response_model=List[Video])
async def get_videos(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    videos = VideoCRUD.get_by_user(db=db, user_id=current_user.id, skip=skip, limit=limit)
    return videos

@router.get("/{video_id}", response_model=VideoWithDetections)
async def get_video(
    video_id: int,
    include_detections: bool = False,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    video = VideoCRUD.get(db=db, video_id=video_id)
    
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    if video.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    if include_detections:
        return await video_service.get_video_with_detections(video_id, db)
    
    return video

@router.put("/{video_id}", response_model=Video)
async def update_video(
    video_id: int,
    video_update: VideoUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    video = VideoCRUD.get(db=db, video_id=video_id)
    
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    if video.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    updated_video = VideoCRUD.update(db=db, video_id=video_id, video_update=video_update)
    return updated_video

@router.delete("/{video_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_video(
    video_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    video = VideoCRUD.get(db=db, video_id=video_id)
    
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    if video.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    await storage_service.delete_video(video.file_path)
    
    VideoCRUD.delete(db=db, video_id=video_id)

@router.get("/{video_id}/download")
async def download_video(
    video_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    video = VideoCRUD.get(db=db, video_id=video_id)
    
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    if video.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    if not os.path.exists(video.file_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    
    return FileResponse(
        path=video.file_path,
        filename=video.original_filename,
        media_type='application/octet-stream'
    )

@router.get("/{video_id}/thumbnail")
async def get_video_thumbnail(
    video_id: int,
    timestamp: float = 0.0,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    video = VideoCRUD.get(db=db, video_id=video_id)
    
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    if video.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    thumbnail_path = await video_service.generate_thumbnail(video.file_path, timestamp)
    
    if not thumbnail_path or not os.path.exists(thumbnail_path):
        raise HTTPException(status_code=404, detail="Thumbnail generation failed")
    
    return FileResponse(
        path=thumbnail_path,
        media_type='image/jpeg'
    )