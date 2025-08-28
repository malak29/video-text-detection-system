import cv2
import os
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List
import tempfile
import logging
import subprocess

from ..ml.utils.preprocessing import VideoProcessor
from ..database import get_db, VideoCRUD, FrameCRUD, TextDetectionCRUD, VideoWithDetections

logger = logging.getLogger(__name__)

class VideoService:
    def __init__(self):
        self.video_processor = VideoProcessor()
    
    async def get_video_metadata(self, video_path: str) -> Dict[str, Any]:
        try:
            return self.video_processor.get_video_info(video_path)
        except Exception as e:
            logger.error(f"Failed to get video metadata: {e}")
            return {}
    
    async def generate_thumbnail(self, video_path: str, timestamp: float = 0.0) -> Optional[str]:
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return None
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_number = int(timestamp * fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                return None
            
            thumb_dir = Path("./temp/thumbnails")
            thumb_dir.mkdir(parents=True, exist_ok=True)
            
            video_name = Path(video_path).stem
            thumb_path = thumb_dir / f"{video_name}_{timestamp}.jpg"
            
            frame_resized = cv2.resize(frame, (320, 240))
            cv2.imwrite(str(thumb_path), frame_resized)
            
            return str(thumb_path)
            
        except Exception as e:
            logger.error(f"Thumbnail generation failed: {e}")
            return None
    
    async def convert_video_format(self, input_path: str, output_format: str = "mp4") -> Optional[str]:
        try:
            output_path = input_path.replace(Path(input_path).suffix, f".{output_format}")
            
            cmd = [
                'ffmpeg',
                '-i', input_path,
                '-c:v', 'libx264',
                '-c:a', 'aac', 
                '-preset', 'fast',
                '-crf', '23',
                '-y',
                output_path
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                return output_path
            else:
                logger.error(f"FFmpeg conversion failed: {stderr.decode()}")
                return None
                
        except Exception as e:
            logger.error(f"Video conversion failed: {e}")
            return None
    
    async def extract_audio(self, video_path: str) -> Optional[str]:
        try:
            output_path = video_path.replace(Path(video_path).suffix, ".wav")
            
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                '-y',
                output_path
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                return output_path
            else:
                logger.error(f"Audio extraction failed: {stderr.decode()}")
                return None
                
        except Exception as e:
            logger.error(f"Audio extraction failed: {e}")
            return None
    
    async def get_video_with_detections(self, video_id: int, db) -> VideoWithDetections:
        try:
            video = VideoCRUD.get(db=db, video_id=video_id)
            if not video:
                return None
            
            frames = FrameCRUD.get_by_video(db=db, video_id=video_id)
            detections = TextDetectionCRUD.get_by_video(db=db, video_id=video_id)
            
            frames_dict = {frame.id: frame for frame in frames}
            
            for detection in detections:
                frame_id = detection.frame_id
                if frame_id in frames_dict:
                    if not hasattr(frames_dict[frame_id], 'text_detections'):
                        frames_dict[frame_id].text_detections = []
                    frames_dict[frame_id].text_detections.append(detection)
            
            video.frames = list(frames_dict.values())
            return video
            
        except Exception as e:
            logger.error(f"Failed to get video with detections: {e}")
            return None
    
    async def validate_video_file(self, file_path: str) -> Dict[str, Any]:
        try:
            video_info = await self.get_video_metadata(file_path)
            
            validation_result = {
                'is_valid': True,
                'errors': [],
                'warnings': [],
                'metadata': video_info
            }
            
            if not video_info:
                validation_result['is_valid'] = False
                validation_result['errors'].append("Cannot read video file")
                return validation_result
            
            if video_info.get('duration', 0) > 600:  # 10 minutes
                validation_result['warnings'].append("Video is longer than 10 minutes")
            
            if video_info.get('width', 0) > 4096 or video_info.get('height', 0) > 4096:
                validation_result['warnings'].append("Very high resolution video may take longer to process")
            
            if video_info.get('fps', 0) > 60:
                validation_result['warnings'].append("High FPS video detected")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Video validation failed: {e}")
            return {
                'is_valid': False,
                'errors': [f"Validation failed: {str(e)}"],
                'warnings': [],
                'metadata': {}
            }