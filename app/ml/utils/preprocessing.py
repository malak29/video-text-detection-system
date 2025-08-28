import cv2
import numpy as np
from typing import Generator, Tuple, Dict, Any, List, AsyncGenerator
import asyncio
import aiofiles
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self):
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            return {
                'fps': fps,
                'frame_count': frame_count,
                'width': width,
                'height': height,
                'duration': duration,
                'format': Path(video_path).suffix.lower()
            }
            
        except Exception as e:
            logger.error(f"Failed to get video info: {e}")
            return {}
    
    def extract_frames_at_fps(self, video_path: str, target_fps: int = 10) -> Generator[Tuple[np.ndarray, int, float], None, None]:
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            
            source_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = max(1, int(source_fps / target_fps))
            
            frame_number = 0
            extracted_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_number % frame_interval == 0:
                    timestamp = frame_number / source_fps
                    yield frame, extracted_count, timestamp
                    extracted_count += 1
                
                frame_number += 1
            
            cap.release()
            
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            return
    
    async def extract_frames_generator(self, video_path: str, target_fps: int = 10) -> AsyncGenerator[Tuple[np.ndarray, int, float], None]:
        loop = asyncio.get_event_loop()
        
        def sync_generator():
            return self.extract_frames_at_fps(video_path, target_fps)
        
        generator = await loop.run_in_executor(None, sync_generator)
        
        for frame_data in generator:
            yield frame_data
            await asyncio.sleep(0)  # Allow other coroutines to run
    
    def extract_single_frame(self, video_path: str, frame_number: int) -> Optional[np.ndarray]:
        try:
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            ret, frame = cap.read()
            cap.release()
            
            return frame if ret else None
            
        except Exception as e:
            logger.error(f"Single frame extraction failed: {e}")
            return None

class ImageProcessor:
    @staticmethod
    def resize_with_aspect_ratio(image: np.ndarray, target_size: int = 640) -> Tuple[np.ndarray, float]:
        height, width = image.shape[:2]
        scale = target_size / max(height, width)
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        y_offset = (target_size - new_height) // 2
        x_offset = (target_size - new_width) // 2
        padded[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
        
        return padded, scale
    
    @staticmethod
    def enhance_text_regions(image: np.ndarray) -> np.ndarray:
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            enhanced = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
            
            denoised = cv2.medianBlur(enhanced, 3)
            
            if len(image.shape) == 3:
                enhanced_color = image.copy()
                enhanced_color[:, :, 0] = denoised
                enhanced_color[:, :, 1] = denoised  
                enhanced_color[:, :, 2] = denoised
                return enhanced_color
            else:
                return denoised
                
        except Exception as e:
            logger.error(f"Image enhancement failed: {e}")
            return image
    
    @staticmethod
    def crop_text_region(image: np.ndarray, bbox: List[int], padding: int = 5) -> np.ndarray:
        try:
            x1, y1, x2, y2 = bbox
            height, width = image.shape[:2]
            
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(width, x2 + padding)
            y2 = min(height, y2 + padding)
            
            return image[y1:y2, x1:x2]
            
        except Exception as e:
            logger.error(f"Text region cropping failed: {e}")
            return image
    
    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        return image.astype(np.float32) / 255.0
    
    @staticmethod
    def denormalize_image(image: np.ndarray) -> np.ndarray:
        return (image * 255.0).astype(np.uint8)

class AnnotationProcessor:
    @staticmethod
    def create_probability_map(image_shape: Tuple[int, int], bboxes: List[List[int]]) -> np.ndarray:
        height, width = image_shape
        prob_map = np.zeros((height, width), dtype=np.float32)
        
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            prob_map[y1:y2, x1:x2] = 1.0
        
        return prob_map
    
    @staticmethod
    def create_threshold_map(prob_map: np.ndarray, shrink_ratio: float = 0.4) -> np.ndarray:
        thresh_map = np.zeros_like(prob_map)
        
        contours, _ = cv2.findContours(
            (prob_map * 255).astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        for contour in contours:
            polygon = contour.reshape(-1, 2)
            distance = cv2.pointPolygonTest(contour, tuple(polygon[0]), True)
            
            shrinked_polygon = AnnotationProcessor._shrink_polygon(polygon, shrink_ratio)
            
            cv2.fillPoly(thresh_map, [shrinked_polygon.astype(np.int32)], 1.0)
        
        return thresh_map
    
    @staticmethod
    def _shrink_polygon(polygon: np.ndarray, ratio: float) -> np.ndarray:
        cx = np.mean(polygon[:, 0])
        cy = np.mean(polygon[:, 1])
        
        shrinked = polygon.copy()
        shrinked[:, 0] = cx + (polygon[:, 0] - cx) * (1 - ratio)
        shrinked[:, 1] = cy + (polygon[:, 1] - cy) * (1 - ratio)
        
        return shrinked