import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import torch
import logging
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

from ..models.text_detector import TextDetector
from ..models.text_recognizer import TextRecognizer
from ..utils.preprocessing import VideoProcessor, ImageProcessor

logger = logging.getLogger(__name__)

class VideoTextPipeline:
    def __init__(self, 
                 detector_path: Optional[str] = None,
                 recognizer_path: Optional[str] = None,
                 use_transformer_ocr: bool = True,
                 confidence_threshold: float = 0.5,
                 batch_size: int = 16):
        
        self.detector = TextDetector(detector_path)
        self.recognizer = TextRecognizer(recognizer_path, use_transformer=use_transformer_ocr)
        self.video_processor = VideoProcessor()
        self.image_processor = ImageProcessor()
        
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def process_video(self, video_path: str, output_dir: str, progress_callback=None) -> Dict[str, Any]:
        try:
            start_time = time.time()
            
            video_info = self.video_processor.get_video_info(video_path)
            frames = self.video_processor.extract_frames_generator(video_path)
            
            all_results = []
            frame_count = 0
            total_frames = video_info.get('frame_count', 0)
            
            batch_frames = []
            batch_numbers = []
            
            async for frame_data in frames:
                frame, frame_number, timestamp = frame_data
                batch_frames.append(frame)
                batch_numbers.append((frame_number, timestamp))
                
                if len(batch_frames) >= self.batch_size:
                    batch_results = await self._process_frame_batch(
                        batch_frames, batch_numbers, output_dir
                    )
                    all_results.extend(batch_results)
                    
                    frame_count += len(batch_frames)
                    batch_frames.clear()
                    batch_numbers.clear()
                    
                    if progress_callback:
                        progress = frame_count / total_frames if total_frames > 0 else 0
                        await progress_callback(progress, frame_count, total_frames)
            
            if batch_frames:
                batch_results = await self._process_frame_batch(
                    batch_frames, batch_numbers, output_dir
                )
                all_results.extend(batch_results)
                frame_count += len(batch_frames)
            
            processing_time = time.time() - start_time
            
            summary = self._generate_summary(all_results, processing_time, frame_count)
            
            return {
                'status': 'success',
                'results': all_results,
                'summary': summary,
                'video_info': video_info
            }
            
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'results': []
            }
    
    async def _process_frame_batch(self, frames: List[np.ndarray], frame_info: List[Tuple], output_dir: str) -> List[Dict]:
        loop = asyncio.get_event_loop()
        
        detection_tasks = [
            loop.run_in_executor(self.executor, self.detector.detect, frame, self.confidence_threshold)
            for frame in frames
        ]
        
        batch_detections = await asyncio.gather(*detection_tasks)
        
        results = []
        for i, detections in enumerate(batch_detections):
            frame_number, timestamp = frame_info[i]
            frame = frames[i]
            
            if not detections:
                results.append({
                    'frame_number': frame_number,
                    'timestamp': timestamp,
                    'detections': []
                })
                continue
            
            text_regions = []
            for detection in detections:
                bbox = detection['bbox']
                x1, y1, x2, y2 = bbox
                
                cropped_text = frame[y1:y2, x1:x2]
                if cropped_text.size == 0:
                    continue
                
                text_result = self.recognizer.recognize(cropped_text)
                
                text_regions.append({
                    'bbox': bbox,
                    'text': text_result['text'],
                    'detection_confidence': detection['confidence'],
                    'recognition_confidence': text_result['confidence'],
                    'polygon': detection.get('polygon', [])
                })
            
            results.append({
                'frame_number': frame_number,
                'timestamp': timestamp,
                'detections': text_regions
            })
        
        return results
    
    def process_single_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        try:
            detections = self.detector.detect(frame, self.confidence_threshold)
            
            if not detections:
                return {'detections': []}
            
            text_regions = []
            for detection in detections:
                bbox = detection['bbox']
                x1, y1, x2, y2 = bbox
                
                cropped_text = frame[y1:y2, x1:x2]
                if cropped_text.size == 0:
                    continue
                
                text_result = self.recognizer.recognize(cropped_text)
                
                text_regions.append({
                    'bbox': bbox,
                    'text': text_result['text'],
                    'detection_confidence': detection['confidence'],
                    'recognition_confidence': text_result['confidence']
                })
            
            return {'detections': text_regions}
            
        except Exception as e:
            logger.error(f"Single frame processing failed: {e}")
            return {'detections': [], 'error': str(e)}
    
    def _generate_summary(self, results: List[Dict], processing_time: float, frame_count: int) -> Dict[str, Any]:
        total_detections = sum(len(frame['detections']) for frame in results)
        frames_with_text = sum(1 for frame in results if frame['detections'])
        
        if total_detections > 0:
            avg_detection_confidence = np.mean([
                det['detection_confidence'] 
                for frame in results 
                for det in frame['detections']
            ])
            
            avg_recognition_confidence = np.mean([
                det['recognition_confidence'] 
                for frame in results 
                for det in frame['detections']
            ])
        else:
            avg_detection_confidence = 0.0
            avg_recognition_confidence = 0.0
        
        detected_texts = set()
        for frame in results:
            for det in frame['detections']:
                if det['text'].strip():
                    detected_texts.add(det['text'].strip())
        
        return {
            'total_frames': frame_count,
            'frames_with_text': frames_with_text,
            'total_detections': total_detections,
            'unique_texts': len(detected_texts),
            'detected_texts': list(detected_texts),
            'avg_detection_confidence': float(avg_detection_confidence),
            'avg_recognition_confidence': float(avg_recognition_confidence),
            'processing_time_seconds': processing_time,
            'fps_processed': frame_count / processing_time if processing_time > 0 else 0
        }