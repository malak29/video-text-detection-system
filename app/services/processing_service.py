import asyncio
from typing import Dict, Any, List, Optional
import json
import csv
import io
from pathlib import Path
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from celery import Celery
import logging

from ..config import settings

logger = logging.getLogger(__name__)

class ProcessingService:
    def __init__(self):
        self.celery_app = Celery('video_text_detection')
        self.celery_app.config_from_object({
            'broker_url': settings.celery_broker_url,
            'result_backend': settings.celery_result_backend,
            'task_serializer': 'json',
            'accept_content': ['json'],
            'result_serializer': 'json',
            'timezone': 'UTC',
            'enable_utc': True,
        })
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        try:
            task_result = self.celery_app.AsyncResult(task_id)
            
            result = {
                'status': task_result.status,
                'info': task_result.info if task_result.info else {},
                'ready': task_result.ready(),
                'successful': task_result.successful() if task_result.ready() else None,
                'failed': task_result.failed() if task_result.ready() else None
            }
            
            if task_result.failed():
                result['traceback'] = task_result.traceback
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get task status: {e}")
            return {'status': 'UNKNOWN', 'info': {}, 'error': str(e)}
    
    def cancel_task(self, task_id: str) -> bool:
        try:
            self.celery_app.control.revoke(task_id, terminate=True)
            return True
        except Exception as e:
            logger.error(f"Failed to cancel task: {e}")
            return False
    
    async def export_results_csv(self, results_data: Dict[str, Any]) -> str:
        try:
            output = io.StringIO()
            writer = csv.writer(output)
            
            headers = [
                'frame_number', 'timestamp', 'text', 'bbox_x1', 'bbox_y1', 
                'bbox_x2', 'bbox_y2', 'detection_confidence', 'recognition_confidence'
            ]
            writer.writerow(headers)
            
            for frame_result in results_data.get('results', []):
                frame_number = frame_result.get('frame_number', 0)
                timestamp = frame_result.get('timestamp', 0.0)
                
                for detection in frame_result.get('detections', []):
                    bbox = detection.get('bbox', [0, 0, 0, 0])
                    row = [
                        frame_number,
                        timestamp,
                        detection.get('text', ''),
                        bbox[0], bbox[1], bbox[2], bbox[3],
                        detection.get('detection_confidence', 0.0),
                        detection.get('recognition_confidence', 0.0)
                    ]
                    writer.writerow(row)
            
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"CSV export failed: {e}")
            return ""
    
    async def export_results_xml(self, results_data: Dict[str, Any]) -> str:
        try:
            root = ET.Element("video_text_detection")
            
            summary_elem = ET.SubElement(root, "summary")
            summary_data = results_data.get('summary', {})
            for key, value in summary_data.items():
                elem = ET.SubElement(summary_elem, key)
                elem.text = str(value)
            
            frames_elem = ET.SubElement(root, "frames")
            
            for frame_result in results_data.get('results', []):
                frame_elem = ET.SubElement(frames_elem, "frame")
                frame_elem.set("number", str(frame_result.get('frame_number', 0)))
                frame_elem.set("timestamp", str(frame_result.get('timestamp', 0.0)))
                
                for detection in frame_result.get('detections', []):
                    obj_elem = ET.SubElement(frame_elem, "object")
                    obj_elem.set("transcription", detection.get('text', ''))
                    obj_elem.set("detection_confidence", str(detection.get('detection_confidence', 0.0)))
                    obj_elem.set("recognition_confidence", str(detection.get('recognition_confidence', 0.0)))
                    
                    bbox = detection.get('bbox', [0, 0, 0, 0])
                    
                    point1 = ET.SubElement(obj_elem, "Point")
                    point1.set("x", str(bbox[0]))
                    point1.set("y", str(bbox[1]))
                    
                    point2 = ET.SubElement(obj_elem, "Point")
                    point2.set("x", str(bbox[2]))
                    point2.set("y", str(bbox[1]))
                    
                    point3 = ET.SubElement(obj_elem, "Point")
                    point3.set("x", str(bbox[2]))
                    point3.set("y", str(bbox[3]))
                    
                    point4 = ET.SubElement(obj_elem, "Point") 
                    point4.set("x", str(bbox[0]))
                    point4.set("y", str(bbox[3]))
            
            return ET.tostring(root, encoding='unicode')
            
        except Exception as e:
            logger.error(f"XML export failed: {e}")
            return ""
    
    async def create_annotated_video(self, video_path: str, results_data: Dict[str, Any]) -> Optional[str]:
        try:
            output_dir = Path(settings.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            video_name = Path(video_path).stem
            output_path = output_dir / f"{video_name}_annotated.mp4"
            
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return None
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            detections_by_frame = {}
            for frame_result in results_data.get('results', []):
                frame_num = frame_result.get('frame_number', 0)
                detections_by_frame[frame_num] = frame_result.get('detections', [])
            
            frame_number = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_number in detections_by_frame:
                    frame = self._draw_detections(frame, detections_by_frame[frame_number])
                
                out.write(frame)
                frame_number += 1
            
            cap.release()
            out.release()
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Annotated video creation failed: {e}")
            return None
    
    def _draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        for detection in detections:
            bbox = detection.get('bbox', [])
            text = detection.get('text', '')
            confidence = detection.get('detection_confidence', 0.0)
            
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                label = f"{text} ({confidence:.2f})"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                
                cv2.rectangle(
                    frame, 
                    (x1, y1 - label_size[1] - 10),
                    (x1 + label_size[0], y1),
                    (0, 255, 0), 
                    -1
                )
                
                cv2.putText(
                    frame, 
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1
                )
        
        return frame