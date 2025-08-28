import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch
import torch

from app.ml.models.text_detector import TextDetector, DBNet
from app.ml.models.text_recognizer import TextRecognizer, CRNN
from app.ml.inference.pipeline import VideoTextPipeline

@pytest.fixture
def sample_image():
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

@pytest.fixture
def sample_video_frame():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, "TEST TEXT", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame

class TestTextDetector:
    def test_detector_initialization(self):
        detector = TextDetector()
        assert detector.device in ['cuda', 'cpu']
        assert isinstance(detector.model, DBNet)
    
    def test_detect_with_valid_image(self, sample_image):
        detector = TextDetector()
        
        with patch.object(detector.model, 'forward') as mock_forward:
            mock_forward.return_value = {
                'probability': torch.rand(1, 1, 160, 160),
                'threshold': torch.rand(1, 1, 160, 160)
            }
            
            detections = detector.detect(sample_image, confidence_threshold=0.5)
            assert isinstance(detections, list)
    
    def test_detect_with_invalid_input(self):
        detector = TextDetector()
        
        detections = detector.detect(None)
        assert detections == []
        
        detections = detector.detect(np.array([]))
        assert detections == []
    
    def test_post_processing(self, sample_image):
        detector = TextDetector()
        
        prob_map = np.random.rand(160, 160)
        detections = detector._post_process(prob_map, 640, 480, 0.5)
        
        assert isinstance(detections, list)
        for detection in detections:
            assert 'bbox' in detection
            assert 'confidence' in detection
            assert len(detection['bbox']) == 4

class TestTextRecognizer:
    def test_crnn_recognizer_initialization(self):
        recognizer = TextRecognizer(use_transformer=False)
        assert isinstance(recognizer.model, CRNN)
        assert hasattr(recognizer, 'vocab')
    
    def test_transformer_recognizer_initialization(self):
        recognizer = TextRecognizer(use_transformer=True)
        assert recognizer.use_transformer is True
    
    def test_recognize_single_image(self, sample_image):
        recognizer = TextRecognizer(use_transformer=False)
        
        with patch.object(recognizer.model, 'forward') as mock_forward:
            mock_forward.return_value = torch.rand(1, 10, len(recognizer.vocab))
            
            result = recognizer.recognize(sample_image)
            
            assert isinstance(result, dict)
            assert 'text' in result
            assert 'confidence' in result
            assert isinstance(result['text'], str)
            assert isinstance(result['confidence'], float)
    
    def test_recognize_batch(self, sample_image):
        recognizer = TextRecognizer(use_transformer=False)
        images = [sample_image, sample_image]
        
        with patch.object(recognizer.model, 'forward') as mock_forward:
            mock_forward.return_value = torch.rand(2, 10, len(recognizer.vocab))
            
            results = recognizer.recognize_batch(images)
            
            assert isinstance(results, list)
            assert len(results) == 2
            for result in results:
                assert 'text' in result
                assert 'confidence' in result

class TestVideoTextPipeline:
    @pytest.fixture
    def pipeline(self):
        return VideoTextPipeline(
            detector_path=None,
            recognizer_path=None,
            use_transformer_ocr=False,
            confidence_threshold=0.5
        )
    
    def test_pipeline_initialization(self, pipeline):
        assert pipeline.confidence_threshold == 0.5
        assert pipeline.batch_size == 16
        assert hasattr(pipeline, 'detector')
        assert hasattr(pipeline, 'recognizer')
    
    def test_process_single_frame(self, pipeline, sample_video_frame):
        with patch.object(pipeline.detector, 'detect') as mock_detect:
            mock_detect.return_value = [{
                'bbox': [50, 80, 200, 120],
                'confidence': 0.8
            }]
            
            with patch.object(pipeline.recognizer, 'recognize') as mock_recognize:
                mock_recognize.return_value = {
                    'text': 'TEST TEXT',
                    'confidence': 0.9
                }
                
                result = pipeline.process_single_frame(sample_video_frame)
                
                assert 'detections' in result
                assert len(result['detections']) == 1
                assert result['detections'][0]['text'] == 'TEST TEXT'
    
    def test_process_single_frame_no_detections(self, pipeline, sample_image):
        with patch.object(pipeline.detector, 'detect') as mock_detect:
            mock_detect.return_value = []
            
            result = pipeline.process_single_frame(sample_image)
            
            assert 'detections' in result
            assert len(result['detections']) == 0

class TestModelIntegration:
    def test_detector_recognizer_integration(self, sample_video_frame):
        detector = TextDetector()
        recognizer = TextRecognizer(use_transformer=False)
        
        with patch.object(detector.model, 'forward') as mock_detector_forward:
            mock_detector_forward.return_value = {
                'probability': torch.ones(1, 1, 160, 160) * 0.8,
                'threshold': torch.ones(1, 1, 160, 160) * 0.5
            }
            
            with patch.object(recognizer.model, 'forward') as mock_recognizer_forward:
                mock_recognizer_forward.return_value = torch.rand(1, 10, len(recognizer.vocab))
                
                detections = detector.detect(sample_video_frame, confidence_threshold=0.5)
                
                if detections:
                    bbox = detections[0]['bbox']
                    x1, y1, x2, y2 = bbox
                    
                    if x2 > x1 and y2 > y1:
                        cropped_text = sample_video_frame[y1:y2, x1:x2]
                        recognition_result = recognizer.recognize(cropped_text)
                        
                        assert 'text' in recognition_result
                        assert 'confidence' in recognition_result

@pytest.mark.parametrize("confidence_threshold", [0.3, 0.5, 0.7, 0.9])
def test_confidence_threshold_effects(sample_image, confidence_threshold):
    detector = TextDetector()
    
    with patch.object(detector.model, 'forward') as mock_forward:
        mock_forward.return_value = {
            'probability': torch.rand(1, 1, 160, 160),
            'threshold': torch.rand(1, 1, 160, 160)
        }
        
        detections = detector.detect(sample_image, confidence_threshold=confidence_threshold)
        
        for detection in detections:
            assert detection['confidence'] >= confidence_threshold

@pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
def test_batch_processing(sample_image, batch_size):
    recognizer = TextRecognizer(use_transformer=False)
    images = [sample_image] * batch_size
    
    with patch.object(recognizer.model, 'forward') as mock_forward:
        mock_forward.return_value = torch.rand(batch_size, 10, len(recognizer.vocab))
        
        results = recognizer.recognize_batch(images)
        
        assert len(results) == batch_size
        for result in results:
            assert isinstance(result['text'], str)
            assert isinstance(result['confidence'], float)