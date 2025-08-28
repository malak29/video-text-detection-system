import pytest
import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock
import cv2
import numpy as np

from app.ml.inference.pipeline import VideoTextPipeline
from app.services.video_service import VideoService
from app.services.storage_service import StorageService
from app.services.processing_service import ProcessingService
from app.database import SessionLocal, VideoCRUD, ProcessingJobCRUD, VideoCreate, ProcessingJobCreate

@pytest.fixture
def sample_video_file():
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(tmp_file.name, fourcc, 30.0, (640, 480))
        
        for i in range(90):  # 3 seconds of video at 30 FPS
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            if i % 30 < 15:  # Show text for first half of each second
                cv2.putText(frame, "HELLO WORLD", (200, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            
            out.write(frame)
        
        out.release()
        yield tmp_file.name
        
        if os.path.exists(tmp_file.name):
            os.unlink(tmp_file.name)

@pytest.fixture
def db_session():
    session = SessionLocal()
    yield session
    session.close()

class TestVideoProcessingPipeline:
    @pytest.mark.asyncio
    async def test_full_video_processing_pipeline(self, sample_video_file):
        pipeline = VideoTextPipeline(
            detector_path=None,
            recognizer_path=None,
            use_transformer_ocr=False,
            confidence_threshold=0.3,
            batch_size=8
        )
        
        with patch.object(pipeline.detector, 'detect') as mock_detect:
            mock_detect.return_value = [{
                'bbox': [200, 200, 400, 280],
                'confidence': 0.9,
                'polygon': [[200, 200], [400, 200], [400, 280], [200, 280]]
            }]
            
            with patch.object(pipeline.recognizer, 'recognize') as mock_recognize:
                mock_recognize.return_value = {
                    'text': 'HELLO WORLD',
                    'confidence': 0.95
                }
                
                results = await pipeline.process_video(
                    video_path=sample_video_file,
                    output_dir="./temp/test_output"
                )
                
                assert results['status'] == 'success'
                assert 'results' in results
                assert 'summary' in results
                
                summary = results['summary']
                assert summary['total_frames'] > 0
                assert summary['total_detections'] >= 0
                assert 'processing_time_seconds' in summary

class TestVideoServiceIntegration:
    @pytest.mark.asyncio
    async def test_video_metadata_extraction(self, sample_video_file):
        video_service = VideoService()
        
        metadata = await video_service.get_video_metadata(sample_video_file)
        
        assert metadata['width'] == 640
        assert metadata['height'] == 480
        assert metadata['fps'] == 30.0
        assert metadata['duration'] == 3.0
        assert metadata['frame_count'] == 90
    
    @pytest.mark.asyncio
    async def test_thumbnail_generation(self, sample_video_file):
        video_service = VideoService()
        
        thumbnail_path = await video_service.generate_thumbnail(sample_video_file, 1.0)
        
        assert thumbnail_path is not None
        assert os.path.exists(thumbnail_path)
        assert Path(thumbnail_path).suffix == '.jpg'
        
        thumbnail = cv2.imread(thumbnail_path)
        assert thumbnail.shape[:2] == (240, 320)  # Resized dimensions
        
        os.unlink(thumbnail_path)
    
    @pytest.mark.asyncio
    async def test_video_validation(self, sample_video_file):
        video_service = VideoService()
        
        validation_result = await video_service.validate_video_file(sample_video_file)
        
        assert validation_result['is_valid'] is True
        assert 'metadata' in validation_result
        assert validation_result['metadata']['duration'] == 3.0

class TestStorageServiceIntegration:
    @pytest.mark.asyncio
    async def test_local_storage_operations(self, sample_video_file):
        storage_service = StorageService()
        
        filename = "test_video.mp4"
        stored_path = await storage_service.store_video(sample_video_file, filename)
        
        assert stored_path is not None
        assert os.path.exists(stored_path)
        
        retrieved_path = await storage_service.retrieve_video(stored_path, "./temp/retrieved.mp4")
        assert retrieved_path == stored_path
        
        success = await storage_service.delete_video(stored_path)
        assert success is True
        assert not os.path.exists(stored_path)
    
    @pytest.mark.asyncio
    async def test_checksum_calculation(self, sample_video_file):
        storage_service = StorageService()
        
        checksum1 = await storage_service.get_file_checksum(sample_video_file)
        checksum2 = await storage_service.get_file_checksum(sample_video_file)
        
        assert checksum1 is not None
        assert checksum1 == checksum2
        assert len(checksum1) == 32  # MD5 hash length

class TestDatabaseIntegration:
    def test_video_crud_operations(self, db_session):
        video_data = VideoCreate(
            filename="test_video.mp4",
            original_filename="original_test.mp4",
            file_path="/path/to/video.mp4",
            file_size=1024000,
            category="activity"
        )
        
        created_video = VideoCRUD.create(db=db_session, video=video_data, owner_id=1)
        assert created_video.id is not None
        assert created_video.filename == "test_video.mp4"
        
        retrieved_video = VideoCRUD.get(db=db_session, video_id=created_video.id)
        assert retrieved_video.id == created_video.id
        
        success = VideoCRUD.delete(db=db_session, video_id=created_video.id)
        assert success is True
    
    def test_processing_job_crud_operations(self, db_session):
        job_data = ProcessingJobCreate(
            video_id=1,
            celery_task_id="test-task-id-123"
        )
        
        created_job = ProcessingJobCRUD.create(db=db_session, job=job_data)
        assert created_job.id is not None
        assert created_job.celery_task_id == "test-task-id-123"
        
        retrieved_job = ProcessingJobCRUD.get_by_task_id(
            db=db_session, 
            task_id="test-task-id-123"
        )
        assert retrieved_job.id == created_job.id

class TestEndToEndWorkflow:
    @pytest.mark.asyncio
    async def test_complete_video_processing_workflow(self, sample_video_file, db_session):
        video_service = VideoService()
        storage_service = StorageService()
        processing_service = ProcessingService()
        
        video_data = VideoCreate(
            filename="workflow_test.mp4",
            original_filename="workflow_original.mp4", 
            file_path=sample_video_file,
            file_size=os.path.getsize(sample_video_file)
        )
        
        db_video = VideoCRUD.create(db=db_session, video=video_data, owner_id=1)
        
        job_data = ProcessingJobCreate(
            video_id=db_video.id,
            celery_task_id="workflow-test-task"
        )
        
        db_job = ProcessingJobCRUD.create(db=db_session, job=job_data)
        
        pipeline = VideoTextPipeline(
            use_transformer_ocr=False,
            confidence_threshold=0.5,
            batch_size=4
        )
        
        with patch.object(pipeline.detector, 'detect') as mock_detect:
            mock_detect.return_value = [{
                'bbox': [200, 200, 400, 280],
                'confidence': 0.85
            }]
            
            with patch.object(pipeline.recognizer, 'recognize') as mock_recognize:
                mock_recognize.return_value = {
                    'text': 'TEST TEXT',
                    'confidence': 0.9
                }
                
                results = await pipeline.process_video(
                    video_path=sample_video_file,
                    output_dir="./temp/workflow_test"
                )
                
                assert results['status'] == 'success'
                
                csv_export = await processing_service.export_results_csv(results)
                assert 'frame_number,timestamp,text' in csv_export
                
                xml_export = await processing_service.export_results_xml(results)
                assert '<video_text_detection>' in xml_export

class TestErrorHandlingAndRecovery:
    @pytest.mark.asyncio
    async def test_corrupted_video_handling(self):
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            tmp_file.write(b"This is not a valid video file")
            tmp_file.flush()
            
            video_service = VideoService()
            metadata = await video_service.get_video_metadata(tmp_file.name)
            
            assert metadata == {}
            
            os.unlink(tmp_file.name)
    
    @pytest.mark.asyncio
    async def test_missing_file_handling(self):
        video_service = VideoService()
        storage_service = StorageService()
        
        metadata = await video_service.get_video_metadata("nonexistent.mp4")
        assert metadata == {}
        
        retrieved = await storage_service.retrieve_video("nonexistent.mp4", "temp.mp4")
        assert retrieved is None
    
    @pytest.mark.asyncio
    async def test_pipeline_error_recovery(self):
        pipeline = VideoTextPipeline()
        
        with patch.object(pipeline.detector, 'detect', side_effect=Exception("Detector failed")):
            result = pipeline.process_single_frame(np.zeros((480, 640, 3), dtype=np.uint8))
            
            assert 'error' in result
            assert result['detections'] == []

class TestPerformanceAndScaling:
    @pytest.mark.asyncio
    async def test_batch_processing_performance(self, sample_video_file):
        pipeline = VideoTextPipeline(batch_size=16)
        
        start_time = asyncio.get_event_loop().time()
        
        with patch.object(pipeline.detector, 'detect') as mock_detect:
            mock_detect.return_value = []
            
            results = await pipeline.process_video(
                video_path=sample_video_file,
                output_dir="./temp/perf_test"
            )
        
        end_time = asyncio.get_event_loop().time()
        processing_time = end_time - start_time
        
        assert results['status'] == 'success'
        assert processing_time < 10  # Should complete within 10 seconds for small test video
    
    @pytest.mark.asyncio
    async def test_memory_usage_during_processing(self, sample_video_file):
        import psutil
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        pipeline = VideoTextPipeline(batch_size=8)
        
        with patch.object(pipeline.detector, 'detect') as mock_detect:
            mock_detect.return_value = []
            
            await pipeline.process_video(
                video_path=sample_video_file,
                output_dir="./temp/memory_test"
            )
        
        final_memory = process.memory_info().rss
        memory_increase_mb = (final_memory - initial_memory) / 1024 / 1024
        
        assert memory_increase_mb < 500  # Should not increase memory by more than 500MB

@pytest.mark.integration
class TestSystemIntegration:
    @pytest.mark.asyncio
    async def test_health_check_integration(self):
        from app.monitoring.health import health_monitor
        
        health_status = await health_monitor.get_health_status(force_refresh=True)
        
        assert 'healthy' in health_status
        assert 'checks' in health_status
        assert 'database' in health_status['checks']
    
    @pytest.mark.asyncio 
    async def test_metrics_collection_integration(self):
        from app.monitoring.metrics import metrics_collector
        
        metrics_collector.record_video_upload(category="test", success=True)
        metrics_collector.record_text_detection("test_model", count=5)
        
        summary = metrics_collector.get_metrics_summary()
        
        assert 'system' in summary
        assert 'database' in summary