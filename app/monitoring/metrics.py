from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server
import time
import psutil
import logging
from typing import Dict, Any
from ..database import db_manager

logger = logging.getLogger(__name__)

video_uploads_total = Counter('video_uploads_total', 'Total video uploads', ['category', 'status'])
video_processing_duration = Histogram('video_processing_duration_seconds', 'Video processing duration')
active_processing_jobs = Gauge('active_processing_jobs', 'Number of active processing jobs')
text_detections_total = Counter('text_detections_total', 'Total text detections', ['model_type'])

system_cpu_usage = Gauge('system_cpu_usage_percent', 'System CPU usage percentage')
system_memory_usage = Gauge('system_memory_usage_bytes', 'System memory usage in bytes')
system_memory_total = Gauge('system_memory_total_bytes', 'Total system memory in bytes')
disk_usage = Gauge('disk_usage_bytes', 'Disk usage in bytes', ['path'])
disk_total = Gauge('disk_total_bytes', 'Total disk space in bytes', ['path'])

database_connections = Gauge('database_connections_active', 'Active database connections')
database_query_duration = Histogram('database_query_duration_seconds', 'Database query duration')

model_inference_duration = Histogram('model_inference_duration_seconds', 'Model inference duration', ['model_type'])
model_batch_size = Histogram('model_batch_size', 'Model batch size', ['model_type'])

api_response_time = Histogram('api_response_time_seconds', 'API response time', ['endpoint', 'method'])
api_requests_total = Counter('api_requests_total', 'Total API requests', ['endpoint', 'method', 'status'])

celery_tasks_total = Counter('celery_tasks_total', 'Total Celery tasks', ['task_name', 'status'])
celery_task_duration = Histogram('celery_task_duration_seconds', 'Celery task duration', ['task_name'])

app_info = Info('app_info', 'Application information')

class MetricsCollector:
    def __init__(self):
        self.last_system_update = 0
        self.update_interval = 60
    
    def update_system_metrics(self):
        current_time = time.time()
        
        if current_time - self.last_system_update < self.update_interval:
            return
        
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            system_cpu_usage.set(cpu_percent)
            
            memory = psutil.virtual_memory()
            system_memory_usage.set(memory.used)
            system_memory_total.set(memory.total)
            
            disk = psutil.disk_usage('/')
            disk_usage.labels(path='/').set(disk.used)
            disk_total.labels(path='/').set(disk.total)
            
            self.last_system_update = current_time
            
        except Exception as e:
            logger.error(f"Failed to update system metrics: {e}")
    
    def update_database_metrics(self):
        try:
            db_health = db_manager.health_check()
            
            if db_health['status'] == 'healthy':
                database_connections.set(1)  
            else:
                database_connections.set(0)
                
        except Exception as e:
            logger.error(f"Failed to update database metrics: {e}")
            database_connections.set(0)
    
    def record_video_upload(self, category: str = None, success: bool = True):
        status = 'success' if success else 'failed'
        video_uploads_total.labels(category=category or 'unknown', status=status).inc()
    
    def record_processing_job_started(self):
        active_processing_jobs.inc()
    
    def record_processing_job_completed(self, duration: float):
        active_processing_jobs.dec()
        video_processing_duration.observe(duration)
    
    def record_processing_job_failed(self):
        active_processing_jobs.dec()
    
    def record_text_detection(self, model_type: str, count: int = 1):
        text_detections_total.labels(model_type=model_type).inc(count)
    
    def record_model_inference(self, model_type: str, duration: float, batch_size: int = 1):
        model_inference_duration.labels(model_type=model_type).observe(duration)
        model_batch_size.labels(model_type=model_type).observe(batch_size)
    
    def record_api_request(self, endpoint: str, method: str, status_code: int, duration: float):
        api_requests_total.labels(endpoint=endpoint, method=method, status=str(status_code)).inc()
        api_response_time.labels(endpoint=endpoint, method=method).observe(duration)
    
    def record_celery_task(self, task_name: str, status: str, duration: float = None):
        celery_tasks_total.labels(task_name=task_name, status=status).inc()
        
        if duration is not None:
            celery_task_duration.labels(task_name=task_name).observe(duration)
    
    def set_app_info(self, version: str, environment: str = 'production'):
        app_info.info({
            'version': version,
            'environment': environment,
            'python_version': '3.11',
            'framework': 'FastAPI'
        })
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        try:
            return {
                'system': {
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_percent': psutil.disk_usage('/').percent
                },
                'database': {
                    'status': db_manager.health_check()['status']
                },
                'processing': {
                    'active_jobs': active_processing_jobs._value._value
                }
            }
        except Exception as e:
            logger.error(f"Failed to get metrics summary: {e}")
            return {}

metrics_collector = MetricsCollector()

def start_metrics_server(port: int = 9090):
    try:
        start_http_server(port)
        logger.info(f"Metrics server started on port {port}")
    except Exception as e:
        logger.error(f"Failed to start metrics server: {e}")

def update_periodic_metrics():
    metrics_collector.update_system_metrics()
    metrics_collector.update_database_metrics()