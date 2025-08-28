from celery import Celery
from celery.signals import worker_ready, worker_shutdown, task_prerun, task_postrun, task_failure, task_success
from prometheus_client import start_http_server, Counter, Histogram, Gauge
import logging
import time
from typing import Dict, Any

from .config import settings

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
    'task_soft_time_limit': 3000,
    'worker_prefetch_multiplier': 1,
    'worker_max_tasks_per_child': 10,
    'worker_disable_rate_limits': False,
    'task_compression': 'gzip',
    'result_compression': 'gzip',
    'task_routes': {
        'process_video_task': {'queue': 'video_processing'},
        'cleanup_temp_files': {'queue': 'maintenance'},
        'health_check_task': {'queue': 'monitoring'}
    },
    'beat_schedule': {
        'cleanup-temp-files': {
            'task': 'cleanup_temp_files_task',
            'schedule': 3600.0,  # Run every hour
        },
        'health-check': {
            'task': 'health_check_task',
            'schedule': 300.0,  # Run every 5 minutes
        },
    },
})

celery_tasks_total = Counter('celery_tasks_total', 'Total Celery tasks executed', ['task_name', 'status'])
celery_task_duration = Histogram('celery_task_duration_seconds', 'Celery task duration', ['task_name'])
celery_active_tasks = Gauge('celery_active_tasks', 'Active Celery tasks', ['task_name'])
celery_worker_status = Gauge('celery_worker_status', 'Celery worker status', ['worker_name'])

task_start_times = {}

@worker_ready.connect
def worker_ready_handler(sender=None, **kwargs):
    worker_name = sender.hostname
    logger.info(f"Celery worker ready: {worker_name}")
    celery_worker_status.labels(worker_name=worker_name).set(1)
    
    if settings.enable_metrics:
        start_http_server(9091)
        logger.info("Celery worker metrics server started on port 9091")

@worker_shutdown.connect
def worker_shutdown_handler(sender=None, **kwargs):
    worker_name = sender.hostname
    logger.info(f"Celery worker shutting down: {worker_name}")
    celery_worker_status.labels(worker_name=worker_name).set(0)

@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **kwds):
    task_name = task.name
    logger.info(f"Starting task {task_name} with ID {task_id}")
    
    task_start_times[task_id] = time.time()
    celery_active_tasks.labels(task_name=task_name).inc()

@task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, retval=None, state=None, **kwds):
    task_name = task.name
    
    if task_id in task_start_times:
        duration = time.time() - task_start_times[task_id]
        celery_task_duration.labels(task_name=task_name).observe(duration)
        del task_start_times[task_id]
    
    celery_active_tasks.labels(task_name=task_name).dec()
    logger.info(f"Task {task_name} completed with state: {state}")

@task_success.connect
def task_success_handler(sender=None, result=None, **kwargs):
    task_name = sender.name
    celery_tasks_total.labels(task_name=task_name, status='success').inc()
    logger.info(f"Task {task_name} succeeded")

@task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, traceback=None, einfo=None, **kwds):
    task_name = sender.name
    celery_tasks_total.labels(task_name=task_name, status='failure').inc()
    
    if task_id in task_start_times:
        del task_start_times[task_id]
    
    celery_active_tasks.labels(task_name=task_name).dec()
    logger.error(f"Task {task_name} failed: {exception}")

celery_app.autodiscover_tasks(['app.tasks'])

def get_celery_stats() -> Dict[str, Any]:
    try:
        inspect = celery_app.control.inspect()
        
        active_tasks = inspect.active()
        scheduled_tasks = inspect.scheduled()
        reserved_tasks = inspect.reserved()
        
        stats = inspect.stats()
        
        return {
            'active_tasks': active_tasks,
            'scheduled_tasks': scheduled_tasks, 
            'reserved_tasks': reserved_tasks,
            'worker_stats': stats,
            'broker_url': settings.celery_broker_url,
            'result_backend': settings.celery_result_backend
        }
    except Exception as e:
        logger.error(f"Failed to get Celery stats: {e}")
        return {'error': str(e)}

if __name__ == '__main__':
    celery_app.start()