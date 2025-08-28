import asyncio
import time
import logging
from typing import Dict, Any, List
from datetime import datetime
import redis.asyncio as redis
import httpx
import psutil
from pathlib import Path

from ..config import settings
from ..database import db_manager

logger = logging.getLogger(__name__)

class HealthCheck:
    def __init__(self):
        self.checks = {
            'database': self._check_database,
            'redis': self._check_redis,
            'disk_space': self._check_disk_space,
            'memory': self._check_memory,
            'celery': self._check_celery,
            'model_files': self._check_model_files,
            'external_apis': self._check_external_apis
        }
    
    async def run_all_checks(self) -> Dict[str, Any]:
        start_time = time.time()
        results = {}
        overall_healthy = True
        
        for check_name, check_func in self.checks.items():
            try:
                result = await check_func()
                results[check_name] = result
                
                if not result.get('healthy', False):
                    overall_healthy = False
                    
            except Exception as e:
                logger.error(f"Health check {check_name} failed: {e}")
                results[check_name] = {
                    'healthy': False,
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                }
                overall_healthy = False
        
        duration = time.time() - start_time
        
        return {
            'healthy': overall_healthy,
            'checks': results,
            'duration_ms': round(duration * 1000, 2),
            'timestamp': datetime.utcnow().isoformat(),
            'version': settings.version
        }
    
    async def _check_database(self) -> Dict[str, Any]:
        try:
            db_health = db_manager.health_check()
            
            return {
                'healthy': db_health['status'] == 'healthy',
                'status': db_health['status'],
                'details': db_health,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def _check_redis(self) -> Dict[str, Any]:
        try:
            redis_client = redis.from_url(settings.redis_url)
            
            await redis_client.set('health_check', 'ok', ex=60)
            value = await redis_client.get('health_check')
            await redis_client.close()
            
            if value == b'ok':
                return {
                    'healthy': True,
                    'status': 'connected',
                    'timestamp': datetime.utcnow().isoformat()
                }
            else:
                return {
                    'healthy': False,
                    'status': 'connection_failed',
                    'timestamp': datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def _check_disk_space(self) -> Dict[str, Any]:
        try:
            disk_usage = psutil.disk_usage('/')
            
            free_space_gb = disk_usage.free / (1024 ** 3)
            total_space_gb = disk_usage.total / (1024 ** 3)
            used_percent = (disk_usage.used / disk_usage.total) * 100
            
            healthy = used_percent < 90 and free_space_gb > 1  # At least 1GB free and less than 90% used
            
            return {
                'healthy': healthy,
                'free_gb': round(free_space_gb, 2),
                'total_gb': round(total_space_gb, 2),
                'used_percent': round(used_percent, 2),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def _check_memory(self) -> Dict[str, Any]:
        try:
            memory = psutil.virtual_memory()
            
            available_gb = memory.available / (1024 ** 3)
            total_gb = memory.total / (1024 ** 3)
            used_percent = memory.percent
            
            healthy = used_percent < 90 and available_gb > 0.5  # Less than 90% used and at least 500MB available
            
            return {
                'healthy': healthy,
                'available_gb': round(available_gb, 2),
                'total_gb': round(total_gb, 2),
                'used_percent': round(used_percent, 2),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def _check_celery(self) -> Dict[str, Any]:
        try:
            redis_client = redis.from_url(settings.celery_broker_url)
            
            await redis_client.lpush('celery_health_check', 'ping')
            result = await redis_client.blpop('celery_health_check', timeout=1)
            await redis_client.close()
            
            return {
                'healthy': result is not None,
                'status': 'active' if result else 'inactive',
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def _check_model_files(self) -> Dict[str, Any]:
        try:
            model_path = Path(settings.model_path)
            
            if not model_path.exists():
                return {
                    'healthy': False,
                    'error': 'Model directory does not exist',
                    'timestamp': datetime.utcnow().isoformat()
                }
            
            required_models = ['text_detector.pth', 'text_recognizer.pth']
            missing_models = []
            
            for model_file in required_models:
                if not (model_path / model_file).exists():
                    missing_models.append(model_file)
            
            healthy = len(missing_models) == 0
            
            result = {
                'healthy': healthy,
                'model_path': str(model_path),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            if missing_models:
                result['missing_models'] = missing_models
            
            return result
            
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def _check_external_apis(self) -> Dict[str, Any]:
        try:
            external_checks = []
            
            if settings.s3_bucket_name:
                s3_health = await self._check_s3_connectivity()
                external_checks.append(('s3', s3_health))
            
            all_healthy = all(check[1]['healthy'] for check in external_checks)
            
            return {
                'healthy': all_healthy,
                'external_services': dict(external_checks),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def _check_s3_connectivity(self) -> Dict[str, Any]:
        try:
            import boto3
            from botocore.exceptions import ClientError
            
            s3_client = boto3.client(
                's3',
                aws_access_key_id=settings.aws_access_key_id,
                aws_secret_access_key=settings.aws_secret_access_key,
                region_name=settings.aws_region
            )
            
            s3_client.head_bucket(Bucket=settings.s3_bucket_name)
            
            return {
                'healthy': True,
                'bucket': settings.s3_bucket_name,
                'region': settings.aws_region
            }
            
        except ClientError as e:
            return {
                'healthy': False,
                'error': f"S3 error: {e.response['Error']['Code']}"
            }
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e)
            }

class HealthMonitor:
    def __init__(self):
        self.health_check = HealthCheck()
        self.last_check_time = 0
        self.last_results = None
        self.check_interval = 30
    
    async def get_health_status(self, force_refresh: bool = False) -> Dict[str, Any]:
        current_time = time.time()
        
        if (force_refresh or 
            self.last_results is None or 
            current_time - self.last_check_time > self.check_interval):
            
            self.last_results = await self.health_check.run_all_checks()
            self.last_check_time = current_time
        
        return self.last_results
    
    async def get_readiness_status(self) -> Dict[str, Any]:
        critical_checks = ['database', 'redis', 'disk_space', 'memory']
        
        health_status = await self.get_health_status()
        
        critical_healthy = all(
            health_status['checks'].get(check, {}).get('healthy', False)
            for check in critical_checks
        )
        
        return {
            'ready': critical_healthy,
            'critical_checks': {
                check: health_status['checks'].get(check, {})
                for check in critical_checks
            },
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def get_liveness_status(self) -> Dict[str, Any]:
        basic_checks = ['memory', 'disk_space']
        
        try:
            results = {}
            for check_name in basic_checks:
                check_func = getattr(self.health_check, f'_check_{check_name}')
                results[check_name] = await check_func()
            
            all_healthy = all(result.get('healthy', False) for result in results.values())
            
            return {
                'alive': all_healthy,
                'checks': results,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                'alive': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }

health_monitor = HealthMonitor()