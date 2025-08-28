import boto3
import aiofiles
import aiofiles.os
from pathlib import Path
import shutil
import hashlib
import logging
from typing import Optional, Dict, Any
import asyncio
from datetime import datetime

from ..config import settings

logger = logging.getLogger(__name__)

class StorageService:
    def __init__(self):
        self.use_s3 = bool(settings.s3_bucket_name and settings.aws_access_key_id)
        
        if self.use_s3:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=settings.aws_access_key_id,
                aws_secret_access_key=settings.aws_secret_access_key,
                region_name=settings.aws_region
            )
            self.bucket_name = settings.s3_bucket_name
        
        self.local_storage_path = Path("./uploads")
        self.local_storage_path.mkdir(parents=True, exist_ok=True)
    
    async def store_video(self, temp_path: str, filename: str) -> str:
        try:
            if self.use_s3:
                return await self._store_to_s3(temp_path, filename)
            else:
                return await self._store_locally(temp_path, filename)
        except Exception as e:
            logger.error(f"Video storage failed: {e}")
            raise
    
    async def _store_to_s3(self, temp_path: str, filename: str) -> str:
        try:
            s3_key = f"videos/{datetime.now().strftime('%Y/%m/%d')}/{filename}"
            
            def upload_file():
                self.s3_client.upload_file(temp_path, self.bucket_name, s3_key)
                return f"s3://{self.bucket_name}/{s3_key}"
            
            loop = asyncio.get_event_loop()
            s3_path = await loop.run_in_executor(None, upload_file)
            
            logger.info(f"Video uploaded to S3: {s3_path}")
            return s3_path
            
        except Exception as e:
            logger.error(f"S3 upload failed: {e}")
            raise
    
    async def _store_locally(self, temp_path: str, filename: str) -> str:
        try:
            date_folder = datetime.now().strftime('%Y/%m/%d')
            storage_dir = self.local_storage_path / date_folder
            await aiofiles.os.makedirs(storage_dir, exist_ok=True)
            
            final_path = storage_dir / filename
            
            async with aiofiles.open(temp_path, 'rb') as src:
                async with aiofiles.open(final_path, 'wb') as dst:
                    await dst.write(await src.read())
            
            logger.info(f"Video stored locally: {final_path}")
            return str(final_path)
            
        except Exception as e:
            logger.error(f"Local storage failed: {e}")
            raise
    
    async def retrieve_video(self, storage_path: str, local_path: str) -> Optional[str]:
        try:
            if storage_path.startswith('s3://'):
                return await self._retrieve_from_s3(storage_path, local_path)
            else:
                return storage_path if Path(storage_path).exists() else None
        except Exception as e:
            logger.error(f"Video retrieval failed: {e}")
            return None
    
    async def _retrieve_from_s3(self, s3_path: str, local_path: str) -> Optional[str]:
        try:
            s3_key = s3_path.replace(f"s3://{self.bucket_name}/", "")
            
            def download_file():
                self.s3_client.download_file(self.bucket_name, s3_key, local_path)
                return local_path
            
            loop = asyncio.get_event_loop()
            downloaded_path = await loop.run_in_executor(None, download_file)
            
            logger.info(f"Video downloaded from S3: {downloaded_path}")
            return downloaded_path
            
        except Exception as e:
            logger.error(f"S3 download failed: {e}")
            return None
    
    async def delete_video(self, storage_path: str) -> bool:
        try:
            if storage_path.startswith('s3://'):
                return await self._delete_from_s3(storage_path)
            else:
                return await self._delete_locally(storage_path)
        except Exception as e:
            logger.error(f"Video deletion failed: {e}")
            return False
    
    async def _delete_from_s3(self, s3_path: str) -> bool:
        try:
            s3_key = s3_path.replace(f"s3://{self.bucket_name}/", "")
            
            def delete_file():
                self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, delete_file)
            
            logger.info(f"Video deleted from S3: {s3_path}")
            return True
            
        except Exception as e:
            logger.error(f"S3 deletion failed: {e}")
            return False
    
    async def _delete_locally(self, file_path: str) -> bool:
        try:
            if Path(file_path).exists():
                await aiofiles.os.remove(file_path)
                logger.info(f"Video deleted locally: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Local deletion failed: {e}")
            return False
    
    async def get_file_checksum(self, file_path: str) -> Optional[str]:
        try:
            async with aiofiles.open(file_path, 'rb') as f:
                hash_md5 = hashlib.md5()
                async for chunk in f:
                    hash_md5.update(chunk)
                return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Checksum calculation failed: {e}")
            return None
    
    async def cleanup_temp_files(self, max_age_hours: int = 24) -> Dict[str, int]:
        try:
            temp_path = Path(settings.temp_dir)
            current_time = datetime.now()
            deleted_count = 0
            total_size_freed = 0
            
            for file_path in temp_path.rglob('*'):
                if file_path.is_file():
                    file_age = current_time - datetime.fromtimestamp(file_path.stat().st_mtime)
                    
                    if file_age.total_seconds() > (max_age_hours * 3600):
                        file_size = file_path.stat().st_size
                        await aiofiles.os.remove(file_path)
                        deleted_count += 1
                        total_size_freed += file_size
                        logger.info(f"Deleted temp file: {file_path}")
            
            return {
                'deleted_files': deleted_count,
                'size_freed_mb': total_size_freed / (1024 * 1024)
            }
            
        except Exception as e:
            logger.error(f"Temp cleanup failed: {e}")
            return {'deleted_files': 0, 'size_freed_mb': 0}