from fastapi import Request, Response, HTTPException, status
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
import time
import logging
import structlog
from prometheus_client import Counter, Histogram, Gauge
import redis.asyncio as redis
from typing import Callable
import json

from ..config import settings

logger = structlog.get_logger()

REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_REQUESTS = Gauge('http_active_requests', 'Active HTTP requests')

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        ACTIVE_REQUESTS.inc()
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            REQUEST_DURATION.observe(process_time)
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code
            ).inc()
            
            logger.info(
                "Request processed",
                method=request.method,
                url=str(request.url),
                status_code=response.status_code,
                process_time=process_time,
                client_ip=request.client.host if request.client else None
            )
            
            response.headers["X-Process-Time"] = str(process_time)
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path, 
                status=500
            ).inc()
            
            logger.error(
                "Request failed",
                method=request.method,
                url=str(request.url),
                error=str(e),
                process_time=process_time
            )
            
            raise
        finally:
            ACTIVE_REQUESTS.dec()

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, redis_url: str = None):
        super().__init__(app)
        self.redis_url = redis_url or settings.redis_url
        self.redis_client = None
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self.redis_client:
            self.redis_client = redis.from_url(self.redis_url)
        
        client_ip = request.client.host if request.client else "unknown"
        
        if request.url.path.startswith("/auth"):
            limit = 10  # 10 requests per minute for auth endpoints
            window = 60
        elif request.url.path.startswith("/processing"):
            limit = 5   # 5 processing requests per minute
            window = 60
        else:
            limit = 100  # 100 requests per minute for other endpoints
            window = 60
        
        key = f"rate_limit:{client_ip}:{request.url.path}"
        
        try:
            current_requests = await self.redis_client.get(key)
            
            if current_requests is None:
                await self.redis_client.setex(key, window, 1)
            else:
                current_count = int(current_requests)
                if current_count >= limit:
                    return JSONResponse(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        content={
                            "detail": f"Rate limit exceeded. Max {limit} requests per {window} seconds",
                            "retry_after": window
                        }
                    )
                await self.redis_client.incr(key)
            
            response = await call_next(request)
            return response
            
        except Exception as e:
            logger.error("Rate limiting error", error=str(e))
            response = await call_next(request)
            return response

class CORSMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.method == "OPTIONS":
            return Response(
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                    "Access-Control-Allow-Headers": "Authorization, Content-Type",
                }
            )
        
        response = await call_next(request)
        
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
        
        return response

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        
        return response

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            response = await call_next(request)
            return response
        except HTTPException:
            raise
        except Exception as e:
            logger.error(
                "Unhandled exception",
                error=str(e),
                path=request.url.path,
                method=request.method
            )
            
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "detail": "Internal server error",
                    "error_id": str(time.time())
                }
            )