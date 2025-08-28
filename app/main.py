from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app
import structlog
import logging
import time
from contextlib import asynccontextmanager

from .config import settings
from .database import init_db, check_db_connection
from .api import api_router
from .api.middleware import (
    LoggingMiddleware, 
    RateLimitMiddleware, 
    CORSMiddleware, 
    SecurityHeadersMiddleware,
    ErrorHandlingMiddleware
)

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logging.basicConfig(level=getattr(logging, settings.log_level.upper()))
logger = structlog.get_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up Video Text Detection API", version=settings.version)
    
    try:
        if not check_db_connection():
            logger.error("Database connection failed")
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        init_db()
        logger.info("Database initialized successfully")
        
        import os
        os.makedirs(settings.temp_dir, exist_ok=True)
        os.makedirs(settings.output_dir, exist_ok=True)
        os.makedirs(settings.model_path, exist_ok=True)
        logger.info("Directory structure created")
        
    except Exception as e:
        logger.error("Startup failed", error=str(e))
        raise
    
    yield
    
    logger.info("Shutting down Video Text Detection API")

app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="Production-ready video text detection and recognition API",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan
)

app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(CORSMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(LoggingMiddleware)

app.include_router(api_router)

if settings.enable_metrics:
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

@app.get("/")
async def root():
    return {
        "service": settings.app_name,
        "version": settings.version,
        "status": "healthy"
    }

@app.get("/health")
async def health_check():
    from .database import db_manager
    
    db_status = db_manager.health_check()
    
    return {
        "status": "healthy" if db_status["status"] == "healthy" else "unhealthy",
        "version": settings.version,
        "database": db_status,
        "timestamp": time.time()
    }

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "status_code": exc.status_code,
            "path": request.url.path
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(
        "Unhandled exception",
        error=str(exc),
        path=request.url.path,
        method=request.method
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "status_code": 500,
            "path": request.url.path
        }
    )