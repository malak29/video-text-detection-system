from fastapi import APIRouter
from .endpoints import auth, videos, processing

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(auth.router)
api_router.include_router(videos.router) 
api_router.include_router(processing.router)

__all__ = ["api_router"]