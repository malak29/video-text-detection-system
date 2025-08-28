from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
import logging
from ..config import settings

logger = logging.getLogger(__name__)

engine = create_engine(
    settings.database_url,
    poolclass=StaticPool,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
    echo=settings.debug,
    pool_pre_ping=True,
    pool_recycle=300
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database error: {e}")
        db.rollback()
        raise
    finally:
        db.close()

def init_db():
    try:
        from .models import Base
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        raise

def check_db_connection():
    try:
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False

class DatabaseManager:
    def __init__(self):
        self.engine = engine
        self.SessionLocal = SessionLocal
    
    def get_session(self) -> Session:
        return SessionLocal()
    
    def create_tables(self):
        from .models import Base
        Base.metadata.create_all(bind=self.engine)
    
    def drop_tables(self):
        from .models import Base
        Base.metadata.drop_all(bind=self.engine)
    
    def health_check(self) -> dict:
        try:
            with self.engine.connect() as conn:
                result = conn.execute("SELECT 1").fetchone()
                return {
                    "status": "healthy",
                    "database": "connected",
                    "result": result[0] if result else None
                }
        except Exception as e:
            return {
                "status": "unhealthy", 
                "database": "disconnected",
                "error": str(e)
            }

db_manager = DatabaseManager()