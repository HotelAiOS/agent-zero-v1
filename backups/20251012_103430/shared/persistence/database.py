"""
Database Manager
SQLAlchemy setup i connection management
"""

import os
from typing import Optional, Generator
from contextlib import contextmanager
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base dla wszystkich modeli
Base = declarative_base()


class DatabaseManager:
    """
    Database Manager
    Zarządza połączeniami z bazą danych
    """
    
    def __init__(
        self,
        db_url: Optional[str] = None,
        echo: bool = False
    ):
        """
        Args:
            db_url: Database URL (default: SQLite in-memory)
            echo: Czy logować SQL queries
        """
        if db_url is None:
            # Default: SQLite w pamięci
            db_url = os.getenv('DATABASE_URL', 'sqlite:///agent_zero.db')
        
        self.db_url = db_url
        self.engine = create_engine(
            db_url,
            echo=echo,
            pool_pre_ping=True,
            pool_recycle=3600
        )
        
        # Session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        logger.info(f"DatabaseManager zainicjalizowany: {db_url}")
    
    def create_tables(self):
        """Utwórz wszystkie tabele"""
        Base.metadata.create_all(bind=self.engine)
        logger.info("Tabele utworzone")
    
    def drop_tables(self):
        """Usuń wszystkie tabele (OSTROŻNIE!)"""
        Base.metadata.drop_all(bind=self.engine)
        logger.warning("Tabele usunięte")
    
    def get_session(self) -> Session:
        """Pobierz nową sesję"""
        return self.SessionLocal()
    
    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """
        Context manager dla sesji
        Automatyczny commit/rollback
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            session.close()
    
    def health_check(self) -> bool:
        """Sprawdź połączenie z bazą"""
        try:
            with self.session_scope() as session:
                session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False


# Global instance (singleton pattern)
_db_manager: Optional[DatabaseManager] = None


def init_database(db_url: Optional[str] = None, echo: bool = False) -> DatabaseManager:
    """Inicjalizuj globalny DatabaseManager"""
    global _db_manager
    _db_manager = DatabaseManager(db_url=db_url, echo=echo)
    _db_manager.create_tables()
    return _db_manager


def get_database() -> DatabaseManager:
    """Pobierz globalny DatabaseManager"""
    global _db_manager
    if _db_manager is None:
        _db_manager = init_database()
    return _db_manager


def get_db_session() -> Generator[Session, None, None]:
    """
    Dependency injection dla FastAPI
    Usage: session: Session = Depends(get_db_session)
    """
    db = get_database()
    session = db.get_session()
    try:
        yield session
    finally:
        session.close()
