"""
Настройка подключения к базе данных через SQLAlchemy.
Поддерживает SQLite (по умолчанию) и PostgreSQL (через DATABASE_URL).
"""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.core.config import settings


# Для SQLite нужен особый аргумент connect_args
connect_args = {}
if settings.database_url.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(
    settings.database_url,
    connect_args=connect_args,
    echo=False,
    future=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)

Base = declarative_base()


def get_db():
    """FastAPI dependency: выдаёт сессию БД и закрывает её после запроса."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Создаёт все таблицы. Вызывается при старте приложения."""
    # Импортируем модели, чтобы они зарегистрировались в metadata
    from app.models import entities  # noqa: F401
    Base.metadata.create_all(bind=engine)
