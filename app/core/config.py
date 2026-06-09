"""
Конфигурация системы формирования профессионального профиля IT-специалиста.

Настройки читаются из переменных окружения (через .env) с разумными
значениями по умолчанию. По умолчанию используется SQLite для лёгкого
старта; для продакшена укажите DATABASE_URL на PostgreSQL.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Общие ---
    app_name: str = "Система формирования профессионального профиля IT-специалиста"
    app_version: str = "1.0.0"
    debug: bool = True

    # --- База данных ---
    # SQLite по умолчанию. Для PostgreSQL:
    #   DATABASE_URL=postgresql+psycopg://user:pass@localhost:5432/competency
    database_url: str = f"sqlite:///{BASE_DIR / 'data' / 'competency.db'}"

    # --- ML-модель ---
    # Базовая многоязычная модель (рус + англ)
    base_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    # Путь к дообученной модели (если есть). Если пусто — берётся base_model.
    finetuned_model_path: str = ""
    max_seq_length: int = 128

    # --- Параметры сопоставления ---
    # Пороги для категоризации совпадений компетенций
    threshold_full: float = 0.85      # полное совпадение
    threshold_partial: float = 0.65   # частичное совпадение
    threshold_related: float = 0.45   # близкое по смыслу
    # ниже threshold_related — несовпадение

    # --- Параметры профиля ---
    profile_top_n: int = 30
    profile_min_similarity: float = 0.45

    # --- Файлы ---
    upload_dir: str = str(BASE_DIR / "data" / "uploads")
    report_dir: str = str(BASE_DIR / "data" / "reports")
    max_upload_mb: int = 50

    # --- Сервер ---
    host: str = "0.0.0.0"
    port: int = 8000


@lru_cache
def get_settings() -> Settings:
    s = get_settings_uncached()
    # Railway выдаёт postgresql://, psycopg v3 требует postgresql+psycopg://
    if s.database_url.startswith("postgresql://"):
        object.__setattr__(s, "database_url", s.database_url.replace("postgresql://", "postgresql+psycopg://", 1))
    # Создаём нужные каталоги
    Path(s.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(s.report_dir).mkdir(parents=True, exist_ok=True)
    # Каталог БД (для SQLite)
    if s.database_url.startswith("sqlite"):
        db_file = s.database_url.replace("sqlite:///", "")
        Path(db_file).parent.mkdir(parents=True, exist_ok=True)
    return s


def get_settings_uncached() -> Settings:
    return Settings()


settings = get_settings()
