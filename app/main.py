"""
main.py
-------
Точка входа FastAPI-приложения системы формирования профессионального
профиля IT-специалиста.

Запуск:
    uvicorn app.main:app --reload
или:
    python -m app.main
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from app.core.config import settings
from app.db.session import init_db
from app.api import sources, matching, profiles, evaluation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("app")

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Интеллектуальная система формирования профессионального "
                "профиля IT-специалиста на основе отраслевых рамок компетенций "
                "и профессиональных стандартов (ESCO, O*NET, профстандарты).",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Роутеры API
app.include_router(sources.router)
app.include_router(matching.router)
app.include_router(profiles.router)
app.include_router(evaluation.router)


@app.on_event("startup")
def on_startup():
    logger.info("Инициализация БД...")
    init_db()
    logger.info(f"Запущено: {settings.app_name} v{settings.app_version}")
    logger.info(f"БД: {settings.database_url}")
    logger.info("Загрузка ML-модели...")
    from app.services.embedding_service import get_embedding_service
    get_embedding_service()._ensure_model()
    logger.info("ML-модель загружена.")


@app.get("/api/health")
def health():
    return {"status": "ok", "app": settings.app_name, "version": settings.app_version}


@app.get("/api/config")
def get_config():
    """Публичные настройки для фронтенда."""
    return {
        "thresholds": {
            "full": settings.threshold_full,
            "partial": settings.threshold_partial,
            "related": settings.threshold_related,
        },
        "model": settings.finetuned_model_path or settings.base_model,
    }


# Статика фронтенда
FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
if (FRONTEND_DIR / "static").exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "static")), name="static")


@app.get("/")
def index():
    index_file = FRONTEND_DIR / "templates" / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "Frontend не найден. API доступно на /docs"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)
