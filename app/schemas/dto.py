"""
Pydantic-схемы для запросов и ответов API.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, Any

from pydantic import BaseModel, Field


# --- Источники ---

class SourceOut(BaseModel):
    id: int
    name: str
    source_type: str
    description: str = ""
    version: str = "1.0"
    lang: str = "en"
    created_at: datetime

    class Config:
        from_attributes = True


# --- Компетенции ---

class CompetencyOut(BaseModel):
    id: int
    title: str
    description: str = ""
    competency_type: str
    proficiency_level: str
    lang: str
    normalized_title: str = ""
    source_id: Optional[int] = None

    class Config:
        from_attributes = True


# --- Профессиональные роли ---

class OccupationOut(BaseModel):
    id: int
    title: str
    code: str = ""
    description: str = ""
    lang: str
    source_id: Optional[int] = None

    class Config:
        from_attributes = True


# --- Сопоставление ---

class MatchRequest(BaseModel):
    text_a: str = Field(..., description="Первая компетенция")
    text_b: str = Field(..., description="Вторая компетенция")


class MatchResult(BaseModel):
    text_a: str
    text_b: str
    similarity: float
    match_type: str


class BatchMatchRequest(BaseModel):
    query: str = Field(..., description="Текст компетенции-запроса")
    candidates: list[str] = Field(..., description="Список компетенций-кандидатов")
    top_k: int = 10


# --- Построение профиля ---

class ProfileBuildRequest(BaseModel):
    role_title: str = Field(..., description="Название профессиональной роли")
    role_description: str = Field("", description="Описание роли / требований")
    source_ids: Optional[list[int]] = Field(
        None, description="Ограничить пул компетенций этими источниками"
    )
    top_n: int = 30
    min_similarity: float = 0.45


class ProfileItemOut(BaseModel):
    id: int
    title: str
    competency_type: str
    proficiency_level: str
    similarity: float
    source_refs: Optional[list[Any]] = None
    n_sources: int = 1
    expert_confirmed: bool = False
    expert_note: str = ""

    class Config:
        from_attributes = True


class ProfileOut(BaseModel):
    id: int
    role_title: str
    role_description: str = ""
    created_at: datetime
    quality_metrics: Optional[dict] = None
    items: list[ProfileItemOut] = []

    class Config:
        from_attributes = True


class ProfileItemUpdate(BaseModel):
    """Экспертная корректировка элемента профиля (раздел 12 ТЗ)."""
    expert_confirmed: Optional[bool] = None
    expert_note: Optional[str] = None
    proficiency_level: Optional[str] = None


# --- Извлечение из текста ---

class ExtractRequest(BaseModel):
    text: str = Field(..., description="Свободный текст (вакансия, ОП и т.п.)")
    role_title: str = ""


class ExtractedEntities(BaseModel):
    role_titles: list[str] = []
    work_functions: list[str] = []
    competencies: list[str] = []
