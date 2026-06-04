"""
Модели базы данных (SQLAlchemy ORM).

Реализуют структуры из раздела 9 ТЗ:
  - Source            — источник данных
  - Occupation        — профессиональная роль
  - WorkFunction      — трудовая функция
  - Competency        — компетенция (раздел 9.1)
  - OccupationCompetency — связь роль↔компетенция
  - Profile           — профессиональный профиль (раздел 9.2)
  - ProfileItem       — компетенция в составе профиля
  - MatchRecord       — результат семантического сопоставления (раздел 7.4)
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Column, Integer, String, Float, Text, DateTime, ForeignKey, JSON, Boolean
)
from sqlalchemy.orm import relationship

from app.db.session import Base


class Source(Base):
    __tablename__ = "sources"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    source_type = Column(String(50), nullable=False)   # SourceType
    description = Column(Text, default="")
    file_path = Column(String(512), default="")        # путь к исходному документу
    version = Column(String(50), default="1.0")
    lang = Column(String(10), default="en")
    created_at = Column(DateTime, default=datetime.utcnow)

    occupations = relationship("Occupation", back_populates="source",
                               cascade="all, delete-orphan")
    competencies = relationship("Competency", back_populates="source",
                                cascade="all, delete-orphan")


class Occupation(Base):
    __tablename__ = "occupations"

    id = Column(Integer, primary_key=True, index=True)
    source_id = Column(Integer, ForeignKey("sources.id"))
    external_id = Column(String(255), index=True)       # URI/код в источнике
    code = Column(String(100), default="")              # ISCO/SOC код
    title = Column(String(512), nullable=False)
    description = Column(Text, default="")
    lang = Column(String(10), default="en")

    source = relationship("Source", back_populates="occupations")
    work_functions = relationship("WorkFunction", back_populates="occupation",
                                  cascade="all, delete-orphan")
    competency_links = relationship("OccupationCompetency",
                                    back_populates="occupation",
                                    cascade="all, delete-orphan")


class WorkFunction(Base):
    __tablename__ = "work_functions"

    id = Column(Integer, primary_key=True, index=True)
    occupation_id = Column(Integer, ForeignKey("occupations.id"))
    title = Column(String(512), nullable=False)
    description = Column(Text, default="")

    occupation = relationship("Occupation", back_populates="work_functions")


class Competency(Base):
    """Модель компетенции (раздел 9.1 ТЗ)."""
    __tablename__ = "competencies"

    id = Column(Integer, primary_key=True, index=True)
    source_id = Column(Integer, ForeignKey("sources.id"))
    external_id = Column(String(255), index=True)       # идентификатор в источнике
    title = Column(String(512), nullable=False)         # текстовое описание
    description = Column(Text, default="")
    competency_type = Column(String(50), default="skill")  # CompetencyType
    proficiency_level = Column(String(50), default="unspecified")  # ProficiencyLevel
    lang = Column(String(10), default="en")
    # Нормализованная форма (после модуля нормализации)
    normalized_title = Column(String(512), default="")
    # Кэш эмбеддинга (как JSON-список float), опционально
    embedding = Column(JSON, nullable=True)

    source = relationship("Source", back_populates="competencies")
    occupation_links = relationship("OccupationCompetency",
                                    back_populates="competency",
                                    cascade="all, delete-orphan")


class OccupationCompetency(Base):
    """Связь профессиональной роли и компетенции."""
    __tablename__ = "occupation_competencies"

    id = Column(Integer, primary_key=True, index=True)
    occupation_id = Column(Integer, ForeignKey("occupations.id"))
    competency_id = Column(Integer, ForeignKey("competencies.id"))
    relation_type = Column(String(50), default="essential")  # essential/optional
    work_function_id = Column(Integer, ForeignKey("work_functions.id"),
                              nullable=True)

    occupation = relationship("Occupation", back_populates="competency_links")
    competency = relationship("Competency", back_populates="occupation_links")


class Profile(Base):
    """Профессиональный профиль (раздел 9.2 ТЗ)."""
    __tablename__ = "profiles"

    id = Column(Integer, primary_key=True, index=True)
    role_title = Column(String(512), nullable=False)
    role_description = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    # Параметры, с которыми построен профиль
    params = Column(JSON, nullable=True)
    # Метрики качества (раздел 7.6): completeness, consistency, applicability
    quality_metrics = Column(JSON, nullable=True)

    items = relationship("ProfileItem", back_populates="profile",
                         cascade="all, delete-orphan")


class ProfileItem(Base):
    """Компетенция в составе профиля с агрегированной информацией."""
    __tablename__ = "profile_items"

    id = Column(Integer, primary_key=True, index=True)
    profile_id = Column(Integer, ForeignKey("profiles.id"))
    competency_id = Column(Integer, ForeignKey("competencies.id"), nullable=True)
    title = Column(String(512), nullable=False)
    competency_type = Column(String(50), default="skill")
    proficiency_level = Column(String(50), default="unspecified")
    similarity = Column(Float, default=0.0)            # релевантность роли
    # Список источников, подтверждающих компетенцию (агрегирование)
    source_refs = Column(JSON, nullable=True)
    n_sources = Column(Integer, default=1)
    # Подтверждено ли экспертом
    expert_confirmed = Column(Boolean, default=False)
    expert_note = Column(Text, default="")

    profile = relationship("Profile", back_populates="items")


class MatchRecord(Base):
    """Результат семантического сопоставления двух компетенций (раздел 7.4)."""
    __tablename__ = "match_records"

    id = Column(Integer, primary_key=True, index=True)
    competency_a_id = Column(Integer, ForeignKey("competencies.id"))
    competency_b_id = Column(Integer, ForeignKey("competencies.id"))
    similarity = Column(Float, nullable=False)
    match_type = Column(String(50), nullable=False)   # MatchType
    created_at = Column(DateTime, default=datetime.utcnow)
