"""
Доменные перечисления системы.
Соответствуют разделу 9 ТЗ (модели представления компетенций и профиля).
"""

from __future__ import annotations

from enum import Enum


class SourceType(str, Enum):
    """Тип источника данных (раздел 8.1 ТЗ)."""
    ESCO = "esco"
    ONET = "onet"
    SFIA = "sfia"                      # только reference / валидация
    PROF_STANDARD = "prof_standard"    # профессиональный стандарт
    FREE_TEXT = "free_text"            # произвольный документ (вакансия, ОП)
    MANUAL = "manual"                  # ручной ввод эксперта


class CompetencyType(str, Enum):
    """Тип компетенции."""
    SKILL = "skill"                    # навык
    KNOWLEDGE = "knowledge"            # знание
    WORK_ACTIVITY = "work_activity"    # трудовое действие
    ABILITY = "ability"                # способность
    UNKNOWN = "unknown"


class ProficiencyLevel(str, Enum):
    """
    Уровень владения компетенцией.
    Унифицированная шкала, к которой приводятся уровни из разных источников
    (SFIA 1-7, ESCO, профстандарты 1-9 квалификационных уровней).
    """
    AWARENESS = "awareness"            # осведомлённость
    BASIC = "basic"                    # базовый
    INTERMEDIATE = "intermediate"      # средний
    ADVANCED = "advanced"              # продвинутый
    EXPERT = "expert"                  # экспертный
    UNSPECIFIED = "unspecified"


class MatchType(str, Enum):
    """
    Тип совпадения компетенций (раздел 7.4 ТЗ):
    полное, частичное, близкое по смыслу, отсутствует.
    """
    FULL = "full"
    PARTIAL = "partial"
    RELATED = "related"
    NONE = "none"


# Маппинг уровней из разных шкал в унифицированную
SFIA_LEVEL_MAP = {
    1: ProficiencyLevel.AWARENESS,
    2: ProficiencyLevel.BASIC,
    3: ProficiencyLevel.INTERMEDIATE,
    4: ProficiencyLevel.INTERMEDIATE,
    5: ProficiencyLevel.ADVANCED,
    6: ProficiencyLevel.ADVANCED,
    7: ProficiencyLevel.EXPERT,
}

# Квалификационные уровни профстандартов (РК/РФ, 1-9)
PROF_STANDARD_LEVEL_MAP = {
    1: ProficiencyLevel.AWARENESS,
    2: ProficiencyLevel.AWARENESS,
    3: ProficiencyLevel.BASIC,
    4: ProficiencyLevel.BASIC,
    5: ProficiencyLevel.INTERMEDIATE,
    6: ProficiencyLevel.INTERMEDIATE,
    7: ProficiencyLevel.ADVANCED,
    8: ProficiencyLevel.EXPERT,
    9: ProficiencyLevel.EXPERT,
}
