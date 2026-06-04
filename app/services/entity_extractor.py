"""
entity_extractor.py
-------------------
Модуль извлечения сущностей (раздел 7.2 ТЗ).

Два режима:
  1) Структурированный импорт — ESCO (CSV) и O*NET (TSV), где роли,
     функции и компетенции уже размечены. Парсится напрямую.
  2) Правила для свободного текста — извлечение компетенций из вакансий,
     образовательных программ, профстандартов по лингвистическим шаблонам.

Извлекаются (раздел 7.2):
  - названия профессиональных ролей,
  - трудовые функции,
  - компетенции,
  - уровни владения.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import pandas as pd

from app.core.enums import (
    CompetencyType, ProficiencyLevel, SourceType,
    SFIA_LEVEL_MAP, PROF_STANDARD_LEVEL_MAP,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Унифицированные структуры
# ---------------------------------------------------------------------------

@dataclass
class ExtractedCompetency:
    title: str
    description: str = ""
    competency_type: str = CompetencyType.SKILL.value
    proficiency_level: str = ProficiencyLevel.UNSPECIFIED.value
    external_id: str = ""
    lang: str = "en"


@dataclass
class ExtractedOccupation:
    title: str
    code: str = ""
    description: str = ""
    external_id: str = ""
    lang: str = "en"
    work_functions: list[str] = field(default_factory=list)
    competencies: list[ExtractedCompetency] = field(default_factory=list)


@dataclass
class ExtractionResult:
    occupations: list[ExtractedOccupation] = field(default_factory=list)
    standalone_competencies: list[ExtractedCompetency] = field(default_factory=list)

    def stats(self) -> dict:
        n_comp = sum(len(o.competencies) for o in self.occupations) + \
                 len(self.standalone_competencies)
        return {
            "n_occupations": len(self.occupations),
            "n_competencies": n_comp,
            "n_work_functions": sum(len(o.work_functions) for o in self.occupations),
        }


# ---------------------------------------------------------------------------
# 1. Структурированный импорт: ESCO
# ---------------------------------------------------------------------------

def extract_from_esco(
    occupations_csv: str,
    skills_csv: str,
    relations_csv: str,
    lang: str = "en",
    it_isco_prefixes: tuple[str, ...] = ("25", "35"),
) -> ExtractionResult:
    """Импорт из ESCO CSV-релиза с фильтрацией по IT-профессиям."""
    occ_df = pd.read_csv(occupations_csv)
    sk_df = pd.read_csv(skills_csv)
    rel_df = pd.read_csv(relations_csv)

    # Фильтр IT
    code_col = "code" if "code" in occ_df.columns else "iscoGroup"
    occ_df[code_col] = occ_df[code_col].astype(str)
    it_mask = occ_df[code_col].str.startswith(it_isco_prefixes)
    occ_df = occ_df[it_mask]

    # Индекс skills
    sk_idx = sk_df.set_index("conceptUri")

    result = ExtractionResult()
    for _, occ in occ_df.iterrows():
        uri = occ["conceptUri"]
        rels = rel_df[rel_df["occupationUri"] == uri]
        comps = []
        for _, rel in rels.iterrows():
            sk_uri = rel["skillUri"]
            if sk_uri not in sk_idx.index:
                continue
            sk = sk_idx.loc[sk_uri]
            comps.append(ExtractedCompetency(
                title=str(sk["preferredLabel"]),
                description=str(sk.get("description", "")),
                competency_type=CompetencyType.SKILL.value,
                external_id=str(sk_uri),
                lang=lang,
            ))
        result.occupations.append(ExtractedOccupation(
            title=str(occ["preferredLabel"]),
            code=str(occ[code_col]),
            description=str(occ.get("description", "")),
            external_id=str(uri),
            lang=lang,
            competencies=comps,
        ))
    logger.info(f"ESCO извлечено: {result.stats()}")
    return result


# ---------------------------------------------------------------------------
# 1b. Структурированный импорт: O*NET
# ---------------------------------------------------------------------------

def extract_from_onet(
    occupation_data_txt: str,
    skills_txt: str | None = None,
    knowledge_txt: str | None = None,
    work_activities_txt: str | None = None,
    soc_prefix: str = "15-1",
    importance_min: float = 3.0,
) -> ExtractionResult:
    """Импорт из O*NET (TSV) с фильтрацией по IT (SOC 15-1xxx)."""
    occ_df = pd.read_csv(occupation_data_txt, sep="\t")
    occ_df["O*NET-SOC Code"] = occ_df["O*NET-SOC Code"].astype(str)
    occ_df = occ_df[occ_df["O*NET-SOC Code"].str.startswith(soc_prefix)]

    def load_elements(path, kind):
        if not path:
            return {}
        df = pd.read_csv(path, sep="\t")
        if "Scale ID" in df.columns:
            df = df[df["Scale ID"] == "IM"]
        if "Data Value" in df.columns:
            df = df[df["Data Value"] >= importance_min]
        out = {}
        for _, r in df.iterrows():
            soc = str(r["O*NET-SOC Code"])
            out.setdefault(soc, []).append(
                (str(r["Element Name"]), kind, str(r["Element ID"]))
            )
        return out

    skills_map = load_elements(skills_txt, CompetencyType.SKILL.value)
    knowledge_map = load_elements(knowledge_txt, CompetencyType.KNOWLEDGE.value)
    wa_map = load_elements(work_activities_txt, CompetencyType.WORK_ACTIVITY.value)

    result = ExtractionResult()
    for _, occ in occ_df.iterrows():
        soc = str(occ["O*NET-SOC Code"])
        comps = []
        for mp in (skills_map, knowledge_map, wa_map):
            for name, kind, eid in mp.get(soc, []):
                comps.append(ExtractedCompetency(
                    title=name, competency_type=kind,
                    external_id=f"onet:{kind}:{eid}", lang="en",
                ))
        result.occupations.append(ExtractedOccupation(
            title=str(occ["Title"]),
            code=soc,
            description=str(occ.get("Description", "")),
            external_id=f"onet:{soc}",
            lang="en",
            competencies=comps,
        ))
    logger.info(f"O*NET извлечено: {result.stats()}")
    return result


# ---------------------------------------------------------------------------
# 2. Извлечение из свободного текста по правилам
# ---------------------------------------------------------------------------

# Маркеры секций требований (рус + англ)
_COMPETENCY_SECTION_MARKERS = [
    r"требовани[яе]", r"обязанност[и]", r"компетенц", r"навык", r"умени[яе]",
    r"знани[яе]", r"трудовые функции", r"квалификац",
    r"requirements", r"responsibilities", r"skills", r"qualifications",
    r"competenc", r"duties", r"what you.ll do", r"must have",
]

# Маркеры пунктов перечня
_BULLET_RE = re.compile(r"^\s*[-•*–—▪]\s*(.+)$", re.MULTILINE)
_NUMBERED_RE = re.compile(r"^\s*\d+[.)]\s*(.+)$", re.MULTILINE)

# Маркеры уровня владения
_LEVEL_PATTERNS = [
    (r"\b(эксперт|expert|senior|ведущий|главн)", ProficiencyLevel.EXPERT),
    (r"\b(продвинут|advanced|углубл)", ProficiencyLevel.ADVANCED),
    (r"\b(средн|intermediate|уверен|middle|опыт)", ProficiencyLevel.INTERMEDIATE),
    (r"\b(базов|basic|начальн|junior|основ)", ProficiencyLevel.BASIC),
    (r"\b(ознаком|awareness|представлени)", ProficiencyLevel.AWARENESS),
]

# Стоп-фразы, не являющиеся компетенциями
_STOP_PHRASES = [
    "опыт работы от", "заработн", "график", "офис", "удалён", "удален",
    "оформление", "соцпакет", "salary", "benefits", "we offer",
]


def detect_level(text: str) -> str:
    low = text.lower()
    for pattern, level in _LEVEL_PATTERNS:
        if re.search(pattern, low):
            return level.value
    return ProficiencyLevel.UNSPECIFIED.value


def _looks_like_competency(line: str) -> bool:
    line_low = line.lower().strip()
    if len(line_low) < 4 or len(line_low) > 250:
        return False
    for stop in _STOP_PHRASES:
        if stop in line_low:
            return False
    return True


def extract_from_free_text(
    text: str,
    role_title: str = "",
    lang: str = "ru",
) -> ExtractionResult:
    """
    Извлекает компетенции из свободного текста (вакансия, ОП, профстандарт)
    по лингвистическим правилам: пункты списков, маркеры секций.
    """
    result = ExtractionResult()

    # Собираем кандидатов из пунктов списков
    candidates: list[str] = []
    for m in _BULLET_RE.finditer(text):
        candidates.append(m.group(1).strip())
    for m in _NUMBERED_RE.finditer(text):
        candidates.append(m.group(1).strip())

    # Если списков нет — режем по строкам и предложениям
    if not candidates:
        for line in text.splitlines():
            line = line.strip()
            if _looks_like_competency(line):
                candidates.append(line)

    # Фильтрация и дедупликация
    seen = set()
    comps: list[ExtractedCompetency] = []
    for c in candidates:
        c = re.sub(r"\s+", " ", c).strip(" .;,")
        key = c.lower()
        if not _looks_like_competency(c) or key in seen:
            continue
        seen.add(key)
        comps.append(ExtractedCompetency(
            title=c,
            competency_type=CompetencyType.SKILL.value,
            proficiency_level=detect_level(c),
            lang=lang,
        ))

    if role_title:
        result.occupations.append(ExtractedOccupation(
            title=role_title, lang=lang, competencies=comps,
        ))
    else:
        result.standalone_competencies = comps

    logger.info(f"Из свободного текста извлечено: {result.stats()}")
    return result
