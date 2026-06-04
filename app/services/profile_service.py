"""
profile_service.py
-------------------
Модуль формирования профессионального профиля (раздел 7.5 ТЗ) и
алгоритм агрегирования (раздел 9.4) + алгоритм формирования профиля (9.5).

Алгоритм (раздел 9.5):
  анализ источников → выделение сущностей → нормализация →
  векторизация компетенций → семантическое сопоставление →
  агрегирование → формирование профиля → оценка качества.

Агрегирование (раздел 9.4):
  - объединяет совпадающие и близкие по смыслу компетенции;
  - учитывает степень семантической близости;
  - устраняет дублирование;
  - сохраняет ссылки на исходные источники;
  - формирует итоговую структуру профиля.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from app.services.embedding_service import get_embedding_service
from app.services.normalizer import build_unified_vocabulary, normalize_text
from app.core.enums import ProficiencyLevel

logger = logging.getLogger(__name__)


@dataclass
class CandidateCompetency:
    """Кандидатная компетенция из пула (для построения профиля)."""
    competency_id: int | None
    title: str
    competency_type: str = "skill"
    proficiency_level: str = "unspecified"
    source_id: int | None = None
    source_name: str = ""


@dataclass
class AggregatedItem:
    title: str
    competency_type: str
    proficiency_level: str
    similarity: float
    source_refs: list[dict] = field(default_factory=list)
    competency_id: int | None = None

    @property
    def n_sources(self) -> int:
        return len(set(r.get("source_id") for r in self.source_refs))


@dataclass
class BuiltProfile:
    role_title: str
    role_description: str
    items: list[AggregatedItem]
    quality_metrics: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Построение профиля
# ---------------------------------------------------------------------------

def build_profile(
    role_title: str,
    role_description: str,
    candidates: list[CandidateCompetency],
    top_n: int = 30,
    min_similarity: float = 0.45,
    dedup_threshold: float = 0.85,
) -> BuiltProfile:
    """
    Полный алгоритм формирования профиля (раздел 9.5).
    """
    svc = get_embedding_service()

    if not candidates:
        return BuiltProfile(role_title, role_description, [], {})

    # Запрос — комбинация названия и описания роли
    query = role_title.strip()
    if role_description and role_description.strip():
        query = f"{role_title.strip()}. {role_description.strip()}"

    # Векторизация кандидатов
    cand_titles = [normalize_text(c.title) for c in candidates]
    cand_emb = svc.encode(cand_titles, normalize=True)
    q_emb = svc.encode_one(query, normalize=True)

    # Семантическое сопоставление: близость каждого кандидата к роли
    sims = cand_emb @ q_emb

    # Отбор по порогу
    scored = [
        (i, float(sims[i])) for i in range(len(candidates))
        if sims[i] >= min_similarity
    ]
    scored.sort(key=lambda x: -x[1])

    # Агрегирование (раздел 9.4): дедупликация близких компетенций
    selected_titles = [cand_titles[i] for i, _ in scored]
    selected_ids = [i for i, _ in scored]
    vocab = build_unified_vocabulary(
        selected_titles, similarity_threshold=dedup_threshold, ids=selected_ids
    )

    # Формируем агрегированные элементы по кластерам
    sim_by_index = {i: s for i, s in scored}
    items: list[AggregatedItem] = []
    for cluster in vocab.clusters:
        # Берём члена с наивысшей близостью к роли как «якорь»
        best_member_idx = max(cluster.member_ids, key=lambda idx: sim_by_index.get(idx, 0))
        anchor = candidates[best_member_idx]
        source_refs = []
        for idx in cluster.member_ids:
            c = candidates[idx]
            source_refs.append({
                "source_id": c.source_id,
                "source_name": c.source_name,
                "title": c.title,
                "similarity": round(sim_by_index.get(idx, 0.0), 4),
            })
        # Уровень владения: берём максимальный из членов
        level = _max_level([candidates[idx].proficiency_level
                            for idx in cluster.member_ids])
        items.append(AggregatedItem(
            title=anchor.title,
            competency_type=anchor.competency_type,
            proficiency_level=level,
            similarity=round(sim_by_index.get(best_member_idx, 0.0), 4),
            source_refs=source_refs,
            competency_id=anchor.competency_id,
        ))

    # Сортировка и ограничение top_n
    items.sort(key=lambda x: -x.similarity)
    items = items[:top_n]

    # Оценка качества (раздел 7.6)
    quality = compute_quality_metrics(items, candidates, q_emb, svc)

    return BuiltProfile(role_title, role_description, items, quality)


_LEVEL_ORDER = [
    ProficiencyLevel.UNSPECIFIED.value,
    ProficiencyLevel.AWARENESS.value,
    ProficiencyLevel.BASIC.value,
    ProficiencyLevel.INTERMEDIATE.value,
    ProficiencyLevel.ADVANCED.value,
    ProficiencyLevel.EXPERT.value,
]


def _max_level(levels: list[str]) -> str:
    best = ProficiencyLevel.UNSPECIFIED.value
    best_rank = 0
    for lvl in levels:
        rank = _LEVEL_ORDER.index(lvl) if lvl in _LEVEL_ORDER else 0
        if rank > best_rank:
            best_rank, best = rank, lvl
    return best


# ---------------------------------------------------------------------------
# Оценка качества (раздел 7.6 ТЗ)
# ---------------------------------------------------------------------------

def compute_quality_metrics(
    items: list[AggregatedItem],
    all_candidates: list[CandidateCompetency],
    query_emb: np.ndarray,
    svc,
) -> dict:
    """
    Метрики качества профиля (раздел 7.6, 13):
      - completeness  — полнота: доля охваченных кандидатных компетенций
      - consistency   — непротиворечивость: внутренняя связность профиля
      - applicability — применимость: средняя релевантность роли
      - source_agreement — согласованность источников
    """
    if not items:
        return {
            "completeness": 0.0, "consistency": 0.0,
            "applicability": 0.0, "source_agreement": 0.0,
            "n_items": 0,
        }

    # Применимость — средняя близость к роли
    applicability = float(np.mean([it.similarity for it in items]))

    # Полнота — сколько уникальных кластеров от общего числа кандидатов
    completeness = min(1.0, len(items) / max(1, len(set(
        normalize_text(c.title).lower() for c in all_candidates
    ))))

    # Непротиворечивость — средняя попарная близость компетенций профиля.
    # Высокая связность = компетенции относятся к одной области (профиль целостен).
    titles = [normalize_text(it.title) for it in items]
    emb = svc.encode(titles, normalize=True)
    if len(emb) > 1:
        sim_matrix = emb @ emb.T
        # Среднее по верхнему треугольнику без диагонали
        n = len(emb)
        triu = sim_matrix[np.triu_indices(n, k=1)]
        consistency = float(np.mean(triu))
    else:
        consistency = 1.0

    # Согласованность источников — доля компетенций, подтверждённых >1 источником
    multi_source = sum(1 for it in items if it.n_sources > 1)
    source_agreement = multi_source / len(items)

    return {
        "completeness": round(completeness, 4),
        "consistency": round(consistency, 4),
        "applicability": round(applicability, 4),
        "source_agreement": round(source_agreement, 4),
        "n_items": len(items),
    }
