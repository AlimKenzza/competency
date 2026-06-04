"""
normalizer.py
-------------
Модуль нормализации и унификации (раздел 7.3 ТЗ).

Обеспечивает:
  - нормализацию терминологии (регистр, пунктуация, пробелы);
  - приведение описаний к единому формату;
  - устранение дублирующихся формулировок (точные дубли);
  - сопоставление синонимичных/близких формулировок (через эмбеддинги);
  - формирование унифицированного словаря компетенций.

Семантическая дедупликация: компетенции с косинусной близостью выше
порога объединяются в один кластер, представителем которого становится
самая частая или самая короткая формулировка.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field

import numpy as np

from app.services.embedding_service import get_embedding_service

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Текстовая нормализация
# ---------------------------------------------------------------------------

_MULTISPACE = re.compile(r"\s+")
_PUNCT_EDGES = re.compile(r"^[\s\.\,\;\:\-–—•*]+|[\s\.\,\;\:\-–—•*]+$")


def normalize_text(text: str) -> str:
    """Базовая нормализация: unicode, регистр, пробелы, краевые знаки."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("ё", "е").replace("Ё", "Е")
    text = _MULTISPACE.sub(" ", text)
    text = _PUNCT_EDGES.sub("", text)
    return text.strip()


def normalize_key(text: str) -> str:
    """Ключ для точной дедупликации (нижний регистр + нормализация)."""
    return normalize_text(text).lower()


# ---------------------------------------------------------------------------
# Унифицированный словарь
# ---------------------------------------------------------------------------

@dataclass
class CompetencyCluster:
    """Кластер семантически близких компетенций."""
    representative: str
    members: list[str] = field(default_factory=list)
    member_ids: list[int] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.members)


@dataclass
class UnifiedVocabulary:
    clusters: list[CompetencyCluster] = field(default_factory=list)
    # Маппинг: исходный текст → индекс кластера
    text_to_cluster: dict[str, int] = field(default_factory=dict)

    def stats(self) -> dict:
        return {
            "n_clusters": len(self.clusters),
            "n_total_members": sum(c.size for c in self.clusters),
            "reduction": round(
                1 - len(self.clusters) / max(1, sum(c.size for c in self.clusters)), 3
            ),
        }


def deduplicate_exact(titles: list[str]) -> tuple[list[str], dict[str, list[int]]]:
    """
    Точная дедупликация по нормализованному ключу.
    Возвращает уникальные тексты и маппинг ключ → исходные индексы.
    """
    groups: dict[str, list[int]] = {}
    repr_text: dict[str, str] = {}
    for i, t in enumerate(titles):
        key = normalize_key(t)
        if not key:
            continue
        groups.setdefault(key, []).append(i)
        # Представитель — самая короткая форма
        if key not in repr_text or len(t) < len(repr_text[key]):
            repr_text[key] = normalize_text(t)
    unique = [repr_text[k] for k in groups]
    return unique, groups


def build_unified_vocabulary(
    titles: list[str],
    similarity_threshold: float = 0.85,
    ids: list[int] | None = None,
) -> UnifiedVocabulary:
    """
    Формирует унифицированный словарь компетенций.

    Шаг 1: точная дедупликация.
    Шаг 2: семантическая кластеризация по эмбеддингам (greedy clustering):
      проходим по компетенциям, каждую относим к существующему кластеру,
      если близость к его представителю выше порога, иначе создаём новый.
    """
    if ids is None:
        ids = list(range(len(titles)))

    # Шаг 1: точная дедупликация
    normalized = [normalize_text(t) for t in titles]

    # Шаг 2: семантическая кластеризация
    svc = get_embedding_service()
    embeddings = svc.encode(normalized, normalize=True)

    vocab = UnifiedVocabulary()
    cluster_embeddings: list[np.ndarray] = []

    for i, (title, emb) in enumerate(zip(normalized, embeddings)):
        if not title:
            continue
        best_idx, best_sim = -1, -1.0
        for ci, c_emb in enumerate(cluster_embeddings):
            sim = float(np.dot(emb, c_emb))
            if sim > best_sim:
                best_sim, best_idx = sim, ci

        if best_idx >= 0 and best_sim >= similarity_threshold:
            cluster = vocab.clusters[best_idx]
            cluster.members.append(title)
            cluster.member_ids.append(ids[i])
            vocab.text_to_cluster[title] = best_idx
            # Представитель — самый короткий
            if len(title) < len(cluster.representative):
                cluster.representative = title
        else:
            new_cluster = CompetencyCluster(
                representative=title, members=[title], member_ids=[ids[i]]
            )
            vocab.clusters.append(new_cluster)
            cluster_embeddings.append(emb)
            vocab.text_to_cluster[title] = len(vocab.clusters) - 1

    logger.info(f"Унифицированный словарь: {vocab.stats()}")
    return vocab
