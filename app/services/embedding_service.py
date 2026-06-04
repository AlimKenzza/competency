"""
embedding_service.py
--------------------
Сервис семантического кодирования и сопоставления компетенций.
Реализует раздел 7.4 (модуль ИИ-семантического сопоставления) и
раздел 9.3 (модель семантического сопоставления) ТЗ.

Модель: Sentence-BERT (paraphrase-multilingual-MiniLM-L12-v2), при наличии
дообученной версии — используется она.

Семантическая близость = косинусная мера между эмбеддингами (раздел 9.3).
Категоризация совпадений по порогам (раздел 7.4):
  full / partial / related / none.

Сервис реализован как singleton с ленивой загрузкой модели, чтобы
не грузить модель при импортах и тестах.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np

from app.core.config import settings
from app.core.enums import MatchType

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Singleton-сервис для кодирования и сопоставления."""

    _instance: Optional["EmbeddingService"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._model = None
        self._model_name = None
        self._dim = None

    @classmethod
    def instance(cls) -> "EmbeddingService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Загрузка модели (ленивая)
    # ------------------------------------------------------------------
    def _ensure_model(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            from sentence_transformers import SentenceTransformer
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_path = settings.finetuned_model_path or settings.base_model
            logger.info(f"Загрузка модели '{model_path}' на {device}...")
            try:
                self._model = SentenceTransformer(model_path, device=device)
            except Exception as e:
                # Фолбэк: если HF недоступен (нет интернета / sandbox),
                # пытаемся загрузить локально собранную мини-модель.
                import os
                fallback = os.environ.get("LOCAL_FALLBACK_MODEL", "")
                if fallback and os.path.exists(fallback):
                    logger.warning(
                        f"Не удалось загрузить '{model_path}' ({e}). "
                        f"Использую локальную модель: {fallback}"
                    )
                    self._model = SentenceTransformer(fallback, device=device)
                    model_path = fallback
                else:
                    raise
            self._model.max_seq_length = settings.max_seq_length
            self._model_name = model_path
            self._dim = self._model.get_sentence_embedding_dimension()
            logger.info(f"Модель загружена. Размерность эмбеддинга: {self._dim}")

    @property
    def model_name(self) -> str:
        return self._model_name or (settings.finetuned_model_path or settings.base_model)

    # ------------------------------------------------------------------
    # Кодирование
    # ------------------------------------------------------------------
    def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        """Преобразует тексты в нормированные эмбеддинги (раздел 9.3)."""
        self._ensure_model()
        if not texts:
            return np.zeros((0, self._dim or 384), dtype=np.float32)
        emb = self._model.encode(
            texts, batch_size=64, convert_to_numpy=True,
            normalize_embeddings=normalize, show_progress_bar=False,
        )
        return emb

    def encode_one(self, text: str, normalize: bool = True) -> np.ndarray:
        return self.encode([text], normalize=normalize)[0]

    # ------------------------------------------------------------------
    # Сопоставление
    # ------------------------------------------------------------------
    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        """Косинусная близость двух эмбеддингов."""
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    @staticmethod
    def classify_match(similarity: float) -> MatchType:
        """
        Категоризация совпадения по порогам (раздел 7.4 ТЗ):
        полное / частичное / близкое по смыслу / отсутствует.
        """
        if similarity >= settings.threshold_full:
            return MatchType.FULL
        if similarity >= settings.threshold_partial:
            return MatchType.PARTIAL
        if similarity >= settings.threshold_related:
            return MatchType.RELATED
        return MatchType.NONE

    def match_pair(self, text_a: str, text_b: str) -> dict:
        """Сопоставляет две компетенции."""
        emb = self.encode([text_a, text_b], normalize=True)
        sim = float(np.dot(emb[0], emb[1]))   # нормированы → dot = cosine
        return {
            "text_a": text_a,
            "text_b": text_b,
            "similarity": sim,
            "match_type": self.classify_match(sim).value,
        }

    def match_batch(self, query: str, candidates: list[str], top_k: int = 10) -> list[dict]:
        """
        Сопоставляет query со списком кандидатов, возвращает top_k
        по убыванию близости.
        """
        if not candidates:
            return []
        q_emb = self.encode_one(query, normalize=True)
        cand_emb = self.encode(candidates, normalize=True)
        sims = cand_emb @ q_emb
        order = np.argsort(-sims)[:top_k]
        return [
            {
                "text": candidates[int(i)],
                "similarity": float(sims[int(i)]),
                "match_type": self.classify_match(float(sims[int(i)])).value,
            }
            for i in order
        ]


def get_embedding_service() -> EmbeddingService:
    return EmbeddingService.instance()
