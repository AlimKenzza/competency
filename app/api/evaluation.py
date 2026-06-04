"""
evaluation.py
-------------
API оценки качества ИИ-модели семантического сопоставления компетенций.

Метрики (раздел 13 ТЗ):
  - точность классификации (accuracy)
  - precision / recall / F1 по каждой категории совпадения
  - macro F1
  - матрица ошибок (confusion matrix)
  - распределение значений схожести по категориям
  - скорость инференса (мс/компетенция, компетенций/сек)
"""

from __future__ import annotations

import gc
import time
import logging

import numpy as np
from fastapi import APIRouter

from app.services.embedding_service import get_embedding_service
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/eval", tags=["evaluation"])

# ---------------------------------------------------------------------------
# Три модели для сравнительной оценки
# ---------------------------------------------------------------------------
COMPARISON_MODELS = [
    {
        "id": "minilm-en",
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "label": "all-MiniLM-L6-v2",
        "description": "Компактная англоязычная модель. Высокая скорость, но не поддерживает русский язык.",
        "multilingual": False,
        "size_mb": 80,
    },
    {
        "id": "multilingual-minilm",
        "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "label": "paraphrase-multilingual-MiniLM-L12-v2",
        "description": "Многоязычная компактная модель (50+ языков). Выбрана как основная: баланс точности и скорости.",
        "multilingual": True,
        "size_mb": 470,
        "is_main": True,
    },
    {
        "id": "multilingual-mpnet",
        "name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "label": "paraphrase-multilingual-mpnet-base-v2",
        "description": "Многоязычная модель на базе MPNet. Более высокая точность, но значительно медленнее.",
        "multilingual": True,
        "size_mb": 1100,
    },
]

CATEGORIES = ["full", "partial", "related", "none"]

# ---------------------------------------------------------------------------
# Размеченный тестовый набор: 28 пар компетенций с эталонными метками.
# Охватывает все 4 категории совпадения, рус/англ смешение, разные области IT.
# ---------------------------------------------------------------------------
BENCHMARK_PAIRS = [
    # ── FULL: одно и то же понятие на двух языках / минимальный парафраз ──
    {"a": "Python programming",        "b": "программирование на Python",          "expected": "full"},
    {"a": "machine learning",          "b": "машинное обучение",                   "expected": "full"},
    {"a": "REST API development",      "b": "разработка REST API",                 "expected": "full"},
    {"a": "Docker containerization",   "b": "контейнеризация приложений Docker",   "expected": "full"},
    {"a": "SQL database queries",      "b": "SQL-запросы к базам данных",          "expected": "full"},
    {"a": "unit testing",              "b": "юнит-тестирование",                   "expected": "full"},
    {"a": "Git version control",       "b": "система контроля версий Git",         "expected": "full"},

    # ── PARTIAL: схожая тематика, частичное пересечение ──
    {"a": "Python",                    "b": "backend разработка на Python и Django","expected": "partial"},
    {"a": "data analysis",             "b": "анализ и визуализация данных",         "expected": "partial"},
    {"a": "CI/CD",                     "b": "непрерывная интеграция и доставка ПО", "expected": "partial"},
    {"a": "testing",                   "b": "написание тест-кейсов и тест-планов",  "expected": "partial"},
    {"a": "cloud computing",           "b": "работа с AWS и облачными сервисами",   "expected": "partial"},
    {"a": "security",                  "b": "информационная безопасность приложений","expected": "partial"},

    # ── RELATED: одна область, разные концепции ──
    {"a": "Docker",                    "b": "Kubernetes оркестрация",               "expected": "related"},
    {"a": "SQL",                       "b": "NoSQL базы данных MongoDB",            "expected": "related"},
    {"a": "machine learning",          "b": "глубокое обучение нейросетей",         "expected": "related"},
    {"a": "Agile",                     "b": "Scrum методология разработки",         "expected": "related"},
    {"a": "Python",                    "b": "JavaScript разработка интерфейсов",    "expected": "related"},
    {"a": "PostgreSQL",                "b": "Redis кэширование данных",             "expected": "related"},

    # ── NONE: полностью разные области ──
    {"a": "Python programming",        "b": "управление персоналом",               "expected": "none"},
    {"a": "машинное обучение",         "b": "бухгалтерский учёт и аудит",          "expected": "none"},
    {"a": "Docker containerization",   "b": "графический дизайн и иллюстрация",    "expected": "none"},
    {"a": "REST API",                  "b": "кулинария и технологии питания",       "expected": "none"},
    {"a": "database queries",          "b": "физическая культура и спорт",         "expected": "none"},
    {"a": "Git version control",       "b": "юридическое сопровождение сделок",    "expected": "none"},
    {"a": "CI/CD pipeline",            "b": "медицинская диагностика",             "expected": "none"},
]


@router.get("/model-info")
def model_info():
    """Информация о загруженной модели без запуска бенчмарка."""
    import torch
    svc = get_embedding_service()
    svc._ensure_model()
    return {
        "model_name": svc.model_name,
        "embedding_dim": svc._dim,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "max_seq_length": settings.max_seq_length,
        "thresholds": {
            "full": settings.threshold_full,
            "partial": settings.threshold_partial,
            "related": settings.threshold_related,
        },
        "benchmark_size": len(BENCHMARK_PAIRS),
    }


@router.post("/benchmark")
def run_benchmark():
    """
    Полный бенчмарк модели:
    1. Скорость инференса (мс/компетенция, компетенций/сек)
    2. Точность классификации на размеченном наборе пар
    3. Precision / Recall / F1 по каждой категории
    4. Macro F1, Weighted F1
    5. Матрица ошибок
    6. Распределение similarity по ожидаемым категориям
    7. Детальные результаты по каждой паре
    """
    import torch
    svc = get_embedding_service()

    # ── 1. Скорость инференса ──────────────────────────────────────────────
    warmup = ["warmup text"] * 5
    svc.encode(warmup)  # прогрев

    batch_sizes = [10, 50, 100]
    speed_results = []
    for n in batch_sizes:
        texts = [f"компетенция номер {i}: разработка программного обеспечения" for i in range(n)]
        t0 = time.perf_counter()
        svc.encode(texts, normalize=True)
        elapsed = (time.perf_counter() - t0) * 1000
        speed_results.append({
            "batch_size": n,
            "elapsed_ms": round(elapsed, 1),
            "ms_per_item": round(elapsed / n, 2),
            "items_per_sec": round(n / (elapsed / 1000), 1),
        })

    # ── 2. Оценка на benchmark-парах ──────────────────────────────────────
    confusion = {exp: {pred: 0 for pred in CATEGORIES} for exp in CATEGORIES}
    pair_results = []
    sim_by_expected: dict[str, list[float]] = {c: [] for c in CATEGORIES}

    for pair in BENCHMARK_PAIRS:
        match = svc.match_pair(pair["a"], pair["b"])
        predicted = match["match_type"]
        expected = pair["expected"]
        sim = round(match["similarity"], 4)
        confusion[expected][predicted] += 1
        sim_by_expected[expected].append(sim)
        pair_results.append({
            "a": pair["a"],
            "b": pair["b"],
            "expected": expected,
            "predicted": predicted,
            "similarity": sim,
            "correct": predicted == expected,
        })

    # ── 3. Precision / Recall / F1 по категориям ──────────────────────────
    per_category: dict[str, dict] = {}
    for cat in CATEGORIES:
        tp = confusion[cat][cat]
        fp = sum(confusion[other][cat] for other in CATEGORIES if other != cat)
        fn = sum(confusion[cat][other] for other in CATEGORIES if other != cat)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        support   = sum(confusion[cat].values())
        sims      = sim_by_expected[cat]
        per_category[cat] = {
            "precision":  round(precision, 3),
            "recall":     round(recall, 3),
            "f1":         round(f1, 3),
            "support":    support,
            "sim_mean":   round(float(np.mean(sims)),  3) if sims else 0.0,
            "sim_min":    round(float(np.min(sims)),   3) if sims else 0.0,
            "sim_max":    round(float(np.max(sims)),   3) if sims else 0.0,
        }

    # ── 4. Агрегированные метрики ──────────────────────────────────────────
    total   = len(pair_results)
    correct = sum(1 for r in pair_results if r["correct"])
    accuracy = correct / total if total > 0 else 0.0

    supports     = [per_category[c]["support"] for c in CATEGORIES]
    f1_scores    = [per_category[c]["f1"]      for c in CATEGORIES]
    macro_f1     = float(np.mean(f1_scores))
    weighted_f1  = float(np.average(f1_scores, weights=supports)) if sum(supports) else 0.0

    return {
        "accuracy":     round(accuracy, 4),
        "macro_f1":     round(macro_f1, 4),
        "weighted_f1":  round(weighted_f1, 4),
        "correct":      correct,
        "total":        total,
        "model": {
            "name":          svc.model_name,
            "embedding_dim": svc._dim,
            "device":        "cuda" if torch.cuda.is_available() else "cpu",
        },
        "thresholds": {
            "full":    settings.threshold_full,
            "partial": settings.threshold_partial,
            "related": settings.threshold_related,
        },
        "per_category":     per_category,
        "confusion_matrix": confusion,
        "speed":            speed_results,
        "pairs":            pair_results,
    }


# ---------------------------------------------------------------------------
# Вспомогательная функция: бенчмарк одной произвольной модели
# ---------------------------------------------------------------------------

def _benchmark_model(model_name: str) -> dict:
    """
    Загружает модель по имени, прогоняет весь бенчмарк,
    затем выгружает из памяти. Возвращает метрики.
    """
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Загрузка и замер времени загрузки
    logger.info(f"Загрузка модели для сравнения: {model_name}")
    t0 = time.perf_counter()
    model = SentenceTransformer(model_name, device=device)
    load_ms = round((time.perf_counter() - t0) * 1000, 0)
    dim = model.get_sentence_embedding_dimension()

    # Прогрев
    model.encode(["warmup text"] * 5, normalize_embeddings=True)

    # Скорость: батч 50 текстов
    speed_texts = [f"competency example number {i} software development" for i in range(50)]
    t1 = time.perf_counter()
    model.encode(speed_texts, normalize_embeddings=True)
    speed_elapsed = (time.perf_counter() - t1) * 1000

    # Бенчмарк по парам
    confusion = {exp: {pred: 0 for pred in CATEGORIES} for exp in CATEGORIES}
    pair_results = []

    for pair in BENCHMARK_PAIRS:
        embs = model.encode([pair["a"], pair["b"]], normalize_embeddings=True,
                            convert_to_numpy=True)
        sim = float(np.dot(embs[0], embs[1]))

        if sim >= settings.threshold_full:
            predicted = "full"
        elif sim >= settings.threshold_partial:
            predicted = "partial"
        elif sim >= settings.threshold_related:
            predicted = "related"
        else:
            predicted = "none"

        expected = pair["expected"]
        confusion[expected][predicted] += 1
        pair_results.append({
            "a": pair["a"], "b": pair["b"],
            "expected": expected, "predicted": predicted,
            "similarity": round(sim, 4),
            "correct": predicted == expected,
        })

    # Метрики
    per_category: dict[str, dict] = {}
    for cat in CATEGORIES:
        tp = confusion[cat][cat]
        fp = sum(confusion[o][cat] for o in CATEGORIES if o != cat)
        fn = sum(confusion[cat][o] for o in CATEGORIES if o != cat)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        per_category[cat] = {
            "precision": round(precision, 3),
            "recall":    round(recall, 3),
            "f1":        round(f1, 3),
            "support":   sum(confusion[cat].values()),
        }

    total   = len(pair_results)
    correct = sum(1 for r in pair_results if r["correct"])
    accuracy  = correct / total if total > 0 else 0.0
    macro_f1  = float(np.mean([per_category[c]["f1"] for c in CATEGORIES]))
    supports  = [per_category[c]["support"] for c in CATEGORIES]
    f1_scores = [per_category[c]["f1"]      for c in CATEGORIES]
    weighted_f1 = float(np.average(f1_scores, weights=supports)) if sum(supports) else 0.0

    # Выгрузка из памяти
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "accuracy":      round(accuracy, 4),
        "macro_f1":      round(macro_f1, 4),
        "weighted_f1":   round(weighted_f1, 4),
        "correct":       correct,
        "total":         total,
        "embedding_dim": dim,
        "load_time_ms":  load_ms,
        "ms_per_item":   round(speed_elapsed / 50, 2),
        "items_per_sec": round(50 / (speed_elapsed / 1000), 1),
        "per_category":  per_category,
        "confusion":     confusion,
        "pairs":         pair_results,
    }


# ---------------------------------------------------------------------------
# Эндпоинт: сравнительная оценка трёх моделей
# ---------------------------------------------------------------------------

@router.post("/compare")
def compare_models():
    """
    Запускает одинаковый бенчмарк последовательно на трёх моделях
    и возвращает сравнительные результаты для обоснования выбора модели.
    Каждая модель загружается и выгружается отдельно (экономия RAM).
    """
    results = []
    for meta in COMPARISON_MODELS:
        logger.info(f"Сравнение: запуск бенчмарка для {meta['label']}")
        try:
            metrics = _benchmark_model(meta["name"])
            results.append({**meta, **metrics, "error": None})
        except Exception as e:
            logger.error(f"Ошибка при оценке {meta['label']}: {e}")
            results.append({**meta, "error": str(e)})

    # Определяем победителя по macro_f1 среди многоязычных
    multilingual = [r for r in results if r.get("multilingual") and not r.get("error")]
    winner_id = max(multilingual, key=lambda r: r["macro_f1"])["id"] if multilingual else None

    return {
        "models":    results,
        "winner_id": winner_id,
        "thresholds": {
            "full":    settings.threshold_full,
            "partial": settings.threshold_partial,
            "related": settings.threshold_related,
        },
        "benchmark_size": len(BENCHMARK_PAIRS),
    }
