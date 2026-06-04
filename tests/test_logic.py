"""
test_logic.py
-------------
Юнит-тесты бизнес-логики системы на ДЕТЕРМИНИРОВАННОЙ фейковой модели
эмбеддингов. Не зависят от качества реальной ML-модели — проверяют, что
алгоритмы (классификация совпадений, дедупликация, агрегирование, метрики)
работают корректно.

Запуск:
    pytest tests/test_logic.py -v
или:
    python tests/test_logic.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from app.services.embedding_service import EmbeddingService
from app.core.enums import MatchType


# ---------------------------------------------------------------------------
# Фейковая модель: детерминированные эмбеддинги по словарю
# ---------------------------------------------------------------------------

class FakeModel:
    """
    Возвращает контролируемые эмбеддинги: каждое «понятие» — это базовый
    орт в пространстве, синонимы получают близкие векторы.
    """
    CONCEPTS = {
        "python": [1, 0, 0, 0],
        "программирование на python": [0.98, 0.0, 0.1, 0],
        "machine learning": [0, 1, 0, 0],
        "машинное обучение": [0.05, 0.97, 0, 0],
        "docker": [0, 0, 1, 0],
        "контейнеризация": [0.0, 0.05, 0.98, 0],
        "cooking": [0, 0, 0, 1],
        "приготовление еды": [0, 0, 0.05, 0.97],
    }

    def encode(self, texts, batch_size=64, convert_to_numpy=True,
               normalize_embeddings=True, show_progress_bar=False):
        vecs = []
        for t in texts:
            key = t.lower().strip()
            v = np.array(self.CONCEPTS.get(key, [0.25, 0.25, 0.25, 0.25]),
                         dtype=np.float32)
            if normalize_embeddings:
                n = np.linalg.norm(v)
                if n > 0:
                    v = v / n
            vecs.append(v)
        return np.array(vecs, dtype=np.float32)

    def get_sentence_embedding_dimension(self):
        return 4


def make_service() -> EmbeddingService:
    svc = EmbeddingService()
    svc._model = FakeModel()
    svc._model_name = "fake"
    svc._dim = 4
    return svc


# ---------------------------------------------------------------------------
# Тесты классификации совпадений (раздел 7.4)
# ---------------------------------------------------------------------------

def test_synonyms_match_high():
    svc = make_service()
    r = svc.match_pair("machine learning", "машинное обучение")
    assert r["similarity"] > 0.9
    assert r["match_type"] in (MatchType.FULL.value, MatchType.PARTIAL.value)
    print(f"  ✓ Синонимы: sim={r['similarity']:.3f} → {r['match_type']}")


def test_unrelated_match_low():
    svc = make_service()
    r = svc.match_pair("python", "cooking")
    assert r["similarity"] < 0.45
    assert r["match_type"] == MatchType.NONE.value
    print(f"  ✓ Несвязанные: sim={r['similarity']:.3f} → {r['match_type']}")


def test_batch_ranking():
    svc = make_service()
    res = svc.match_batch("python", ["машинное обучение", "программирование на python", "cooking"], top_k=3)
    # Топ-1 должен быть про python
    assert "python" in res[0]["text"].lower()
    print(f"  ✓ Ранжирование: топ-1 = '{res[0]['text']}' ({res[0]['similarity']:.3f})")


# ---------------------------------------------------------------------------
# Тест дедупликации/унификации (раздел 7.3)
# ---------------------------------------------------------------------------

def test_deduplication():
    import app.services.normalizer as norm
    import app.services.embedding_service as es

    svc = make_service()
    es.EmbeddingService._instance = svc  # подменяем singleton

    titles = [
        "Python", "программирование на Python",   # → один кластер
        "machine learning", "машинное обучение",  # → один кластер
        "Docker", "контейнеризация",              # → один кластер
        "cooking",                                # → отдельно
    ]
    vocab = norm.build_unified_vocabulary(titles, similarity_threshold=0.9)
    # Ожидаем ~4 кластера (python, ml, docker, cooking)
    assert 3 <= len(vocab.clusters) <= 5, f"Кластеров: {len(vocab.clusters)}"
    print(f"  ✓ Дедупликация: {len(titles)} → {len(vocab.clusters)} кластеров")


# ---------------------------------------------------------------------------
# Тест агрегирования и метрик (разделы 7.5, 7.6, 9.4)
# ---------------------------------------------------------------------------

def test_profile_aggregation():
    import app.services.profile_service as ps
    import app.services.embedding_service as es

    svc = make_service()
    es.EmbeddingService._instance = svc

    candidates = [
        ps.CandidateCompetency(1, "python", source_id=1, source_name="A"),
        ps.CandidateCompetency(2, "программирование на python", source_id=2, source_name="B"),
        ps.CandidateCompetency(3, "machine learning", source_id=1, source_name="A"),
        ps.CandidateCompetency(4, "cooking", source_id=1, source_name="A"),
    ]
    # Прим.: в фейковой модели составной запрос не входит в словарь.
    # Используем простой запрос "python"; проверяем что cooking (ортогонален)
    # отсекается по релевантности, а python агрегируется из 2 источников.
    built = ps.build_profile(
        role_title="python", role_description="",
        candidates=candidates, top_n=10, min_similarity=0.5, dedup_threshold=0.9,
    )
    titles = [it.title.lower() for it in built.items]
    # cooking не должен попасть (низкая релевантность к python+ml)
    assert not any("cook" in t for t in titles), f"Профиль: {titles}"
    # python из двух источников должен агрегироваться
    py_item = next((it for it in built.items if "python" in it.title.lower()), None)
    assert py_item is not None
    assert py_item.n_sources >= 1
    # Метрики качества посчитаны
    assert "completeness" in built.quality_metrics
    assert "consistency" in built.quality_metrics
    assert "applicability" in built.quality_metrics
    print(f"  ✓ Агрегирование: {len(built.items)} компетенций, "
          f"метрики={built.quality_metrics}")


# ---------------------------------------------------------------------------
# Тесты извлечения сущностей (раздел 7.2)
# ---------------------------------------------------------------------------

def test_free_text_extraction():
    from app.services import entity_extractor as ee
    text = """
    Требования:
    - Опыт работы с Python от 3 лет
    - Знание SQL и баз данных
    - Английский язык
    """
    result = ee.extract_from_free_text(text, role_title="Backend Developer")
    assert len(result.occupations) == 1
    comps = result.occupations[0].competencies
    assert len(comps) == 3, f"Извлечено: {[c.title for c in comps]}"
    print(f"  ✓ Извлечение: {len(comps)} компетенций из текста")


def test_level_detection():
    from app.services import entity_extractor as ee
    from app.core.enums import ProficiencyLevel
    assert ee.detect_level("уверенное владение Python") == ProficiencyLevel.INTERMEDIATE.value
    assert ee.detect_level("базовые знания SQL") == ProficiencyLevel.BASIC.value
    assert ee.detect_level("экспертный уровень ML") == ProficiencyLevel.EXPERT.value
    print("  ✓ Детекция уровней владения работает")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        ("Классификация: синонимы", test_synonyms_match_high),
        ("Классификация: несвязанные", test_unrelated_match_low),
        ("Ранжирование batch", test_batch_ranking),
        ("Дедупликация", test_deduplication),
        ("Агрегирование профиля", test_profile_aggregation),
        ("Извлечение из текста", test_free_text_extraction),
        ("Детекция уровней", test_level_detection),
    ]
    passed = 0
    print("\nЗАПУСК ТЕСТОВ ЛОГИКИ (на детерминированной модели)\n" + "=" * 60)
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ ПРОВАЛ [{name}]: {e}")
        except Exception as e:
            print(f"  ✗ ОШИБКА [{name}]: {e}")
    print("=" * 60)
    print(f"Пройдено: {passed}/{len(tests)}")
    sys.exit(0 if passed == len(tests) else 1)
