"""
seed_and_test.py
----------------
Наполняет систему демонстрационными данными и прогоняет полный сценарий
через FastAPI TestClient (без поднятия реального сервера):
  1. импорт компетенций из свободного текста (две вакансии);
  2. сопоставление пары компетенций;
  3. построение профиля;
  4. экспертная корректировка;
  5. экспорт отчётов (xlsx/docx/pdf).

Запуск:
    LOCAL_FALLBACK_MODEL=data/local_model python scripts/seed_and_test.py
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Указываем фолбэк-модель и БД до импорта приложения
os.environ.setdefault("LOCAL_FALLBACK_MODEL", str(ROOT / "data" / "local_model"))
os.environ.setdefault("DATABASE_URL", f"sqlite:///{ROOT / 'data' / 'test.db'}")

# Чистим тестовую БД
test_db = ROOT / "data" / "test.db"
if test_db.exists():
    test_db.unlink()

from fastapi.testclient import TestClient
from app.main import app
from app.db.session import init_db

# Явно создаём таблицы (в TestClient startup-события требуют контекст-менеджера)
init_db()

client = TestClient(app)


VACANCY_ML = """
Senior Machine Learning Engineer

Требования:
- Опыт промышленной разработки на Python от 3 лет
- Глубокое знание машинного обучения и нейронных сетей
- Опыт с PyTorch или TensorFlow
- Уверенное владение SQL и работа с большими данными
- Опыт развёртывания моделей в продакшен (MLOps)
- Знание Docker и Kubernetes
- Английский язык на уровне чтения документации
"""

VACANCY_DEVOPS = """
DevOps Engineer

Обязанности и требования:
- Настройка и поддержка CI/CD пайплайнов
- Уверенное владение Docker и Kubernetes
- Опыт работы с облаками AWS или Azure
- Infrastructure as Code (Terraform, Ansible)
- Мониторинг и наблюдаемость (Prometheus, Grafana)
- Администрирование Linux серверов
- Базовое знание Python для автоматизации
"""


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    section("HEALTH CHECK")
    r = client.get("/api/health")
    print(r.json())

    section("1. ИМПОРТ ИЗ СВОБОДНОГО ТЕКСТА (2 вакансии)")
    r1 = client.post("/api/sources/import/free-text", json={
        "text": VACANCY_ML, "role_title": "Machine Learning Engineer"
    })
    print("ML вакансия:", r1.json())
    r2 = client.post("/api/sources/import/free-text", json={
        "text": VACANCY_DEVOPS, "role_title": "DevOps Engineer"
    })
    print("DevOps вакансия:", r2.json())

    section("2. СПИСОК ИСТОЧНИКОВ")
    sources = client.get("/api/sources").json()
    for s in sources:
        print(f"  [{s['id']}] {s['name']} ({s['source_type']})")

    section("3. СОПОСТАВЛЕНИЕ ПАРЫ КОМПЕТЕНЦИЙ")
    pairs = [
        ("машинное обучение", "machine learning"),
        ("Docker и Kubernetes", "контейнеризация и оркестрация"),
        ("программирование на Python", "приготовление еды"),
    ]
    for a, b in pairs:
        r = client.post("/api/match/pair", json={"text_a": a, "text_b": b})
        d = r.json()
        print(f"  '{a}' ↔ '{b}': sim={d['similarity']:.3f} → {d['match_type']}")

    section("4. ПОСТРОЕНИЕ ПРОФИЛЯ (роль: ML Engineer)")
    rp = client.post("/api/profiles/build", json={
        "role_title": "Machine Learning Engineer",
        "role_description": "Разработка и внедрение моделей машинного обучения, "
                            "глубокое обучение, MLOps, развёртывание в продакшен",
        "top_n": 15, "min_similarity": 0.2,
    })
    profile = rp.json()
    print(f"  Профиль #{profile['id']}: {profile['role_title']}")
    print(f"  Метрики качества: {profile['quality_metrics']}")
    print(f"  Компетенций в профиле: {len(profile['items'])}")
    print("  Топ компетенций:")
    for it in profile["items"][:10]:
        print(f"    [{it['similarity']:.3f}] {it['title']} "
              f"(источников: {it['n_sources']})")

    section("5. ЭКСПЕРТНАЯ КОРРЕКТИРОВКА")
    if profile["items"]:
        item_id = profile["items"][0]["id"]
        ru = client.patch(f"/api/profiles/items/{item_id}", json={
            "expert_confirmed": True,
            "expert_note": "Подтверждено экспертом как ключевая компетенция",
        })
        print(f"  Элемент {item_id}: {ru.json()}")

    section("6. ЭКСПОРТ ОТЧЁТОВ")
    pid = profile["id"]
    reports_dir = ROOT / "data" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ("xlsx", "docx", "pdf"):
        r = client.get(f"/api/profiles/{pid}/export/{fmt}")
        if r.status_code == 200:
            out = reports_dir / f"profile_{pid}.{fmt}"
            out.write_bytes(r.content)
            print(f"  {fmt.upper()}: {out} ({len(r.content)} байт)")
        else:
            print(f"  {fmt.upper()}: ОШИБКА {r.status_code} — {r.text[:200]}")

    section("ГОТОВО — все этапы пройдены")


if __name__ == "__main__":
    main()
