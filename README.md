# Интеллектуальная система формирования профессионального профиля IT-специалиста

Программная реализация по техническому заданию: автоматизированное формирование
профессионального профиля IT-специалиста на основе анализа, сопоставления и
интеграции отраслевых рамок компетенций (ESCO, O*NET) и профессиональных
стандартов с применением ИИ-модели семантического сопоставления компетенций.

## Возможности (по разделам ТЗ)

- **7.1** Загрузка документов: TXT, DOCX, PDF, CSV, XLSX
- **7.2** Извлечение сущностей: структурный импорт ESCO/O*NET + правила для свободного текста
- **7.3** Нормализация и унификация: семантическая дедупликация компетенций в кластеры
- **7.4** ИИ-сопоставление: Sentence-BERT + косинусная мера, категоризация (полное / частичное / близкое / отсутствует)
- **7.5** Формирование профиля: отбор, агрегирование, учёт уровней владения
- **7.6** Оценка качества: полнота, непротиворечивость, применимость, согласованность источников
- **7.7** Отчётность: экспорт в PDF, DOCX, XLSX
- **12**  Веб-интерфейс с экспертной корректировкой

## Архитектура

```
competency_system/
├── app/
│   ├── main.py                  # точка входа FastAPI
│   ├── core/
│   │   ├── config.py            # настройки (env)
│   │   └── enums.py             # доменные перечисления
│   ├── db/session.py            # подключение к БД (SQLite/PostgreSQL)
│   ├── models/entities.py       # модели БД (раздел 9 ТЗ)
│   ├── schemas/dto.py           # Pydantic-схемы API
│   ├── services/
│   │   ├── embedding_service.py # ИИ-сопоставление (7.4, 9.3)
│   │   ├── document_loader.py   # загрузка документов (7.1)
│   │   ├── entity_extractor.py  # извлечение сущностей (7.2)
│   │   ├── normalizer.py        # нормализация/унификация (7.3)
│   │   ├── profile_service.py   # агрегирование + профиль (7.5, 9.4, 9.5)
│   │   └── report_service.py    # отчётность (7.7)
│   └── api/
│       ├── sources.py           # импорт источников
│       ├── matching.py          # сопоставление
│       └── profiles.py          # профили и экспорт
├── frontend/templates/index.html  # веб-интерфейс (12)
├── scripts/
│   ├── seed_and_test.py         # e2e-сценарий через TestClient
│   ├── build_local_model.py    # мини-модель (только если нет доступа к HF)
│   └── import_esco_onet.py     # импорт реальных ESCO/O*NET
├── tests/test_logic.py          # юнит-тесты логики
└── requirements.txt
```

## Установка

```bash
cd competency_system
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Проверка GPU (опционально, ускоряет ИИ-модель):
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

## Запуск

### Вариант 1. Быстрый старт на SQLite (по умолчанию)

```bash
uvicorn app.main:app --reload
```

Откройте в браузере: http://localhost:8000
Документация API (Swagger): http://localhost:8000/docs

При первом сопоставлении система скачает модель
`paraphrase-multilingual-MiniLM-L12-v2` (~120 МБ) с HuggingFace.

### Вариант 2. PostgreSQL (продакшен, как в ТЗ)

Создайте БД и задайте переменную окружения:

```bash
# Linux/Mac
export DATABASE_URL="postgresql+psycopg://user:password@localhost:5432/competency"
# Windows (PowerShell)
$env:DATABASE_URL="postgresql+psycopg://user:password@localhost:5432/competency"

uvicorn app.main:app
```

Или создайте файл `.env` в каталоге `competency_system/`:
```
DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/competency
```

## Проверка работоспособности

Юнит-тесты логики (не требуют ML-модели):
```bash
python tests/test_logic.py
# Ожидается: Пройдено: 7/7
```

End-to-end сценарий (импорт → сопоставление → профиль → экспорт):
```bash
python scripts/seed_and_test.py
```

## Импорт реальных данных ESCO / O*NET

1. Скачайте данные:
   - ESCO (CSV): https://esco.ec.europa.eu/en/use-esco/download
   - O*NET (Text): https://www.onetcenter.org/database.html
2. Распакуйте в `data/esco/` и `data/onet/`
3. Запустите импорт:
   ```bash
   python scripts/import_esco_onet.py --esco data/esco --onet data/onet --lang en
   ```

## Настройка порогов сопоставления

В `app/core/config.py` или через переменные окружения:

| Параметр | По умолчанию | Значение |
|----------|--------------|----------|
| `THRESHOLD_FULL` | 0.85 | порог полного совпадения |
| `THRESHOLD_PARTIAL` | 0.65 | порог частичного совпадения |
| `THRESHOLD_RELATED` | 0.45 | порог близкого совпадения |

## Использование дообученной модели

Если вы дообучили sentence-transformer (см. проект profile_builder):
```bash
export FINETUNED_MODEL_PATH="/path/to/your/model"
uvicorn app.main:app
```

## API (основные эндпоинты)

| Метод | Путь | Назначение |
|-------|------|------------|
| POST | `/api/sources/import/free-text` | импорт из текста |
| POST | `/api/sources/import/upload` | загрузка файла |
| GET | `/api/sources` | список источников |
| POST | `/api/match/pair` | сопоставить две компетенции |
| POST | `/api/match/batch` | сопоставить с кандидатами |
| POST | `/api/profiles/build` | построить профиль |
| GET | `/api/profiles` | список профилей |
| PATCH | `/api/profiles/items/{id}` | экспертная корректировка |
| GET | `/api/profiles/{id}/export/{pdf\|docx\|xlsx}` | экспорт отчёта |

## Примечание о ML-модели

Система использует `paraphrase-multilingual-MiniLM-L12-v2` — компактную
многоязычную модель (русский, английский и 50+ языков). Семантическая
близость вычисляется как косинусная мера между эмбеддингами (раздел 9.3 ТЗ).

Файл `scripts/build_local_model.py` нужен ТОЛЬКО для сред без доступа к
интернету (создаёт мини-модель с нуля для проверки кода). В обычной работе
не используется — модель скачивается автоматически.

## Технологический стек

- Python 3.10+
- FastAPI + Uvicorn (backend)
- SQLAlchemy 2.0 (ORM)
- SQLite / PostgreSQL (БД)
- sentence-transformers, PyTorch (ИИ)
- python-docx, pypdf, openpyxl, reportlab (документы)
- HTML/CSS/JS (веб-интерфейс, без сборки)
