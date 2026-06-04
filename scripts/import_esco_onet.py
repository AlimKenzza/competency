#!/usr/bin/env python3
"""
import_esco_onet.py
-------------------
Импорт реальных данных ESCO и O*NET в базу данных системы.

Запуск (после скачивания данных):
    python scripts/import_esco_onet.py --esco data/esco --onet data/onet --lang en

ESCO CSV bundle:  https://esco.ec.europa.eu/en/use-esco/download
O*NET Text files: https://www.onetcenter.org/database.html
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.db.session import init_db, SessionLocal
from app.models import entities as m
from app.core.enums import SourceType
from app.services import entity_extractor as ee
from app.api.sources import _persist_extraction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--esco", help="Каталог ESCO CSV (occupations_*, skills_*, occupationSkillRelations_*)")
    parser.add_argument("--onet", help="Каталог O*NET (Occupation Data.txt, Skills.txt, ...)")
    parser.add_argument("--lang", default="en", help="Язык ESCO (en, ru, ...)")
    args = parser.parse_args()

    if not args.esco and not args.onet:
        parser.print_help()
        sys.exit(1)

    init_db()
    db = SessionLocal()

    try:
        if args.esco:
            esco_dir = Path(args.esco)
            print(f"Импорт ESCO из {esco_dir} (язык: {args.lang})...")
            result = ee.extract_from_esco(
                occupations_csv=str(esco_dir / f"occupations_{args.lang}.csv"),
                skills_csv=str(esco_dir / f"skills_{args.lang}.csv"),
                relations_csv=str(esco_dir / f"occupationSkillRelations_{args.lang}.csv"),
                lang=args.lang,
            )
            source = m.Source(
                name=f"ESCO ({args.lang})", source_type=SourceType.ESCO.value,
                description="ESCO classification dataset", lang=args.lang,
            )
            db.add(source)
            db.flush()
            stats = _persist_extraction(db, source, result)
            print(f"  ESCO импортирован: {stats}")

        if args.onet:
            onet_dir = Path(args.onet)
            print(f"Импорт O*NET из {onet_dir}...")
            result = ee.extract_from_onet(
                occupation_data_txt=str(onet_dir / "Occupation Data.txt"),
                skills_txt=str(onet_dir / "Skills.txt"),
                knowledge_txt=str(onet_dir / "Knowledge.txt"),
                work_activities_txt=str(onet_dir / "Work Activities.txt"),
            )
            source = m.Source(
                name="O*NET", source_type=SourceType.ONET.value,
                description="O*NET database", lang="en",
            )
            db.add(source)
            db.flush()
            stats = _persist_extraction(db, source, result)
            print(f"  O*NET импортирован: {stats}")

        print("Готово. Запустите сервер: uvicorn app.main:app")
    finally:
        db.close()


if __name__ == "__main__":
    main()
