"""
API: источники данных и импорт (разделы 7.1, 7.2 ТЗ).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.enums import SourceType
from app.db.session import get_db
from app.models import entities as m
from app.schemas.dto import SourceOut, ExtractRequest, ExtractedEntities
from app.services import document_loader, entity_extractor

router = APIRouter(prefix="/api/sources", tags=["sources"])


@router.get("", response_model=list[SourceOut])
def list_sources(db: Session = Depends(get_db)):
    return db.query(m.Source).order_by(m.Source.created_at.desc()).all()


@router.delete("/{source_id}")
def delete_source(source_id: int, db: Session = Depends(get_db)):
    src = db.get(m.Source, source_id)
    if not src:
        raise HTTPException(404, "Источник не найден")
    db.delete(src)
    db.commit()
    return {"deleted": source_id}


def _persist_extraction(db: Session, source: m.Source,
                        result: entity_extractor.ExtractionResult) -> dict:
    """Сохраняет извлечённые сущности в БД."""
    n_occ = n_comp = 0
    for occ in result.occupations:
        occ_row = m.Occupation(
            source_id=source.id, external_id=occ.external_id,
            code=occ.code, title=occ.title, description=occ.description,
            lang=occ.lang,
        )
        db.add(occ_row)
        db.flush()
        n_occ += 1
        for wf in occ.work_functions:
            db.add(m.WorkFunction(occupation_id=occ_row.id, title=wf))
        for comp in occ.competencies:
            comp_row = m.Competency(
                source_id=source.id, external_id=comp.external_id,
                title=comp.title, description=comp.description,
                competency_type=comp.competency_type,
                proficiency_level=comp.proficiency_level, lang=comp.lang,
            )
            db.add(comp_row)
            db.flush()
            db.add(m.OccupationCompetency(
                occupation_id=occ_row.id, competency_id=comp_row.id,
            ))
            n_comp += 1
    for comp in result.standalone_competencies:
        db.add(m.Competency(
            source_id=source.id, external_id=comp.external_id,
            title=comp.title, description=comp.description,
            competency_type=comp.competency_type,
            proficiency_level=comp.proficiency_level, lang=comp.lang,
        ))
        n_comp += 1
    db.commit()
    return {"occupations": n_occ, "competencies": n_comp}


@router.post("/import/free-text")
def import_free_text(req: ExtractRequest, db: Session = Depends(get_db)):
    """Импорт компетенций из свободного текста (вакансия, ОП, профстандарт)."""
    result = entity_extractor.extract_from_free_text(
        req.text, role_title=req.role_title, lang="ru"
    )
    source = m.Source(
        name=req.role_title or "Свободный текст",
        source_type=SourceType.FREE_TEXT.value,
        description="Импорт из свободного текста", lang="ru",
    )
    db.add(source)
    db.flush()
    stats = _persist_extraction(db, source, result)
    return {"source_id": source.id, "stats": stats}


@router.post("/import/upload")
async def import_upload(
    file: UploadFile = File(...),
    role_title: str = Form(""),
    db: Session = Depends(get_db),
):
    """
    Загрузка документа (TXT/DOCX/PDF/CSV/XLSX) и извлечение компетенций.
    Для текстовых форматов применяется извлечение по правилам.
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in document_loader.SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"Неподдерживаемый формат: {ext}")

    dest = Path(settings.upload_dir) / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    loaded = document_loader.load_any(dest)
    if loaded["kind"] == "text":
        result = entity_extractor.extract_from_free_text(
            loaded["text"], role_title=role_title, lang="ru"
        )
    else:
        # Табличный импорт: ожидаем колонку с компетенциями
        df = loaded["table"]
        title_col = next((c for c in df.columns
                          if c.lower() in ("title", "competency", "компетенция",
                                           "skill", "навык", "preferredlabel")), None)
        if title_col is None:
            raise HTTPException(400, "В таблице не найдена колонка с компетенциями")
        result = entity_extractor.ExtractionResult(
            standalone_competencies=[
                entity_extractor.ExtractedCompetency(title=str(v))
                for v in df[title_col].dropna().unique()
            ]
        )

    source = m.Source(
        name=role_title or file.filename,
        source_type=SourceType.FREE_TEXT.value,
        description=f"Загружен файл {file.filename}",
        file_path=str(dest), lang="ru",
    )
    db.add(source)
    db.flush()
    stats = _persist_extraction(db, source, result)
    return {"source_id": source.id, "filename": file.filename, "stats": stats}


@router.post("/extract/preview", response_model=ExtractedEntities)
def extract_preview(req: ExtractRequest):
    """Предпросмотр извлечения без сохранения в БД."""
    result = entity_extractor.extract_from_free_text(req.text, lang="ru")
    comps = (result.standalone_competencies +
             [c for o in result.occupations for c in o.competencies])
    return ExtractedEntities(
        role_titles=[o.title for o in result.occupations],
        work_functions=[w for o in result.occupations for w in o.work_functions],
        competencies=[c.title for c in comps],
    )
