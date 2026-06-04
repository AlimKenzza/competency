"""
API: формирование профессионального профиля (разделы 7.5, 7.6, 12 ТЗ).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models import entities as m
from app.schemas.dto import (
    ProfileBuildRequest, ProfileOut, ProfileItemUpdate
)
from app.services import profile_service, report_service

router = APIRouter(prefix="/api/profiles", tags=["profiles"])


@router.post("/build", response_model=ProfileOut)
def build_profile(req: ProfileBuildRequest, db: Session = Depends(get_db)):
    """
    Формирование профиля по роли на основе пула компетенций из БД
    (раздел 7.5 + алгоритм 9.5).
    """
    # Собираем кандидатов из БД
    q = db.query(m.Competency)
    if req.source_ids:
        q = q.filter(m.Competency.source_id.in_(req.source_ids))
    db_comps = q.all()
    if not db_comps:
        raise HTTPException(400, "Нет компетенций в БД. Сначала импортируйте источники.")

    # Карта source_id → имя
    src_names = {s.id: s.name for s in db.query(m.Source).all()}

    candidates = [
        profile_service.CandidateCompetency(
            competency_id=c.id, title=c.title,
            competency_type=c.competency_type,
            proficiency_level=c.proficiency_level,
            source_id=c.source_id,
            source_name=src_names.get(c.source_id, ""),
        )
        for c in db_comps
    ]

    built = profile_service.build_profile(
        role_title=req.role_title,
        role_description=req.role_description,
        candidates=candidates,
        top_n=req.top_n,
        min_similarity=req.min_similarity,
    )

    # Сохраняем профиль
    profile = m.Profile(
        role_title=built.role_title,
        role_description=built.role_description,
        params={"top_n": req.top_n, "min_similarity": req.min_similarity,
                "source_ids": req.source_ids},
        quality_metrics=built.quality_metrics,
    )
    db.add(profile)
    db.flush()
    for it in built.items:
        db.add(m.ProfileItem(
            profile_id=profile.id, competency_id=it.competency_id,
            title=it.title, competency_type=it.competency_type,
            proficiency_level=it.proficiency_level, similarity=it.similarity,
            source_refs=it.source_refs, n_sources=it.n_sources,
        ))
    db.commit()
    db.refresh(profile)
    return profile


@router.get("", response_model=list[ProfileOut])
def list_profiles(db: Session = Depends(get_db)):
    return db.query(m.Profile).order_by(m.Profile.created_at.desc()).all()


@router.get("/{profile_id}", response_model=ProfileOut)
def get_profile(profile_id: int, db: Session = Depends(get_db)):
    p = db.get(m.Profile, profile_id)
    if not p:
        raise HTTPException(404, "Профиль не найден")
    return p


@router.patch("/items/{item_id}")
def update_item(item_id: int, upd: ProfileItemUpdate, db: Session = Depends(get_db)):
    """Экспертная корректировка элемента профиля (раздел 12 ТЗ)."""
    item = db.get(m.ProfileItem, item_id)
    if not item:
        raise HTTPException(404, "Элемент не найден")
    if upd.expert_confirmed is not None:
        item.expert_confirmed = upd.expert_confirmed
    if upd.expert_note is not None:
        item.expert_note = upd.expert_note
    if upd.proficiency_level is not None:
        item.proficiency_level = upd.proficiency_level
    db.commit()
    return {"updated": item_id}


@router.delete("/{profile_id}")
def delete_profile(profile_id: int, db: Session = Depends(get_db)):
    p = db.get(m.Profile, profile_id)
    if not p:
        raise HTTPException(404, "Профиль не найден")
    db.delete(p)
    db.commit()
    return {"deleted": profile_id}


# --- Экспорт отчётов (раздел 7.7) ---

@router.get("/{profile_id}/export/{fmt}")
def export_profile(profile_id: int, fmt: str, db: Session = Depends(get_db)):
    p = db.get(m.Profile, profile_id)
    if not p:
        raise HTTPException(404, "Профиль не найден")
    items = p.items
    if fmt == "xlsx":
        path = report_service.export_profile_xlsx(p, items)
        media = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    elif fmt == "docx":
        path = report_service.export_profile_docx(p, items)
        media = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    elif fmt == "pdf":
        path = report_service.export_profile_pdf(p, items)
        media = "application/pdf"
    else:
        raise HTTPException(400, "Формат должен быть pdf, docx или xlsx")
    return FileResponse(path, media_type=media, filename=Path(path).name)


from pathlib import Path  # noqa: E402
