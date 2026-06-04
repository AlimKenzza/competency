"""
API: семантическое сопоставление компетенций (раздел 7.4 ТЗ).
"""

from __future__ import annotations

from fastapi import APIRouter
from app.schemas.dto import MatchRequest, MatchResult, BatchMatchRequest
from app.services.embedding_service import get_embedding_service

router = APIRouter(prefix="/api/match", tags=["matching"])


@router.post("/pair", response_model=MatchResult)
def match_pair(req: MatchRequest):
    """Сопоставление двух компетенций с категоризацией совпадения."""
    svc = get_embedding_service()
    return svc.match_pair(req.text_a, req.text_b)


@router.post("/batch")
def match_batch(req: BatchMatchRequest):
    """Сопоставление компетенции-запроса со списком кандидатов."""
    svc = get_embedding_service()
    results = svc.match_batch(req.query, req.candidates, top_k=req.top_k)
    return {"query": req.query, "results": results}
