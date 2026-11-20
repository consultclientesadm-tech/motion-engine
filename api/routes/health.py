from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from api.budget import daily_cap, spent_today
from db.session import get_session

router = APIRouter()


@router.get("/health")
def health(session: Session = Depends(get_session)):
    session.execute(text("select 1"))
    return {"ok": True, "spent_today_usd": round(spent_today(session), 4), "daily_cap_usd": daily_cap(None)}
