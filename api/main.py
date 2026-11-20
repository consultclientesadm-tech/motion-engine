from __future__ import annotations

from fastapi import FastAPI, Request

from api import activity
from api.routes import characters, health, jobs, ui

# tabelas: `alembic upgrade head` (o docker-compose roda antes de subir a API)
app = FastAPI(title="Motion Engine")


@app.middleware("http")
async def _activity(request: Request, call_next):
    if request.url.path != "/health":  # o health check do proxy não conta como uso
        activity.touch()
    return await call_next(request)

app.include_router(health.router)
app.include_router(jobs.router, prefix="/jobs", tags=["jobs"])
app.include_router(characters.router, prefix="/characters", tags=["characters"])
app.include_router(ui.router, tags=["ui"])  # página de teste em /ui
