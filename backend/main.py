from fastapi import FastAPI

from backend.config.settings import settings
from backend.routes.evaluations import router as evaluations_router
from backend.routes.followup import router as followup_router


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Backend API for ARIES interview evaluation.",
)

app.include_router(evaluations_router)
app.include_router(followup_router)
