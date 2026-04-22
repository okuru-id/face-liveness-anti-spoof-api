from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from app.api.routes.health import router as health_router
from app.api.routes.liveness import router as liveness_router
from app.api.routes.demo import router as demo_router
from app.api.routes.stream import router as stream_router
from app.core.config import settings
from app.core.errors import AppError
from app.core.logging import setup_logging
from app.api.responses import create_error_response

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
)

app.include_router(health_router, prefix=f"/v{settings.app_version.split('.')[0]}")
app.include_router(liveness_router, prefix=f"/v{settings.app_version.split('.')[0]}")
app.include_router(stream_router, prefix=f"/v{settings.app_version.split('.')[0]}")
app.include_router(demo_router)

@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    return create_error_response(exc)

setup_logging()
