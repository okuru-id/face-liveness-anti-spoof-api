from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

router = APIRouter()

templates = Jinja2Templates(directory=str(Path(__file__).parent.parent.parent / "templates"))


@router.get("/demo", response_class=HTMLResponse)
async def demo_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="demo.html",
        context={
            "model_version": "liveness-v1.0",
            "api_endpoint": "/v1/liveness/check",
            "api_stream_endpoint": "/v1/liveness/stream/frame",
        },
    )
