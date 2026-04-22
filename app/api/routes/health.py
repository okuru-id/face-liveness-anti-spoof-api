from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    return {"status": "ok", "version": "1.0.0", "model_version": "liveness-v1.0"}
