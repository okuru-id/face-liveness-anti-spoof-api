from fastapi import Header, HTTPException, status

from app.core.config import settings
from app.core.errors import UnauthorizedError


def verify_api_key(x_api_key: str | None = Header(None, alias="X-API-Key")) -> str:
    if x_api_key is None:
        raise UnauthorizedError("Missing X-API-Key header")
    if x_api_key not in settings.api_keys:
        raise UnauthorizedError("Invalid API Key")
    return x_api_key
