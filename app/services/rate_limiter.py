import time
from collections import defaultdict
from dataclasses import dataclass

from app.core.config import settings


@dataclass
class RateLimitInfo:
    remaining: int
    reset: int
    limit: int


store: dict[str, list[float]] = defaultdict(list)


def rate_limit(key: str) -> RateLimitInfo:
    now = time.time()
    window = 60.0
    reset_time = int(now) + 60
    limit = settings.rate_limit_per_minute

    timestamps = store[key]
    timestamps = [ts for ts in timestamps if now - ts < window]
    store[key] = timestamps

    if len(timestamps) >= limit:
        return RateLimitInfo(remaining=0, reset=reset_time, limit=limit)

    remaining = limit - len(timestamps)
    return RateLimitInfo(remaining=remaining, reset=reset_time, limit=limit)


def record_request(key: str) -> None:
    store[key].append(time.time())
