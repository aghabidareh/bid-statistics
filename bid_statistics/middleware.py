import math
import time
from collections.abc import Callable

from django.conf import settings
from django.core.cache import caches
from django.http import HttpRequest, HttpResponse


class RateLimitMiddleware:
    def __init__(self, get_response: Callable[[HttpRequest], HttpResponse]):
        self.get_response = get_response
        self.cache = caches[settings.RATE_LIMIT_CACHE_ALIAS]
        self.max_requests = settings.RATE_LIMIT_MAX_REQUESTS
        self.window_seconds = settings.RATE_LIMIT_WINDOW_SECONDS

    def __call__(self, request: HttpRequest) -> HttpResponse:
        now = time.time()
        client_ip = self.get_client_ip(request)
        cache_key = f"rate-limit:{client_ip}"
        timestamps = self.cache.get(cache_key, [])
        window_start = now - self.window_seconds
        timestamps = [timestamp for timestamp in timestamps if timestamp > window_start]

        if len(timestamps) >= self.max_requests:
            retry_after = max(1, math.ceil(self.window_seconds - (now - timestamps[0])))
            response = HttpResponse("Too Many Requests", status=429)
            response["Retry-After"] = str(retry_after)
            return response

        timestamps.append(now)
        self.cache.set(cache_key, timestamps, timeout=self.window_seconds)
        return self.get_response(request)

    @staticmethod
    def get_client_ip(request: HttpRequest) -> str:
        return request.META.get("REMOTE_ADDR", "unknown")
