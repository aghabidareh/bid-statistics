from django.conf import settings
from django.core.cache import cache
from django.http import HttpResponse
from django.test import SimpleTestCase, override_settings
from django.urls import path


def rate_limit_test_view(request):
    return HttpResponse("ok")


urlpatterns = [
    path("rate-limit-test/", rate_limit_test_view),
]


@override_settings(
    ROOT_URLCONF=__name__,
    CACHES={
        "default": {
            "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
            "LOCATION": "rate-limit-tests",
        }
    },
    RATE_LIMIT_CACHE_ALIAS="default",
    RATE_LIMIT_MAX_REQUESTS=10,
    RATE_LIMIT_WINDOW_SECONDS=10,
)
class RateLimitMiddlewareTests(SimpleTestCase):
    def setUp(self):
        super().setUp()
        cache.clear()

    def tearDown(self):
        cache.clear()
        super().tearDown()

    def get_response(self, ip_address="127.0.0.1"):
        return self.client.get("/rate-limit-test/", REMOTE_ADDR=ip_address)

    def test_first_ten_requests_succeed(self):
        for _ in range(settings.RATE_LIMIT_MAX_REQUESTS):
            response = self.get_response()
            self.assertEqual(response.status_code, 200)

    def test_eleventh_request_returns_429_with_retry_after(self):
        for _ in range(settings.RATE_LIMIT_MAX_REQUESTS):
            self.get_response()

        response = self.get_response()

        self.assertEqual(response.status_code, 429)
        self.assertIn("Retry-After", response.headers)
        self.assertGreaterEqual(int(response.headers["Retry-After"]), 1)
        self.assertLessEqual(
            int(response.headers["Retry-After"]),
            settings.RATE_LIMIT_WINDOW_SECONDS,
        )

    def test_separate_ips_have_independent_counters(self):
        for _ in range(settings.RATE_LIMIT_MAX_REQUESTS):
            self.get_response("127.0.0.1")

        blocked_response = self.get_response("127.0.0.1")
        allowed_response = self.get_response("127.0.0.2")

        self.assertEqual(blocked_response.status_code, 429)
        self.assertEqual(allowed_response.status_code, 200)
