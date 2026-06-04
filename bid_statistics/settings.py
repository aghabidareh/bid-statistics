from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = 'django-insecure-0*cx9!vdh$oy8wxv6rp_b+6j_7w0(#3h&$k&6ot=0u-@bf-9-q'
DEBUG = True
ALLOWED_HOSTS = []

INSTALLED_APPS = [
    "django.contrib.messages",
    "django.contrib.sessions",
    "django.contrib.staticfiles",
    "django_vite",
    "inertia",
    "test_statistics.apps.TestStatisticsConfig",
    "regression.apps.RegressionConfig",
    "statistical_tables.apps.StatisticalTablesConfig",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "bid_statistics.middleware.RateLimitMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "inertia.middleware.InertiaMiddleware",
]

ROOT_URLCONF = "bid_statistics.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "bid_statistics.wsgi.application"

CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
        "LOCATION": "bid-statistics",
    }
}

RATE_LIMIT_CACHE_ALIAS = "default"
RATE_LIMIT_MAX_REQUESTS = 10
RATE_LIMIT_WINDOW_SECONDS = 10

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "/static/"
STATICFILES_DIRS = [BASE_DIR / "frontend" / "dist"]

INERTIA_LAYOUT = "base.html"
MESSAGE_STORAGE = "django.contrib.messages.storage.cookie.CookieStorage"
SESSION_ENGINE = "django.contrib.sessions.backends.signed_cookies"

DJANGO_VITE = {
    "default": {
        "dev_mode": DEBUG,
        "dev_server_host": "localhost",
        "dev_server_port": 5173,
        "manifest_path": BASE_DIR / "frontend" / "dist" / "manifest.json",
    }
}

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"