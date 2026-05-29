from django.http import HttpRequest, HttpResponse
from django.views.decorators.http import require_GET, require_POST

from bid_statistics.section_views import render_calculation, render_catalog, render_show

SECTION_SLUG = "test-statistics"


@require_GET
def index(request: HttpRequest) -> HttpResponse:
    return render_catalog(request, section_slug=SECTION_SLUG)


@require_GET
def show(request: HttpRequest, slug: str) -> HttpResponse:
    return render_show(request, slug=slug, section_slug=SECTION_SLUG)


@require_POST
def calculate(request: HttpRequest, slug: str) -> HttpResponse:
    return render_calculation(request, slug=slug, section_slug=SECTION_SLUG)
