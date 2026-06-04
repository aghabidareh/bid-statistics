from django.http import HttpRequest, HttpResponse
from django.views.decorators.http import require_GET
from inertia import render

from bid_statistics.section_views import build_section


@require_GET
def home(request: HttpRequest) -> HttpResponse:
    return render(
        request,
        "Home",
        {
            "sections": [build_section(section_slug) for section_slug in ("test-statistics", "regression", "statistical-tables")],
        },
    )
