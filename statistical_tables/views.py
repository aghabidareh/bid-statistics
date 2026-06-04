from django.http import Http404, HttpRequest, HttpResponse
from django.views.decorators.http import require_GET
from inertia import render

from statistical_tables.tables import build_table_payload, get_table_metadata, list_tables


@require_GET
def index(request: HttpRequest) -> HttpResponse:
    return render(
        request,
        "StatisticalTables/Index",
        {
            "tables": [table.to_dict() for table in list_tables()],
        },
    )


@require_GET
def show(request: HttpRequest, slug: str) -> HttpResponse:
    metadata = get_table_metadata(slug)
    table = build_table_payload(slug)
    if metadata is None or table is None:
        raise Http404("Statistical table not found.")
    return render(
        request,
        "StatisticalTables/Show",
        {
            "table": table,
            "metadata": metadata.to_dict(),
        },
    )
