from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from django.http import Http404, HttpRequest, HttpResponse
from django.middleware.csrf import get_token
from django.views.decorators.http import require_GET, require_POST
from inertia import render

from test_statistics.services.calculators.base import UnknownCalculatorError
from test_statistics.services.calculators.registry import (
    calculate_test_statistic,
    get_calculator_metadata,
    list_calculators,
)
from test_statistics.services.validators import ValidationIssues, errors_by_field


@require_GET
def home(request: HttpRequest) -> HttpResponse:
    return render(
        request,
        "Home",
        {
            "catalog": [calculator.to_dict() for calculator in list_calculators()],
        },
    )


@require_GET
def show(request: HttpRequest, slug: str) -> HttpResponse:
    metadata = _get_metadata_or_404(slug)
    return render(request, "TestStatistics/Show", _build_show_props(request, metadata))


@require_POST
def calculate(request: HttpRequest, slug: str) -> HttpResponse:
    metadata = _get_metadata_or_404(slug)
    form_values = {
        **metadata.default_values,
        **_get_request_data(request),
    }

    try:
        result = calculate_test_statistic(slug, form_values)
        errors: dict[str, list[str]] = {}
    except ValidationIssues as error:
        result = None
        errors = errors_by_field(error.issues)

    return render(
        request,
        "TestStatistics/Show",
        _build_show_props(
            request,
            metadata,
            form_values=form_values,
            validation_errors=errors,
            result=result.to_dict() if result else None,
        ),
    )



def _build_show_props(
    request: HttpRequest,
    metadata,
    *,
    form_values: Mapping[str, Any] | None = None,
    validation_errors: Mapping[str, list[str]] | None = None,
    result: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    merged_form_values = {
        **metadata.default_values,
        **(dict(form_values) if form_values else {}),
    }
    return {
        "calculator": metadata.to_dict(),
        "form": {
            "action": f"/test-statistics/{metadata.slug}/calculate/",
            "values": merged_form_values,
            "errors": dict(validation_errors or {}),
            "csrfToken": get_token(request),
        },
        "result": result,
    }



def _get_request_data(request: HttpRequest) -> dict[str, str]:
    if request.headers.get("Content-Type", "").startswith("application/json") and request.body:
        payload = json.loads(request.body)
        return {key: "" if value is None else str(value) for key, value in payload.items()}
    return {key: value for key, value in request.POST.items() if key != "csrfmiddlewaretoken"}



def _get_metadata_or_404(slug: str):
    try:
        return get_calculator_metadata(slug)
    except UnknownCalculatorError as error:
        raise Http404("Calculator not found.") from error
