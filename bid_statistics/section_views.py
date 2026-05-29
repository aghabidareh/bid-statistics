from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from django.http import Http404, HttpRequest, HttpResponse
from django.middleware.csrf import get_token
from inertia import render

from services.calculators.base import UnknownCalculatorError
from services.calculators.registry import calculate_test_statistic, get_calculator_metadata, list_calculators
from services.validators import ValidationIssues, errors_by_field

SECTION_CONFIG = {
    "test-statistics": {
        "name": "Test Statistics",
        "description": "Run the full 26-calculator test statistics catalog from one shared workflow.",
        "href": "/test-statistics/",
        "index_component": "TestStatistics/Index",
        "show_component": "TestStatistics/Show",
    },
    "regression": {
        "name": "Regression",
        "description": "Fit regression, classification, and matching workflows with spreadsheet-style data entry and prediction rows.",
        "href": "/regression/",
        "index_component": "Regression/Index",
        "show_component": "Regression/Show",
    },
}



def build_section(section_slug: str) -> dict[str, object]:
    config = SECTION_CONFIG[section_slug]
    catalog = list_calculators(section_slug=section_slug)
    return {
        "slug": section_slug,
        "name": config["name"],
        "description": config["description"],
        "href": config["href"],
        "itemCount": len(catalog),
    }



def render_catalog(request: HttpRequest, *, section_slug: str) -> HttpResponse:
    config = SECTION_CONFIG[section_slug]
    return render(
        request,
        config["index_component"],
        {
            "catalog": [calculator.to_dict() for calculator in list_calculators(section_slug=section_slug)],
        },
    )



def render_show(request: HttpRequest, *, slug: str, section_slug: str) -> HttpResponse:
    metadata = get_metadata_or_404(slug, section_slug=section_slug)
    return render(request, SECTION_CONFIG[section_slug]["show_component"], build_show_props(request, metadata))



def render_calculation(request: HttpRequest, *, slug: str, section_slug: str) -> HttpResponse:
    metadata = get_metadata_or_404(slug, section_slug=section_slug)
    form_values = {
        **metadata.default_values,
        **get_request_data(request),
    }

    try:
        result = calculate_test_statistic(metadata.slug, form_values)
        errors: dict[str, list[str]] = {}
    except ValidationIssues as error:
        result = None
        errors = errors_by_field(error.issues)

    return render(
        request,
        SECTION_CONFIG[section_slug]["show_component"],
        build_show_props(
            request,
            metadata,
            form_values=form_values,
            validation_errors=errors,
            result=result.to_dict() if result else None,
        ),
    )



def build_show_props(
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
            "action": f"/{metadata.section_slug}/{metadata.slug}/calculate/",
            "values": merged_form_values,
            "errors": dict(validation_errors or {}),
            "csrfToken": get_token(request),
        },
        "result": result,
    }



def get_request_data(request: HttpRequest) -> dict[str, Any]:
    if request.headers.get("Content-Type", "").startswith("application/json") and request.body:
        payload = json.loads(request.body)
        return {key: normalize_request_value(value) for key, value in payload.items()}

    data: dict[str, Any] = {}
    for key, value in request.POST.items():
        if key == "csrfmiddlewaretoken":
            continue
        if key == "dataset":
            try:
                data[key] = json.loads(value)
                continue
            except json.JSONDecodeError:
                pass
        data[key] = value
    return data



def normalize_request_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool, list, dict)):
        return value
    return str(value)



def get_metadata_or_404(slug: str, *, section_slug: str):
    try:
        metadata = get_calculator_metadata(slug)
    except UnknownCalculatorError as error:
        raise Http404("Calculator not found.") from error
    if metadata.section_slug != section_slug:
        raise Http404("Calculator not found.")
    return metadata
