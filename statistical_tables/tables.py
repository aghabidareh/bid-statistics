from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Any

from scipy import stats


@dataclass(frozen=True)
class StatisticalTableMetadata:
    catalog_position: int
    slug: str
    name: str
    description: str
    href: str

    def to_dict(self) -> dict[str, object]:
        return {
            "catalogPosition": self.catalog_position,
            "slug": self.slug,
            "name": self.name,
            "description": self.description,
            "href": self.href,
        }


TABLE_CATALOG = (
    StatisticalTableMetadata(
        catalog_position=1,
        slug="z-table",
        name="Z Table",
        description="Standard normal cumulative probabilities and commonly used inverse Z critical values.",
        href="/statistical-tables/z-table/",
    ),
    StatisticalTableMetadata(
        catalog_position=2,
        slug="t-table",
        name="T Table",
        description="Student's t critical values by degrees of freedom for one-tail and two-tail tests.",
        href="/statistical-tables/t-table/",
    ),
)

_Z_COLUMNS = tuple(round(column / 100, 2) for column in range(10))
_Z_ROWS = tuple(round(row / 10, 1) for row in range(-34, 35))
_INVERSE_ALPHA_LEVELS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4)
_T_ONE_TAIL_LEVELS = (0.1, 0.05, 0.025, 0.01, 0.005)
_T_DFS: tuple[int | float, ...] = (*range(1, 31), 40, 50, 60, 80, 100, 120, inf)


def list_tables() -> tuple[StatisticalTableMetadata, ...]:
    return TABLE_CATALOG


def get_table_metadata(slug: str) -> StatisticalTableMetadata | None:
    return next((table for table in TABLE_CATALOG if table.slug == slug), None)


def build_table_payload(slug: str) -> dict[str, Any] | None:
    if slug == "z-table":
        return _build_z_table()
    if slug == "t-table":
        return _build_t_table()
    return None


def _format_probability(value: float) -> str:
    return f"{value:.8f}".rstrip("0").rstrip(".")


def _format_critical(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _format_alpha(alpha: float) -> str:
    return f"{alpha:g}"


def _build_z_table() -> dict[str, Any]:
    probability_rows = []
    for row_base in _Z_ROWS:
        cells = []
        for column in _Z_COLUMNS:
            z_score = round(row_base + column, 2)
            cells.append(
                {
                    "z": _format_critical(z_score),
                    "value": _format_probability(float(stats.norm.cdf(z_score))),
                }
            )
        probability_rows.append({"z": f"{row_base:g}", "cells": cells})

    inverse_rows = []
    for alpha in _INVERSE_ALPHA_LEVELS:
        z_alpha = float(stats.norm.ppf(alpha))
        z_one_minus_alpha = float(stats.norm.ppf(1 - alpha))
        z_alpha_over_two = float(stats.norm.ppf(alpha / 2))
        z_one_minus_alpha_over_two = float(stats.norm.ppf(1 - alpha / 2))
        inverse_rows.append(
            {
                "alpha": _format_alpha(alpha),
                "zAlpha": _format_critical(z_alpha),
                "zOneMinusAlpha": _format_critical(z_one_minus_alpha),
                "zAlphaOverTwo": _format_critical(z_alpha_over_two),
                "zOneMinusAlphaOverTwo": _format_critical(z_one_minus_alpha_over_two),
            }
        )

    return {
        "kind": "z",
        "title": "Z Table",
        "intro": "Explore standard normal cumulative probabilities and inverse Z critical values.",
        "probability": {
            "columns": [f"{column:.2f}" for column in _Z_COLUMNS],
            "rows": probability_rows,
            "defaultCell": {"z": "1.96", "value": _format_probability(float(stats.norm.cdf(1.96)))},
        },
        "inverse": {
            "columns": ["α", "Zα", "Z1-α", "Zα/2", "Z1-α/2"],
            "rows": inverse_rows,
        },
        "education": [
            {
                "title": "What is a Z-Table?",
                "body": "A Z-table gives probabilities for the standard normal distribution, where the mean is 0 and the standard deviation is 1.",
            },
            {
                "title": "How to read it",
                "body": "The left column gives the first decimal place of the z-score, and the top row adds the hundredths place. The cell is P(X ≤ z).",
            },
            {
                "title": "When to use it",
                "body": "Use Z critical values for normal-approximation tests, confidence intervals, and probability calculations when values have been standardized.",
            },
        ],
    }


def _t_critical(alpha: float, degrees_of_freedom: int | float) -> float:
    if degrees_of_freedom == inf:
        return float(stats.norm.ppf(1 - alpha))
    return float(stats.t.ppf(1 - alpha, degrees_of_freedom))


def _build_t_table() -> dict[str, Any]:
    rows = []
    for degrees_of_freedom in _T_DFS:
        rows.append(
            {
                "df": "∞" if degrees_of_freedom == inf else str(degrees_of_freedom),
                "cells": [
                    {
                        "alpha": _format_alpha(alpha),
                        "twoTailAlpha": _format_alpha(2 * alpha),
                        "value": _format_critical(_t_critical(alpha, degrees_of_freedom)),
                    }
                    for alpha in _T_ONE_TAIL_LEVELS
                ],
            }
        )

    return {
        "kind": "t",
        "title": "T Table",
        "intro": "Look up Student's t critical values by degrees of freedom for common one-tail and two-tail significance levels.",
        "criticalValues": {
            "oneTailColumns": [_format_alpha(alpha) for alpha in _T_ONE_TAIL_LEVELS],
            "twoTailColumns": [_format_alpha(2 * alpha) for alpha in _T_ONE_TAIL_LEVELS],
            "rows": rows,
            "defaultCell": {"df": "10", "alpha": "0.05", "value": _format_critical(_t_critical(0.05, 10))},
        },
        "education": [
            {
                "title": "What is a T-Table?",
                "body": "A T-table lists critical values from Student's t distribution, which is commonly used when the population standard deviation is unknown.",
            },
            {
                "title": "Degrees of freedom",
                "body": "For a one-sample t procedure, degrees of freedom are usually n − 1. As degrees of freedom increase, t critical values approach Z critical values.",
            },
            {
                "title": "One-tail and two-tail tests",
                "body": "Use the one-tail α header for directional tests and the two-tail α header for two-sided tests. The table body shows the positive critical value.",
            },
        ],
    }
