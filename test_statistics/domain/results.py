from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _json_scalar(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def format_number(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}g}"


def format_p_value(value: float) -> str:
    if value < 0.0001:
        return "< 0.0001"
    return f"{value:.4f}"


@dataclass(frozen=True)
class DisplayValue:
    raw: float | str | None
    display: str

    def to_dict(self) -> dict[str, float | str | None]:
        return {"raw": _json_scalar(self.raw), "display": self.display}


@dataclass(frozen=True)
class ValidationIssue:
    field: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"field": self.field, "message": self.message}


@dataclass(frozen=True)
class ResultMetric:
    label: str
    value: str
    emphasis: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "value": self.value,
            "emphasis": self.emphasis,
        }


@dataclass(frozen=True)
class ResultSection:
    title: str
    metrics: tuple[ResultMetric, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "metrics": [metric.to_dict() for metric in self.metrics],
        }


@dataclass(frozen=True)
class ResultTable:
    title: str
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    caption: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "columns": list(self.columns),
            "rows": [list(row) for row in self.rows],
            "caption": self.caption,
        }


@dataclass(frozen=True)
class DecisionSummary:
    alpha: float
    reject_null: bool
    conclusion: str

    def to_dict(self) -> dict[str, object]:
        return {
            "alpha": _json_scalar(self.alpha),
            "rejectNull": bool(self.reject_null),
            "conclusion": self.conclusion,
        }


@dataclass(frozen=True)
class CalculationResult:
    slug: str
    test_name: str
    statistic_name: str | None = None
    statistic: DisplayValue | None = None
    p_value: DisplayValue | None = None
    metrics: tuple[ResultMetric, ...] = field(default_factory=tuple)
    sections: tuple[ResultSection, ...] = field(default_factory=tuple)
    tables: tuple[ResultTable, ...] = field(default_factory=tuple)
    decision: DecisionSummary | None = None
    interpretation: str = ""
    warnings: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "testName": self.test_name,
            "statisticName": self.statistic_name,
            "statistic": self.statistic.to_dict() if self.statistic else None,
            "pValue": self.p_value.to_dict() if self.p_value else None,
            "metrics": [metric.to_dict() for metric in self.metrics],
            "sections": [section.to_dict() for section in self.sections],
            "tables": [table.to_dict() for table in self.tables],
            "decision": self.decision.to_dict() if self.decision else None,
            "interpretation": self.interpretation,
            "warnings": list(self.warnings),
            "notes": list(self.notes),
        }


def display_number(value: float | None, *, digits: int = 6, empty: str = "—") -> DisplayValue | None:
    if value is None:
        return DisplayValue(raw=None, display=empty)
    return DisplayValue(raw=value, display=format_number(value, digits=digits))


def display_p_value(value: float | None, *, empty: str = "—") -> DisplayValue | None:
    if value is None:
        return DisplayValue(raw=None, display=empty)
    return DisplayValue(raw=value, display=format_p_value(value))
