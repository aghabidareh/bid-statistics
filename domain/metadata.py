from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from domain.enums import InputKind, TestFamily, WorkflowKind


@dataclass(frozen=True)
class FormOption:
    label: str
    value: str

    def to_dict(self) -> dict[str, str]:
        return {"label": self.label, "value": self.value}


@dataclass(frozen=True)
class InputFieldDefinition:
    name: str
    label: str
    kind: InputKind
    help_text: str
    placeholder: str = ""
    default_value: str = ""
    required: bool = True
    rows: int | None = None
    options: tuple[FormOption, ...] = ()
    step: str = "any"
    min_value: str | None = None
    max_value: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "label": self.label,
            "kind": self.kind.value,
            "helpText": self.help_text,
            "placeholder": self.placeholder,
            "defaultValue": self.default_value,
            "required": self.required,
            "rows": self.rows,
            "options": [option.to_dict() for option in self.options],
            "step": self.step,
            "min": self.min_value,
            "max": self.max_value,
        }


@dataclass(frozen=True)
class CalculatorMetadata:
    catalog_position: int
    slug: str
    name: str
    family: TestFamily
    description: str
    check: str
    statistic_formula: str
    assumptions: tuple[str, ...]
    required_sample_data: tuple[str, ...]
    input_fields: tuple[InputFieldDefinition, ...] = field(default_factory=tuple)
    section_slug: str = "test-statistics"
    workflow_kind: WorkflowKind = WorkflowKind.FORM
    dataset_schema: Mapping[str, Any] | None = None
    extra_default_values: Mapping[str, Any] = field(default_factory=dict)

    @property
    def default_values(self) -> dict[str, Any]:
        return {
            **{input_field.name: input_field.default_value for input_field in self.input_fields},
            **dict(self.extra_default_values),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "catalogPosition": self.catalog_position,
            "slug": self.slug,
            "name": self.name,
            "family": self.family.value,
            "description": self.description,
            "check": self.check,
            "statisticFormula": self.statistic_formula,
            "assumptions": list(self.assumptions),
            "requiredSampleData": list(self.required_sample_data),
            "inputFields": [input_field.to_dict() for input_field in self.input_fields],
            "defaultValues": self.default_values,
            "sectionSlug": self.section_slug,
            "workflowKind": self.workflow_kind.value,
            "datasetSchema": dict(self.dataset_schema) if self.dataset_schema is not None else None,
        }
