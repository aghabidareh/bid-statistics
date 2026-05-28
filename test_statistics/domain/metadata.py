from __future__ import annotations

from dataclasses import dataclass, field

from test_statistics.domain.enums import InputKind, TestFamily


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

    @property
    def default_values(self) -> dict[str, str]:
        return {input_field.name: input_field.default_value for input_field in self.input_fields}

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
        }
