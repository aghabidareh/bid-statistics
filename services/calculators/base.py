from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
import inspect
from typing import Any, ClassVar

from domain.enums import AlternativeHypothesis, InputKind
from domain.metadata import CalculatorMetadata, FormOption, InputFieldDefinition
from domain.regression_inputs import RegressionDataset
from domain.results import CalculationResult
from services.regression_validators import parse_regression_dataset
from services.validators import ValidationIssues, raise_if_issues

ALTERNATIVE_OPTIONS = (
    FormOption(label="Two-sided", value=AlternativeHypothesis.TWO_SIDED.value),
    FormOption(label="Greater than", value=AlternativeHypothesis.GREATER.value),
    FormOption(label="Less than", value=AlternativeHypothesis.LESS.value),
)

KS_DISTRIBUTION_OPTIONS = (
    FormOption(label="Normal", value="norm"),
    FormOption(label="Exponential", value="expon"),
    FormOption(label="Uniform", value="uniform"),
)


class UnknownCalculatorError(LookupError):
    pass


class DuplicateCalculatorSlugError(ValueError):
    pass


class BaseCalculator(ABC):
    metadata: ClassVar[CalculatorMetadata]
    _registry: ClassVar[dict[str, type[BaseCalculator]]] = {}
    _instances: ClassVar[dict[type[BaseCalculator], BaseCalculator]] = {}

    def __init_subclass__(cls, *, register: bool = True, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not register or inspect.isabstract(cls):
            return

        metadata = getattr(cls, "metadata", None)
        if metadata is None:
            raise TypeError(f"{cls.__name__} must define metadata before registration.")

        existing = cls._registry.get(metadata.slug)
        if existing is not None and existing is not cls:
            raise DuplicateCalculatorSlugError(f"Calculator slug '{metadata.slug}' is already registered.")
        cls._registry[metadata.slug] = cls

    @classmethod
    def instance(cls) -> BaseCalculator:
        instance = cls._instances.get(cls)
        if instance is None:
            instance = cls()
            cls._instances[cls] = instance
        return instance

    @classmethod
    def for_slug(cls, slug: str) -> BaseCalculator:
        try:
            calculator_type = cls._registry[slug]
        except KeyError as error:
            raise UnknownCalculatorError(slug) from error
        return calculator_type.instance()

    @classmethod
    def all_calculators(cls, section_slug: str | None = None) -> tuple[BaseCalculator, ...]:
        calculator_types = cls._registry.values()
        if section_slug is not None:
            calculator_types = (calculator_type for calculator_type in calculator_types if calculator_type.metadata.section_slug == section_slug)
        ordered_types = sorted(
            calculator_types,
            key=lambda calculator_type: (
                calculator_type.metadata.section_slug,
                calculator_type.metadata.catalog_position,
                calculator_type.metadata.slug,
            ),
        )
        return tuple(calculator_type.instance() for calculator_type in ordered_types)

    @classmethod
    def all_metadata(cls, section_slug: str | None = None) -> tuple[CalculatorMetadata, ...]:
        return tuple(calculator.metadata for calculator in cls.all_calculators(section_slug=section_slug))

    @classmethod
    def run_for_slug(cls, slug: str, raw_data: Mapping[str, Any]) -> CalculationResult:
        return cls.for_slug(slug).run(raw_data)

    @abstractmethod
    def normalize(self, raw_data: Mapping[str, Any]) -> object:
        raise NotImplementedError

    @abstractmethod
    def calculate_result(self, normalized_input: object) -> CalculationResult:
        raise NotImplementedError

    def run(self, raw_data: Mapping[str, Any]) -> CalculationResult:
        normalized_input = self.normalize(raw_data)
        return self.calculate_result(normalized_input)

    def default_values(self) -> dict[str, Any]:
        return self.metadata.default_values


class SingleSampleCalculator(BaseCalculator, ABC, register=False):
    pass


class TwoIndependentSampleCalculator(BaseCalculator, ABC, register=False):
    pass


class PairedSampleCalculator(BaseCalculator, ABC, register=False):
    pass


class NamedGroupCalculator(BaseCalculator, ABC, register=False):
    pass


class TableCalculator(BaseCalculator, ABC, register=False):
    pass


class SurvivalCalculator(BaseCalculator, ABC, register=False):
    pass


class RocComparisonCalculator(BaseCalculator, ABC, register=False):
    pass


class MultivariateCalculator(BaseCalculator, ABC, register=False):
    pass


class RegressionCalculator(BaseCalculator, ABC, register=False):
    def parse_dataset(self, raw_data: Mapping[str, Any], field: str = "dataset") -> RegressionDataset:
        dataset, issues = parse_regression_dataset(raw_data.get(field), field=field)
        raise_if_issues(issues)
        assert dataset is not None
        return dataset



def alpha_field(*, default_value: str = "0.05", help_text: str = "Set the significance level.") -> InputFieldDefinition:
    return InputFieldDefinition(
        name="alpha",
        label="Alpha",
        kind=InputKind.NUMBER,
        help_text=help_text,
        default_value=default_value,
        min_value="0",
        max_value="1",
    )



def alternative_field(*, help_text: str = "Choose the alternative hypothesis.") -> InputFieldDefinition:
    return InputFieldDefinition(
        name="alternative",
        label="Alternative hypothesis",
        kind=InputKind.RADIO,
        help_text=help_text,
        default_value=AlternativeHypothesis.TWO_SIDED.value,
        options=ALTERNATIVE_OPTIONS,
    )



def numeric_field(
    name: str,
    label: str,
    help_text: str,
    *,
    placeholder: str = "",
    default_value: str = "",
    min_value: str | None = None,
    max_value: str | None = None,
) -> InputFieldDefinition:
    return InputFieldDefinition(
        name=name,
        label=label,
        kind=InputKind.NUMBER,
        help_text=help_text,
        placeholder=placeholder,
        default_value=default_value,
        min_value=min_value,
        max_value=max_value,
    )



def text_field(
    name: str,
    label: str,
    help_text: str,
    *,
    placeholder: str = "",
    default_value: str = "",
) -> InputFieldDefinition:
    return InputFieldDefinition(
        name=name,
        label=label,
        kind=InputKind.TEXT,
        help_text=help_text,
        placeholder=placeholder,
        default_value=default_value,
    )



def textarea_field(
    name: str,
    label: str,
    help_text: str,
    *,
    placeholder: str = "",
    rows: int = 5,
) -> InputFieldDefinition:
    return InputFieldDefinition(
        name=name,
        label=label,
        kind=InputKind.TEXTAREA,
        help_text=help_text,
        placeholder=placeholder,
        rows=rows,
    )


__all__ = [
    "ALTERNATIVE_OPTIONS",
    "KS_DISTRIBUTION_OPTIONS",
    "BaseCalculator",
    "DuplicateCalculatorSlugError",
    "MultivariateCalculator",
    "NamedGroupCalculator",
    "PairedSampleCalculator",
    "RegressionCalculator",
    "RocComparisonCalculator",
    "SingleSampleCalculator",
    "SurvivalCalculator",
    "TableCalculator",
    "TwoIndependentSampleCalculator",
    "UnknownCalculatorError",
    "ValidationIssues",
    "alpha_field",
    "alternative_field",
    "numeric_field",
    "text_field",
    "textarea_field",
]
