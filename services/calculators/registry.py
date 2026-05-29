from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from domain.metadata import CalculatorMetadata
from domain.results import CalculationResult
from services import calculators as _calculator_modules
from services.calculators.base import BaseCalculator

_ = _calculator_modules

DEFAULT_SECTION_SLUG = "test-statistics"



def list_calculators(*, section_slug: str = DEFAULT_SECTION_SLUG) -> tuple[CalculatorMetadata, ...]:
    return BaseCalculator.all_metadata(section_slug=section_slug)



def list_all_calculators() -> tuple[CalculatorMetadata, ...]:
    return BaseCalculator.all_metadata(section_slug=None)



def get_calculator(slug: str) -> BaseCalculator:
    return BaseCalculator.for_slug(slug)



def get_calculator_metadata(slug: str) -> CalculatorMetadata:
    return BaseCalculator.for_slug(slug).metadata



def calculate_test_statistic(slug: str, raw_data: Mapping[str, Any]) -> CalculationResult:
    return BaseCalculator.run_for_slug(slug, raw_data)
