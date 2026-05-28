from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from test_statistics.domain.metadata import CalculatorMetadata
from test_statistics.domain.results import CalculationResult
from test_statistics.services import calculators as _calculator_modules
from test_statistics.services.calculators.base import BaseCalculator

_ = _calculator_modules


def list_calculators() -> tuple[CalculatorMetadata, ...]:
    return BaseCalculator.all_metadata()



def get_calculator(slug: str) -> BaseCalculator:
    return BaseCalculator.for_slug(slug)



def get_calculator_metadata(slug: str) -> CalculatorMetadata:
    return BaseCalculator.for_slug(slug).metadata



def calculate_test_statistic(slug: str, raw_data: Mapping[str, Any]) -> CalculationResult:
    return BaseCalculator.run_for_slug(slug, raw_data)
