from __future__ import annotations

from collections.abc import Mapping
from math import isclose
from typing import Any

import numpy as np
from scipy import stats

from test_statistics.domain.enums import AlternativeHypothesis, InputKind, TestFamily
from test_statistics.domain.inputs import PearsonCorrelationInput
from test_statistics.domain.metadata import CalculatorMetadata, FormOption, InputFieldDefinition
from test_statistics.domain.results import CalculationResult, DecisionSummary, ResultMetric, format_number
from test_statistics.services.calculators.base import BaseCalculator
from test_statistics.services.validators import (
    build_error,
    parse_alpha,
    parse_alternative,
    parse_numeric_series,
    raise_if_issues,
)


ALTERNATIVE_OPTIONS = (
    FormOption(label="Two-sided", value=AlternativeHypothesis.TWO_SIDED.value),
    FormOption(label="Positive", value=AlternativeHypothesis.GREATER.value),
    FormOption(label="Negative", value=AlternativeHypothesis.LESS.value),
)


class PearsonCorrelationCalculator(BaseCalculator):
    metadata = CalculatorMetadata(
        slug="pearson-correlation",
        name="Pearson correlation test",
        family=TestFamily.CORRELATION,
        description="Measure linear correlation between two numeric variables.",
        check="Whether the linear correlation differs from zero.",
        statistic_formula="r = Σ[(x - x̄)(y - ȳ)] / √(Σ(x - x̄)² Σ(y - ȳ)²)",
        assumptions=(
            "Each pair of observations is independent.",
            "The relationship is approximately linear.",
            "Both variables are approximately continuous and not constant.",
        ),
        required_sample_data=(
            "One numeric series for x.",
            "One numeric series for y with the same length.",
        ),
        input_fields=(
            InputFieldDefinition(
                name="x_values",
                label="X values",
                kind=InputKind.TEXTAREA,
                help_text="Enter comma-separated x values.",
                placeholder="1, 2, 3, 4, 5",
                rows=5,
            ),
            InputFieldDefinition(
                name="y_values",
                label="Y values",
                kind=InputKind.TEXTAREA,
                help_text="Enter comma-separated y values.",
                placeholder="2, 4, 5, 4, 5",
                rows=5,
            ),
            InputFieldDefinition(
                name="alternative",
                label="Alternative hypothesis",
                kind=InputKind.SELECT,
                help_text="Choose the alternative hypothesis.",
                default_value=AlternativeHypothesis.TWO_SIDED.value,
                options=ALTERNATIVE_OPTIONS,
            ),
            InputFieldDefinition(
                name="alpha",
                label="Alpha",
                kind=InputKind.NUMBER,
                help_text="Set the significance level.",
                default_value="0.05",
            ),
        ),
    )

    def normalize(self, raw_data: Mapping[str, Any]) -> PearsonCorrelationInput:
        issues = []
        x_values, x_issues = parse_numeric_series(raw_data.get("x_values"), "x_values", "X values", minimum_length=3)
        y_values, y_issues = parse_numeric_series(raw_data.get("y_values"), "y_values", "Y values", minimum_length=3)
        issues.extend(x_issues)
        issues.extend(y_issues)
        alternative, alternative_issues = parse_alternative(raw_data.get("alternative"))
        issues.extend(alternative_issues)
        alpha, alpha_issues = parse_alpha(raw_data.get("alpha"))
        issues.extend(alpha_issues)

        if x_values is not None and y_values is not None and len(x_values) != len(y_values):
            issues.append(build_error("y_values", "X values and Y values must have the same length."))

        if x_values is not None and len(set(x_values)) == 1:
            issues.append(build_error("x_values", "X values must not all be identical."))
        if y_values is not None and len(set(y_values)) == 1:
            issues.append(build_error("y_values", "Y values must not all be identical."))

        raise_if_issues(issues)
        return PearsonCorrelationInput(
            x_values=x_values,
            y_values=y_values,
            alternative=alternative,
            alpha=alpha,
        )

    def calculate_result(self, normalized_input: PearsonCorrelationInput) -> CalculationResult:
        x_values = np.asarray(normalized_input.x_values, dtype=float)
        y_values = np.asarray(normalized_input.y_values, dtype=float)
        pearson_result = stats.pearsonr(
            x_values,
            y_values,
            alternative=normalized_input.alternative.value,
        )
        statistic = float(pearson_result.statistic)
        p_value = float(pearson_result.pvalue)
        confidence_interval = pearson_result.confidence_interval(confidence_level=1 - normalized_input.alpha)
        slope, intercept = np.polyfit(x_values, y_values, 1)
        r_squared = statistic**2
        reject_null = p_value < normalized_input.alpha

        metrics = (
            ResultMetric("Paired observations", str(x_values.size)),
            ResultMetric("r-squared", format_number(r_squared)),
            ResultMetric("Slope of fitted line", format_number(float(slope)), emphasis=True),
            ResultMetric("Intercept of fitted line", format_number(float(intercept))),
            ResultMetric(
                f"{int((1 - normalized_input.alpha) * 100)}% CI for r",
                f"[{format_number(float(confidence_interval.low))}, {format_number(float(confidence_interval.high))}]",
            ),
        )
        direction = "positive" if statistic > 0 and not isclose(statistic, 0.0) else "negative" if statistic < 0 else "no clear"
        return CalculationResult(
            slug=self.metadata.slug,
            test_name=self.metadata.name,
            statistic_name="r",
            statistic=statistic,
            p_value=p_value,
            metrics=metrics,
            decision=DecisionSummary(
                alpha=normalized_input.alpha,
                reject_null=reject_null,
                conclusion=(
                    f"Reject the null hypothesis. The data support a {direction} linear correlation."
                    if reject_null
                    else "Fail to reject the null hypothesis that the linear correlation is zero."
                ),
            ),
            interpretation=(
                f"The paired data show an estimated Pearson correlation of {format_number(statistic)}."
            ),
        )
