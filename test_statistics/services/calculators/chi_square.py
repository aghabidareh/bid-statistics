from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy import stats

from test_statistics.domain.enums import InputKind, TestFamily
from test_statistics.domain.inputs import ChiSquareIndependenceInput
from test_statistics.domain.metadata import CalculatorMetadata, InputFieldDefinition
from test_statistics.domain.results import CalculationResult, DecisionSummary, ResultMetric, format_number
from test_statistics.services.calculators.base import BaseCalculator
from test_statistics.services.validators import parse_alpha, parse_contingency_table, raise_if_issues


class ChiSquareIndependenceCalculator(BaseCalculator):
    metadata = CalculatorMetadata(
        slug="chi-square-independence",
        name="Chi-squared test of independence",
        family=TestFamily.CHI_SQUARE,
        description="Check whether two categorical variables are associated in a contingency table.",
        check="Whether row and column categories are independent.",
        statistic_formula="χ² = Σ (O - E)² / E",
        assumptions=(
            "Observations are independent.",
            "The table contains frequency counts.",
            "Expected counts should be sufficiently large in most cells.",
        ),
        required_sample_data=(
            "Observed counts for a contingency table.",
            "At least two rows and two columns.",
        ),
        input_fields=(
            InputFieldDefinition(
                name="contingency_table",
                label="Observed counts",
                kind=InputKind.TEXTAREA,
                help_text="Enter one comma-separated row per line.",
                placeholder="12, 8, 10\n5, 9, 6",
                rows=6,
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

    def normalize(self, raw_data: Mapping[str, Any]) -> ChiSquareIndependenceInput:
        issues = []
        table, table_issues = parse_contingency_table(raw_data.get("contingency_table"), "contingency_table")
        issues.extend(table_issues)
        alpha, alpha_issues = parse_alpha(raw_data.get("alpha"))
        issues.extend(alpha_issues)
        raise_if_issues(issues)
        return ChiSquareIndependenceInput(contingency_table=table, alpha=alpha)

    def calculate_result(self, normalized_input: ChiSquareIndependenceInput) -> CalculationResult:
        table = np.asarray(normalized_input.contingency_table, dtype=float)
        statistic, p_value, degrees_of_freedom, expected = stats.chi2_contingency(table, correction=False)
        total = float(table.sum())
        minimum_expected = float(expected.min())
        reject_null = float(p_value) < normalized_input.alpha
        metrics = (
            ResultMetric("Rows", str(table.shape[0])),
            ResultMetric("Columns", str(table.shape[1])),
            ResultMetric("Total count", format_number(total)),
            ResultMetric("Degrees of freedom", format_number(float(degrees_of_freedom))),
            ResultMetric("Minimum expected count", format_number(minimum_expected), emphasis=True),
            ResultMetric(
                "Expected counts",
                "; ".join(
                    "[" + ", ".join(format_number(float(value)) for value in row) + "]"
                    for row in expected
                ),
            ),
        )
        return CalculationResult(
            slug=self.metadata.slug,
            test_name=self.metadata.name,
            statistic_name="χ²",
            statistic=float(statistic),
            p_value=float(p_value),
            metrics=metrics,
            decision=DecisionSummary(
                alpha=normalized_input.alpha,
                reject_null=reject_null,
                conclusion=(
                    "Reject the null hypothesis. The row and column variables appear associated."
                    if reject_null
                    else "Fail to reject the null hypothesis of independence."
                ),
            ),
            interpretation=(
                f"The contingency table contains {format_number(total)} observed counts across "
                f"{table.shape[0]} rows and {table.shape[1]} columns."
            ),
            warnings=(
                ("Some expected counts are below 5. Interpret the chi-squared approximation carefully.",)
                if minimum_expected < 5
                else ()
            ),
        )
