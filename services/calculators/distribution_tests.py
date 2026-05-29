from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy import stats

from domain.enums import InputKind, TestFamily
from domain.inputs import GoodnessOfFitInput, OneSampleKsInput, ShapiroWilkInput, TwoSampleKsInput
from domain.metadata import CalculatorMetadata, InputFieldDefinition
from domain.results import CalculationResult, DecisionSummary, ResultMetric, ResultTable, display_number, display_p_value, format_number
from services.calculators.base import BaseCalculator, KS_DISTRIBUTION_OPTIONS, alpha_field, alternative_field, textarea_field, text_field
from services.validators import parse_alpha, parse_alternative, parse_ks_distribution, parse_numeric_series, parse_observed_expected, raise_if_issues



def _decision(test_name: str, reject_null: bool) -> str:
    if reject_null:
        return f"Reject the null hypothesis for the {test_name}."
    return f"Fail to reject the null hypothesis for the {test_name}."


class ChiSquaredGoodnessOfFitCalculator(BaseCalculator):
    metadata = CalculatorMetadata(
        catalog_position=20,
        slug="chi-squared-goodness-of-fit-test",
        name="Chi-Squared Test for Goodness of Fit",
        family=TestFamily.DISTRIBUTION,
        description="Compare observed category counts against expected counts or probabilities.",
        check="Whether observed category frequencies follow a specified distribution.",
        statistic_formula="χ² = Σ (O - E)² / E",
        assumptions=(
            "Observations are independent.",
            "Categories are mutually exclusive.",
            "Expected counts are large enough for the chi-squared approximation.",
        ),
        required_sample_data=(
            "Observed category counts.",
            "Expected counts or probabilities with the same category order.",
        ),
        input_fields=(
            textarea_field("observed", "Observed counts", "Enter comma-separated observed counts.", placeholder="18, 22, 20", rows=4),
            textarea_field("expected", "Expected counts or probabilities", "Enter comma-separated expected counts or probabilities. Probabilities should sum to 1.", placeholder="20, 20, 20", rows=4),
            alpha_field(),
        ),
    )

    def normalize(self, raw_data: Mapping[str, Any]) -> GoodnessOfFitInput:
        observed, expected, issues = parse_observed_expected(raw_data.get("observed"), raw_data.get("expected"))
        alpha, alpha_issues = parse_alpha(raw_data.get("alpha"))
        issues.extend(alpha_issues)
        raise_if_issues(issues)
        return GoodnessOfFitInput(observed=observed, expected=expected, alpha=alpha)

    def calculate_result(self, normalized_input: GoodnessOfFitInput) -> CalculationResult:
        result = stats.chisquare(f_obs=normalized_input.observed, f_exp=normalized_input.expected)
        expected_min = min(normalized_input.expected)
        p_value = float(result.pvalue)
        reject_null = p_value < normalized_input.alpha
        table = ResultTable(
            title="Observed vs expected",
            columns=("Category", "Observed", "Expected"),
            rows=tuple(
                (
                    str(index + 1),
                    format_number(float(observed)),
                    format_number(float(expected)),
                )
                for index, (observed, expected) in enumerate(zip(normalized_input.observed, normalized_input.expected, strict=True))
            ),
        )
        warnings = ()
        if expected_min < 5:
            warnings = ("Some expected counts are below 5. Interpret the chi-squared approximation carefully.",)
        return CalculationResult(
            slug=self.metadata.slug,
            test_name=self.metadata.name,
            statistic_name="χ²",
            statistic=display_number(float(result.statistic)),
            p_value=display_p_value(p_value),
            metrics=(
                ResultMetric("Categories", str(len(normalized_input.observed))),
                ResultMetric("Minimum expected count", format_number(float(expected_min)), emphasis=True),
                ResultMetric("Degrees of freedom", str(len(normalized_input.observed) - 1)),
            ),
            tables=(table,),
            decision=DecisionSummary(alpha=normalized_input.alpha, reject_null=reject_null, conclusion=_decision(self.metadata.name, reject_null)),
            interpretation="The test compares the observed frequency pattern with the expected pattern category by category.",
            warnings=warnings,
        )


class ShapiroWilkCalculator(BaseCalculator):
    metadata = CalculatorMetadata(
        catalog_position=21,
        slug="shapiro-wilk-test",
        name="Shapiro-Wilk Test",
        family=TestFamily.DISTRIBUTION,
        description="Assess whether a numeric sample is plausibly drawn from a normal distribution.",
        check="Whether one sample departs from normality.",
        statistic_formula="W = ordered-normality statistic",
        assumptions=(
            "Observations are independent.",
            "The outcome is continuous.",
            "The sample is not too large for the Shapiro-Wilk approximation.",
        ),
        required_sample_data=("One numeric sample.",),
        input_fields=(
            textarea_field("sample", "Sample values", "Enter comma-separated values for one numeric sample.", placeholder="12, 15, 14, 13, 16", rows=5),
            alpha_field(),
        ),
    )

    def normalize(self, raw_data: Mapping[str, Any]) -> ShapiroWilkInput:
        issues = []
        sample, sample_issues = parse_numeric_series(raw_data.get("sample"), "sample", "Sample values", minimum_length=3)
        alpha, alpha_issues = parse_alpha(raw_data.get("alpha"))
        issues.extend(sample_issues)
        issues.extend(alpha_issues)
        raise_if_issues(issues)
        return ShapiroWilkInput(sample=sample, alpha=alpha)

    def calculate_result(self, normalized_input: ShapiroWilkInput) -> CalculationResult:
        result = stats.shapiro(normalized_input.sample)
        p_value = float(result.pvalue)
        reject_null = p_value < normalized_input.alpha
        sample_array = np.asarray(normalized_input.sample, dtype=float)
        return CalculationResult(
            slug=self.metadata.slug,
            test_name=self.metadata.name,
            statistic_name="W",
            statistic=display_number(float(result.statistic)),
            p_value=display_p_value(p_value),
            metrics=(
                ResultMetric("Sample size", str(len(normalized_input.sample))),
                ResultMetric("Sample mean", format_number(float(sample_array.mean())), emphasis=True),
                ResultMetric("Sample standard deviation", format_number(float(sample_array.std(ddof=1)))),
            ),
            decision=DecisionSummary(alpha=normalized_input.alpha, reject_null=reject_null, conclusion=_decision(self.metadata.name, reject_null)),
            interpretation="Small p-values suggest the sample does not look consistent with a normal distribution.",
        )


class OneSampleKsCalculator(BaseCalculator):
    metadata = CalculatorMetadata(
        catalog_position=22,
        slug="one-sample-kolmogorov-smirnov-test",
        name="Kolmogorov-Smirnov Test",
        family=TestFamily.DISTRIBUTION,
        description="Compare a numeric sample against a named theoretical distribution.",
        check="Whether one sample differs from a specified reference distribution.",
        statistic_formula="D = max |Fₙ(x) - F₀(x)|",
        assumptions=(
            "Observations are independent.",
            "The reference distribution is fully specified.",
            "The sample is numeric and ordered on a continuous scale.",
        ),
        required_sample_data=(
            "One numeric sample.",
            "A reference distribution and its parameters.",
        ),
        input_fields=(
            textarea_field("sample", "Sample values", "Enter comma-separated values for one numeric sample.", placeholder="-0.5, 0.1, 0.2, 0.8, 1.1", rows=5),
            InputFieldDefinition(
                name="distribution",
                label="Reference distribution",
                kind=InputKind.SELECT,
                help_text="Choose the reference distribution for the one-sample KS test.",
                default_value="norm",
                options=KS_DISTRIBUTION_OPTIONS,
            ),
            text_field("distribution_parameters", "Distribution parameters", "Enter comma-separated parameters as 'location, scale'. Leave blank to use 0, 1.", placeholder="0, 1"),
            alternative_field(help_text="Choose the alternative cumulative-distribution relationship."),
            alpha_field(),
        ),
    )

    def normalize(self, raw_data: Mapping[str, Any]) -> OneSampleKsInput:
        issues = []
        sample, sample_issues = parse_numeric_series(raw_data.get("sample"), "sample", "Sample values", minimum_length=3)
        distribution, parameters, distribution_issues = parse_ks_distribution(
            raw_data.get("distribution"),
            raw_data.get("distribution_parameters"),
        )
        alternative, alternative_issues = parse_alternative(raw_data.get("alternative"))
        alpha, alpha_issues = parse_alpha(raw_data.get("alpha"))
        issues.extend(sample_issues)
        issues.extend(distribution_issues)
        issues.extend(alternative_issues)
        issues.extend(alpha_issues)
        raise_if_issues(issues)
        return OneSampleKsInput(sample=sample, distribution=distribution, parameters=parameters, alternative=alternative, alpha=alpha)

    def calculate_result(self, normalized_input: OneSampleKsInput) -> CalculationResult:
        result = stats.kstest(
            normalized_input.sample,
            normalized_input.distribution,
            args=normalized_input.parameters,
            alternative=normalized_input.alternative.value,
        )
        p_value = float(result.pvalue)
        reject_null = p_value < normalized_input.alpha
        metrics = (
            ResultMetric("Sample size", str(len(normalized_input.sample))),
            ResultMetric("Reference distribution", normalized_input.distribution, emphasis=True),
            ResultMetric(
                "Distribution parameters",
                ", ".join(format_number(float(parameter)) for parameter in normalized_input.parameters),
            ),
        )
        notes = ()
        statistic_location = getattr(result, "statistic_location", None)
        if statistic_location is not None:
            notes = (f"Maximum deviation occurred near x = {format_number(float(statistic_location))}.",)
        return CalculationResult(
            slug=self.metadata.slug,
            test_name=self.metadata.name,
            statistic_name="D",
            statistic=display_number(float(result.statistic)),
            p_value=display_p_value(p_value),
            metrics=metrics,
            decision=DecisionSummary(alpha=normalized_input.alpha, reject_null=reject_null, conclusion=_decision(self.metadata.name, reject_null)),
            interpretation="The one-sample KS statistic captures the largest gap between the empirical and specified cumulative distributions.",
            notes=notes,
        )


class TwoSampleKsCalculator(BaseCalculator):
    metadata = CalculatorMetadata(
        catalog_position=23,
        slug="two-sample-kolmogorov-smirnov-test",
        name="Two Sample Kolmogorov-Smirnov Test",
        family=TestFamily.DISTRIBUTION,
        description="Compare two independent numeric samples without assuming a particular parametric distribution.",
        check="Whether two samples follow the same distribution.",
        statistic_formula="D = max |Fₙ(x) - Gₘ(x)|",
        assumptions=(
            "The two samples are independent.",
            "The outcome is ordered on at least an ordinal scale.",
            "The test is sensitive to location, spread, and shape differences.",
        ),
        required_sample_data=(
            "One numeric sample for group A.",
            "One numeric sample for group B.",
        ),
        input_fields=(
            textarea_field("sample_a", "Group A values", "Enter comma-separated values for the first sample.", placeholder="1, 2, 3, 4, 5", rows=5),
            textarea_field("sample_b", "Group B values", "Enter comma-separated values for the second sample.", placeholder="2, 3, 4, 5, 6", rows=5),
            alternative_field(help_text="Choose the alternative cumulative-distribution relationship."),
            alpha_field(),
        ),
    )

    def normalize(self, raw_data: Mapping[str, Any]) -> TwoSampleKsInput:
        issues = []
        sample_a, sample_a_issues = parse_numeric_series(raw_data.get("sample_a"), "sample_a", "Group A values", minimum_length=2)
        sample_b, sample_b_issues = parse_numeric_series(raw_data.get("sample_b"), "sample_b", "Group B values", minimum_length=2)
        alternative, alternative_issues = parse_alternative(raw_data.get("alternative"))
        alpha, alpha_issues = parse_alpha(raw_data.get("alpha"))
        issues.extend(sample_a_issues)
        issues.extend(sample_b_issues)
        issues.extend(alternative_issues)
        issues.extend(alpha_issues)
        raise_if_issues(issues)
        return TwoSampleKsInput(sample_a=sample_a, sample_b=sample_b, alternative=alternative, alpha=alpha)

    def calculate_result(self, normalized_input: TwoSampleKsInput) -> CalculationResult:
        result = stats.ks_2samp(normalized_input.sample_a, normalized_input.sample_b, alternative=normalized_input.alternative.value)
        p_value = float(result.pvalue)
        reject_null = p_value < normalized_input.alpha
        metrics = (
            ResultMetric("Group A size", str(len(normalized_input.sample_a))),
            ResultMetric("Group B size", str(len(normalized_input.sample_b))),
            ResultMetric("Group A median", format_number(float(np.median(np.asarray(normalized_input.sample_a, dtype=float)))), emphasis=True),
            ResultMetric("Group B median", format_number(float(np.median(np.asarray(normalized_input.sample_b, dtype=float)))), emphasis=True),
        )
        notes = ()
        statistic_location = getattr(result, "statistic_location", None)
        if statistic_location is not None:
            notes = (f"Maximum empirical-distribution gap occurred near x = {format_number(float(statistic_location))}.",)
        return CalculationResult(
            slug=self.metadata.slug,
            test_name=self.metadata.name,
            statistic_name="D",
            statistic=display_number(float(result.statistic)),
            p_value=display_p_value(p_value),
            metrics=metrics,
            decision=DecisionSummary(alpha=normalized_input.alpha, reject_null=reject_null, conclusion=_decision(self.metadata.name, reject_null)),
            interpretation="The two-sample KS statistic summarizes the largest empirical-distribution gap between the two samples.",
            notes=notes,
        )
