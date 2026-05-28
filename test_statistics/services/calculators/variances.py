from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy import stats

from test_statistics.domain.enums import AlternativeHypothesis, TestFamily
from test_statistics.domain.inputs import NamedGroupsInput, OneSampleVarianceInput, TwoSampleVarianceInput
from test_statistics.domain.metadata import CalculatorMetadata
from test_statistics.domain.results import CalculationResult, DecisionSummary, ResultMetric, display_number, display_p_value, format_number
from test_statistics.services.calculators.base import BaseCalculator, NamedGroupCalculator, alpha_field, alternative_field, numeric_field, textarea_field
from test_statistics.services.validators import build_error, parse_alpha, parse_alternative, parse_float, parse_named_groups, parse_numeric_series, raise_if_issues



def _two_sided_tail_probability(cdf_value: float) -> float:
    return min(1.0, 2 * min(cdf_value, 1 - cdf_value))



def _decision(test_name: str, reject_null: bool) -> str:
    if reject_null:
        return f"Reject the null hypothesis for the {test_name}."
    return f"Fail to reject the null hypothesis for the {test_name}."


class ChiSquaredVarianceCalculator(BaseCalculator):
    metadata = CalculatorMetadata(
        catalog_position=17,
        slug="chi-squared-variance-test",
        name="Chi-Squared Test for Variance",
        family=TestFamily.VARIANCE,
        description="Compare a sample variance against a hypothesized population variance.",
        check="Whether one sample's variance differs from a reference variance.",
        statistic_formula="χ² = (n - 1)s² / σ₀²",
        assumptions=(
            "Observations are independent.",
            "The population is approximately normal.",
            "The null variance is specified in advance.",
        ),
        required_sample_data=(
            "One numeric sample.",
            "A hypothesized variance.",
        ),
        input_fields=(
            textarea_field("sample", "Sample values", "Enter comma-separated values for one numeric sample.", placeholder="12, 15, 14, 13, 16", rows=5),
            numeric_field("null_variance", "Hypothesized variance", "Enter the variance to test against.", placeholder="4", min_value="0"),
            alternative_field(),
            alpha_field(),
        ),
    )

    def normalize(self, raw_data: Mapping[str, Any]) -> OneSampleVarianceInput:
        issues = []
        sample, sample_issues = parse_numeric_series(raw_data.get("sample"), "sample", "Sample values")
        issues.extend(sample_issues)
        null_variance = parse_float(raw_data.get("null_variance"), "null_variance", "Hypothesized variance", issues)
        if null_variance is not None and null_variance <= 0:
            issues.append(build_error("null_variance", "Hypothesized variance must be greater than zero."))
        alternative, alternative_issues = parse_alternative(raw_data.get("alternative"))
        alpha, alpha_issues = parse_alpha(raw_data.get("alpha"))
        issues.extend(alternative_issues)
        issues.extend(alpha_issues)
        raise_if_issues(issues)
        return OneSampleVarianceInput(sample=sample, null_variance=null_variance, alternative=alternative, alpha=alpha)

    def calculate_result(self, normalized_input: OneSampleVarianceInput) -> CalculationResult:
        sample = np.asarray(normalized_input.sample, dtype=float)
        n = int(sample.size)
        df = n - 1
        sample_variance = float(sample.var(ddof=1))
        statistic = df * sample_variance / normalized_input.null_variance
        cdf_value = float(stats.chi2.cdf(statistic, df))
        if normalized_input.alternative is AlternativeHypothesis.TWO_SIDED:
            p_value = _two_sided_tail_probability(cdf_value)
        elif normalized_input.alternative is AlternativeHypothesis.GREATER:
            p_value = float(stats.chi2.sf(statistic, df))
        else:
            p_value = cdf_value
        ci_low = df * sample_variance / stats.chi2.ppf(1 - normalized_input.alpha / 2, df)
        ci_high = df * sample_variance / stats.chi2.ppf(normalized_input.alpha / 2, df)
        reject_null = p_value < normalized_input.alpha
        metrics = (
            ResultMetric("Sample size", str(n)),
            ResultMetric("Sample variance", format_number(sample_variance), emphasis=True),
            ResultMetric("Hypothesized variance", format_number(normalized_input.null_variance)),
            ResultMetric("Degrees of freedom", str(df)),
            ResultMetric(
                f"{int((1 - normalized_input.alpha) * 100)}% CI for the variance",
                f"[{format_number(float(ci_low))}, {format_number(float(ci_high))}]",
            ),
        )
        return CalculationResult(
            slug=self.metadata.slug,
            test_name=self.metadata.name,
            statistic_name="χ²",
            statistic=display_number(float(statistic)),
            p_value=display_p_value(float(p_value)),
            metrics=metrics,
            decision=DecisionSummary(alpha=normalized_input.alpha, reject_null=reject_null, conclusion=_decision(self.metadata.name, reject_null)),
            interpretation="The chi-squared test compares the observed sample variance against the hypothesized normal-theory variance.",
        )


class FVarianceCalculator(BaseCalculator):
    metadata = CalculatorMetadata(
        catalog_position=18,
        slug="f-test-for-variances",
        name="F Test for Variances",
        family=TestFamily.VARIANCE,
        description="Compare the variances of two independent samples.",
        check="Whether two independent samples have different variances.",
        statistic_formula="F = s₁² / s₂²",
        assumptions=(
            "The two samples are independent.",
            "Both populations are approximately normal.",
            "The samples are randomly drawn from their populations.",
        ),
        required_sample_data=(
            "One numeric sample for group A.",
            "One numeric sample for group B.",
        ),
        input_fields=(
            textarea_field("sample_a", "Group A values", "Enter comma-separated values for the first sample.", placeholder="10, 12, 13, 11, 9", rows=5),
            textarea_field("sample_b", "Group B values", "Enter comma-separated values for the second sample.", placeholder="16, 18, 15, 17, 19", rows=5),
            alternative_field(),
            alpha_field(),
        ),
    )

    def normalize(self, raw_data: Mapping[str, Any]) -> TwoSampleVarianceInput:
        issues = []
        sample_a, sample_a_issues = parse_numeric_series(raw_data.get("sample_a"), "sample_a", "Group A values")
        sample_b, sample_b_issues = parse_numeric_series(raw_data.get("sample_b"), "sample_b", "Group B values")
        issues.extend(sample_a_issues)
        issues.extend(sample_b_issues)
        alternative, alternative_issues = parse_alternative(raw_data.get("alternative"))
        alpha, alpha_issues = parse_alpha(raw_data.get("alpha"))
        issues.extend(alternative_issues)
        issues.extend(alpha_issues)
        raise_if_issues(issues)
        return TwoSampleVarianceInput(sample_a=sample_a, sample_b=sample_b, alternative=alternative, alpha=alpha)

    def calculate_result(self, normalized_input: TwoSampleVarianceInput) -> CalculationResult:
        sample_a = np.asarray(normalized_input.sample_a, dtype=float)
        sample_b = np.asarray(normalized_input.sample_b, dtype=float)
        variance_a = float(sample_a.var(ddof=1))
        variance_b = float(sample_b.var(ddof=1))
        df_a = int(sample_a.size - 1)
        df_b = int(sample_b.size - 1)
        statistic = variance_a / variance_b
        cdf_value = float(stats.f.cdf(statistic, df_a, df_b))
        if normalized_input.alternative is AlternativeHypothesis.TWO_SIDED:
            p_value = _two_sided_tail_probability(cdf_value)
        elif normalized_input.alternative is AlternativeHypothesis.GREATER:
            p_value = float(stats.f.sf(statistic, df_a, df_b))
        else:
            p_value = cdf_value
        lower = statistic / stats.f.ppf(1 - normalized_input.alpha / 2, df_a, df_b)
        upper = statistic / stats.f.ppf(normalized_input.alpha / 2, df_a, df_b)
        reject_null = p_value < normalized_input.alpha
        metrics = (
            ResultMetric("Group A variance", format_number(variance_a), emphasis=True),
            ResultMetric("Group B variance", format_number(variance_b), emphasis=True),
            ResultMetric("Variance ratio", format_number(statistic)),
            ResultMetric("Degrees of freedom", f"{df_a}, {df_b}"),
            ResultMetric(
                f"{int((1 - normalized_input.alpha) * 100)}% CI for the variance ratio",
                f"[{format_number(float(lower))}, {format_number(float(upper))}]",
            ),
        )
        return CalculationResult(
            slug=self.metadata.slug,
            test_name=self.metadata.name,
            statistic_name="F",
            statistic=display_number(float(statistic)),
            p_value=display_p_value(float(p_value)),
            metrics=metrics,
            decision=DecisionSummary(alpha=normalized_input.alpha, reject_null=reject_null, conclusion=_decision(self.metadata.name, reject_null)),
            interpretation="The F-statistic compares the sample variance ratio against what is expected under equal population variances.",
        )


class LeveneVarianceCalculator(NamedGroupCalculator):
    metadata = CalculatorMetadata(
        catalog_position=19,
        slug="levene-test-for-variances",
        name="Levene's Test for Variances",
        family=TestFamily.VARIANCE,
        description="Assess whether multiple groups have similar variances using a robust variance-equality test.",
        check="Whether group variances are equal across two or more groups.",
        statistic_formula="W = Levene's homogeneity-of-variance statistic",
        assumptions=(
            "Observations are independent within and across groups.",
            "The outcome is numeric.",
            "Groups are defined before examining the outcome.",
        ),
        required_sample_data=(
            "At least two named groups of numeric values.",
            "Each group should contain at least two observations.",
        ),
        input_fields=(
            textarea_field("groups", "Group samples", "Enter one group per line, such as 'Control: 4, 5, 6'.", placeholder="Control: 4, 5, 6\nTreatment A: 6, 7, 8\nTreatment B: 8, 9, 10", rows=7),
            alpha_field(),
        ),
    )

    def normalize(self, raw_data: Mapping[str, Any]) -> NamedGroupsInput:
        issues = []
        groups, group_issues = parse_named_groups(raw_data.get("groups"), "groups")
        alpha, alpha_issues = parse_alpha(raw_data.get("alpha"))
        issues.extend(group_issues)
        issues.extend(alpha_issues)
        raise_if_issues(issues)
        return NamedGroupsInput(groups=groups, alpha=alpha)

    def calculate_result(self, normalized_input: NamedGroupsInput) -> CalculationResult:
        arrays = [np.asarray(values, dtype=float) for _, values in normalized_input.groups]
        result = stats.levene(*arrays, center="median")
        variances = "; ".join(
            f"{group_name}: {format_number(float(np.asarray(values, dtype=float).var(ddof=1)))}"
            for group_name, values in normalized_input.groups
        )
        p_value = float(result.pvalue)
        reject_null = p_value < normalized_input.alpha
        metrics = (
            ResultMetric("Groups", str(len(arrays))),
            ResultMetric("Group variances", variances, emphasis=True),
            ResultMetric("Total observations", str(sum(array.size for array in arrays))),
        )
        return CalculationResult(
            slug=self.metadata.slug,
            test_name=self.metadata.name,
            statistic_name="W",
            statistic=display_number(float(result.statistic)),
            p_value=display_p_value(p_value),
            metrics=metrics,
            decision=DecisionSummary(alpha=normalized_input.alpha, reject_null=reject_null, conclusion=_decision(self.metadata.name, reject_null)),
            interpretation="Levene's test is centered on group medians here, which makes it more robust to non-normality.",
        )
