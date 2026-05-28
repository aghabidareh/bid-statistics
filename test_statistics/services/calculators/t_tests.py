from __future__ import annotations

from collections.abc import Mapping
from math import sqrt
from typing import Any

import numpy as np
from scipy import stats
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW

from test_statistics.domain.enums import AlternativeHypothesis, TestFamily
from test_statistics.domain.inputs import OneSampleTTestInput, OneSampleZTestInput, PairedTTestInput, TwoSampleTTestInput, TwoSampleZTestInput, WelchTTestInput
from test_statistics.domain.metadata import CalculatorMetadata
from test_statistics.domain.results import CalculationResult, DecisionSummary, ResultMetric, display_number, display_p_value, format_number
from test_statistics.services.calculators.base import SingleSampleCalculator, TwoIndependentSampleCalculator, PairedSampleCalculator, alpha_field, alternative_field, numeric_field, textarea_field
from test_statistics.services.validators import parse_alpha, parse_alternative, parse_float, parse_numeric_series, parse_paired_numeric_samples, parse_positive_float, raise_if_issues



def _directional_decision(test_name: str, reject_null: bool, alternative: AlternativeHypothesis) -> str:
    if not reject_null:
        return f"Fail to reject the null hypothesis for the {test_name}."
    if alternative is AlternativeHypothesis.TWO_SIDED:
        return f"Reject the null hypothesis. The data provide evidence of a difference for the {test_name}."
    direction = "greater" if alternative is AlternativeHypothesis.GREATER else "less"
    return f"Reject the null hypothesis. The data support the '{direction}' direction for the {test_name}."



def _single_sample_fields(*, sample_placeholder: str, include_known_std: bool = False) -> tuple:
    fields = [
        textarea_field(
            "sample",
            "Sample values",
            "Enter comma-separated values for one numeric sample.",
            placeholder=sample_placeholder,
            rows=5,
        ),
        numeric_field(
            "population_mean",
            "Hypothesized mean",
            "Enter the mean to test against.",
            placeholder="0",
            default_value="0",
        ),
    ]
    if include_known_std:
        fields.append(
            numeric_field(
                "known_std",
                "Known population standard deviation",
                "Enter the known population standard deviation.",
                placeholder="2.5",
                min_value="0",
            )
        )
    fields.extend((alternative_field(), alpha_field()))
    return tuple(fields)



def _two_sample_fields(*, include_known_stds: bool = False) -> tuple:
    fields = [
        textarea_field(
            "sample_a",
            "Group A values",
            "Enter comma-separated values for the first sample.",
            placeholder="10, 12, 13, 11, 9",
            rows=5,
        ),
        textarea_field(
            "sample_b",
            "Group B values",
            "Enter comma-separated values for the second sample.",
            placeholder="16, 18, 15, 17, 19",
            rows=5,
        ),
    ]
    if include_known_stds:
        fields.extend(
            (
                numeric_field(
                    "known_std_a",
                    "Known standard deviation for group A",
                    "Enter the known population standard deviation for group A.",
                    placeholder="2",
                    min_value="0",
                ),
                numeric_field(
                    "known_std_b",
                    "Known standard deviation for group B",
                    "Enter the known population standard deviation for group B.",
                    placeholder="2",
                    min_value="0",
                ),
            )
        )
    fields.extend((alternative_field(), alpha_field()))
    return tuple(fields)


class OneSampleZTestCalculator(SingleSampleCalculator):
    metadata = CalculatorMetadata(
        catalog_position=1,
        slug="one-sample-z-test",
        name="One Sample Z-Test",
        family=TestFamily.PARAMETRIC,
        description="Compare a sample mean against a hypothesized mean when the population standard deviation is known.",
        check="Whether one sample mean differs from a reference mean using a z-statistic.",
        statistic_formula="z = (x̄ - μ₀) / (σ / √n)",
        assumptions=(
            "Observations are independent.",
            "The outcome is approximately continuous.",
            "The population standard deviation is known.",
        ),
        required_sample_data=(
            "One numeric sample.",
            "A hypothesized population mean.",
            "A known population standard deviation.",
        ),
        input_fields=_single_sample_fields(sample_placeholder="12, 15, 14, 13, 16", include_known_std=True),
    )

    def normalize(self, raw_data: Mapping[str, Any]) -> OneSampleZTestInput:
        issues = []
        sample, sample_issues = parse_numeric_series(raw_data.get("sample"), "sample", "Sample values")
        issues.extend(sample_issues)
        null_mean = parse_float(raw_data.get("population_mean"), "population_mean", "Hypothesized mean", issues)
        known_std = parse_positive_float(raw_data.get("known_std"), "known_std", "Known population standard deviation", issues)
        alternative, alternative_issues = parse_alternative(raw_data.get("alternative"))
        alpha, alpha_issues = parse_alpha(raw_data.get("alpha"))
        issues.extend(alternative_issues)
        issues.extend(alpha_issues)
        raise_if_issues(issues)
        return OneSampleZTestInput(sample=sample, null_mean=null_mean, known_std=known_std, alternative=alternative, alpha=alpha)

    def calculate_result(self, normalized_input: OneSampleZTestInput) -> CalculationResult:
        sample = np.asarray(normalized_input.sample, dtype=float)
        n = int(sample.size)
        sample_mean = float(sample.mean())
        sample_std = float(sample.std(ddof=1))
        standard_error = normalized_input.known_std / sqrt(n)
        statistic = (sample_mean - normalized_input.null_mean) / standard_error
        if normalized_input.alternative is AlternativeHypothesis.TWO_SIDED:
            p_value = 2 * stats.norm.sf(abs(statistic))
        elif normalized_input.alternative is AlternativeHypothesis.GREATER:
            p_value = stats.norm.sf(statistic)
        else:
            p_value = stats.norm.cdf(statistic)
        ci_low, ci_high = stats.norm.interval(1 - normalized_input.alpha, loc=sample_mean, scale=standard_error)
        reject_null = p_value < normalized_input.alpha
        metrics = (
            ResultMetric("Sample size", str(n)),
            ResultMetric("Sample mean", format_number(sample_mean), emphasis=True),
            ResultMetric("Hypothesized mean", format_number(normalized_input.null_mean)),
            ResultMetric("Known population standard deviation", format_number(normalized_input.known_std)),
            ResultMetric("Observed sample standard deviation", format_number(sample_std)),
            ResultMetric("Standard error", format_number(standard_error)),
            ResultMetric(
                f"{int((1 - normalized_input.alpha) * 100)}% CI for the mean",
                f"[{format_number(float(ci_low))}, {format_number(float(ci_high))}]",
            ),
        )
        return CalculationResult(
            slug=self.metadata.slug,
            test_name=self.metadata.name,
            statistic_name="z",
            statistic=display_number(float(statistic)),
            p_value=display_p_value(float(p_value)),
            metrics=metrics,
            decision=DecisionSummary(
                alpha=normalized_input.alpha,
                reject_null=reject_null,
                conclusion=_directional_decision(self.metadata.name, reject_null, normalized_input.alternative),
            ),
            interpretation=(
                f"The sample mean is {format_number(sample_mean)} versus the hypothesized mean of "
                f"{format_number(normalized_input.null_mean)} when σ = {format_number(normalized_input.known_std)} is treated as known."
            ),
        )


class OneSampleTTestCalculator(SingleSampleCalculator):
    metadata = CalculatorMetadata(
        catalog_position=2,
        slug="one-sample-t-test",
        name="One Sample T-Test",
        family=TestFamily.PARAMETRIC,
        description="Compare a sample mean against a hypothesized population mean when the standard deviation is estimated from the sample.",
        check="Whether one sample mean differs from a reference mean.",
        statistic_formula="t = (x̄ - μ₀) / (s / √n)",
        assumptions=(
            "Observations are independent.",
            "The outcome is approximately continuous.",
            "The sampled population is approximately normal for small samples.",
        ),
        required_sample_data=(
            "One numeric sample.",
            "A hypothesized population mean.",
        ),
        input_fields=_single_sample_fields(sample_placeholder="12, 15, 14, 13, 16"),
    )

    def normalize(self, raw_data: Mapping[str, Any]) -> OneSampleTTestInput:
        issues = []
        sample, sample_issues = parse_numeric_series(raw_data.get("sample"), "sample", "Sample values")
        issues.extend(sample_issues)
        null_mean = parse_float(raw_data.get("population_mean"), "population_mean", "Hypothesized mean", issues)
        alternative, alternative_issues = parse_alternative(raw_data.get("alternative"))
        alpha, alpha_issues = parse_alpha(raw_data.get("alpha"))
        issues.extend(alternative_issues)
        issues.extend(alpha_issues)
        raise_if_issues(issues)
        return OneSampleTTestInput(sample=sample, null_mean=null_mean, alternative=alternative, alpha=alpha)

    def calculate_result(self, normalized_input: OneSampleTTestInput) -> CalculationResult:
        sample = np.asarray(normalized_input.sample, dtype=float)
        n = int(sample.size)
        df = n - 1
        sample_mean = float(sample.mean())
        sample_std = float(sample.std(ddof=1))
        standard_error = sample_std / sqrt(n)
        test_result = stats.ttest_1samp(sample, popmean=normalized_input.null_mean, alternative=normalized_input.alternative.value)
        ci_low, ci_high = stats.t.interval(1 - normalized_input.alpha, df=df, loc=sample_mean, scale=standard_error)
        statistic = float(test_result.statistic)
        p_value = float(test_result.pvalue)
        reject_null = p_value < normalized_input.alpha
        metrics = (
            ResultMetric("Sample size", str(n)),
            ResultMetric("Sample mean", format_number(sample_mean), emphasis=True),
            ResultMetric("Hypothesized mean", format_number(normalized_input.null_mean)),
            ResultMetric("Mean difference", format_number(sample_mean - normalized_input.null_mean)),
            ResultMetric("Sample standard deviation", format_number(sample_std)),
            ResultMetric("Standard error", format_number(standard_error)),
            ResultMetric("Degrees of freedom", str(df)),
            ResultMetric(
                f"{int((1 - normalized_input.alpha) * 100)}% CI for the mean",
                f"[{format_number(float(ci_low))}, {format_number(float(ci_high))}]",
            ),
        )
        return CalculationResult(
            slug=self.metadata.slug,
            test_name=self.metadata.name,
            statistic_name="t",
            statistic=display_number(statistic),
            p_value=display_p_value(p_value),
            metrics=metrics,
            decision=DecisionSummary(
                alpha=normalized_input.alpha,
                reject_null=reject_null,
                conclusion=_directional_decision(self.metadata.name, reject_null, normalized_input.alternative),
            ),
            interpretation=(
                f"The sample mean is {format_number(sample_mean)} compared with the hypothesized mean of "
                f"{format_number(normalized_input.null_mean)}."
            ),
        )


class TwoSampleZTestCalculator(TwoIndependentSampleCalculator):
    metadata = CalculatorMetadata(
        catalog_position=3,
        slug="two-sample-z-test",
        name="Two Sample Z-Test",
        family=TestFamily.PARAMETRIC,
        description="Compare two independent sample means when both population standard deviations are known.",
        check="Whether two independent sample means differ using a z-statistic.",
        statistic_formula="z = (x̄₁ - x̄₂) / √(σ₁² / n₁ + σ₂² / n₂)",
        assumptions=(
            "The two samples are independent.",
            "The outcome is approximately continuous.",
            "Both population standard deviations are known.",
        ),
        required_sample_data=(
            "One numeric sample for group A.",
            "One numeric sample for group B.",
            "Known population standard deviations for both groups.",
        ),
        input_fields=_two_sample_fields(include_known_stds=True),
    )

    def normalize(self, raw_data: Mapping[str, Any]) -> TwoSampleZTestInput:
        issues = []
        sample_a, sample_a_issues = parse_numeric_series(raw_data.get("sample_a"), "sample_a", "Group A values")
        sample_b, sample_b_issues = parse_numeric_series(raw_data.get("sample_b"), "sample_b", "Group B values")
        issues.extend(sample_a_issues)
        issues.extend(sample_b_issues)
        known_std_a = parse_positive_float(raw_data.get("known_std_a"), "known_std_a", "Known standard deviation for group A", issues)
        known_std_b = parse_positive_float(raw_data.get("known_std_b"), "known_std_b", "Known standard deviation for group B", issues)
        alternative, alternative_issues = parse_alternative(raw_data.get("alternative"))
        alpha, alpha_issues = parse_alpha(raw_data.get("alpha"))
        issues.extend(alternative_issues)
        issues.extend(alpha_issues)
        raise_if_issues(issues)
        return TwoSampleZTestInput(
            sample_a=sample_a,
            sample_b=sample_b,
            known_std_a=known_std_a,
            known_std_b=known_std_b,
            alternative=alternative,
            alpha=alpha,
        )

    def calculate_result(self, normalized_input: TwoSampleZTestInput) -> CalculationResult:
        sample_a = np.asarray(normalized_input.sample_a, dtype=float)
        sample_b = np.asarray(normalized_input.sample_b, dtype=float)
        n_a = int(sample_a.size)
        n_b = int(sample_b.size)
        mean_a = float(sample_a.mean())
        mean_b = float(sample_b.mean())
        diff = mean_a - mean_b
        standard_error = sqrt((normalized_input.known_std_a**2 / n_a) + (normalized_input.known_std_b**2 / n_b))
        statistic = diff / standard_error
        if normalized_input.alternative is AlternativeHypothesis.TWO_SIDED:
            p_value = 2 * stats.norm.sf(abs(statistic))
        elif normalized_input.alternative is AlternativeHypothesis.GREATER:
            p_value = stats.norm.sf(statistic)
        else:
            p_value = stats.norm.cdf(statistic)
        ci_low, ci_high = stats.norm.interval(1 - normalized_input.alpha, loc=diff, scale=standard_error)
        reject_null = p_value < normalized_input.alpha
        metrics = (
            ResultMetric("Group A size", str(n_a)),
            ResultMetric("Group B size", str(n_b)),
            ResultMetric("Group A mean", format_number(mean_a), emphasis=True),
            ResultMetric("Group B mean", format_number(mean_b), emphasis=True),
            ResultMetric("Known standard deviation for group A", format_number(normalized_input.known_std_a)),
            ResultMetric("Known standard deviation for group B", format_number(normalized_input.known_std_b)),
            ResultMetric("Mean difference", format_number(diff)),
            ResultMetric("Standard error", format_number(standard_error)),
            ResultMetric(
                f"{int((1 - normalized_input.alpha) * 100)}% CI for the mean difference",
                f"[{format_number(float(ci_low))}, {format_number(float(ci_high))}]",
            ),
        )
        return CalculationResult(
            slug=self.metadata.slug,
            test_name=self.metadata.name,
            statistic_name="z",
            statistic=display_number(float(statistic)),
            p_value=display_p_value(float(p_value)),
            metrics=metrics,
            decision=DecisionSummary(
                alpha=normalized_input.alpha,
                reject_null=reject_null,
                conclusion=_directional_decision(self.metadata.name, reject_null, normalized_input.alternative),
            ),
            interpretation=(
                f"Group A has mean {format_number(mean_a)} and Group B has mean {format_number(mean_b)} under known-population standard deviations."
            ),
        )


class TwoSamplePooledTTestCalculator(TwoIndependentSampleCalculator):
    metadata = CalculatorMetadata(
        catalog_position=4,
        slug="two-sample-t-test-pooled",
        name="Two Sample T-Test (Pooled Variance)",
        family=TestFamily.PARAMETRIC,
        description="Compare two independent sample means under an equal-variance assumption.",
        check="Whether two independent sample means differ when a pooled variance estimate is appropriate.",
        statistic_formula="t = (x̄₁ - x̄₂) / (sₚ √(1/n₁ + 1/n₂))",
        assumptions=(
            "The two samples are independent.",
            "The outcome is approximately continuous.",
            "The two populations have approximately equal variances.",
        ),
        required_sample_data=(
            "One numeric sample for group A.",
            "One numeric sample for group B.",
        ),
        input_fields=_two_sample_fields(),
    )

    def normalize(self, raw_data: Mapping[str, Any]) -> TwoSampleTTestInput:
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
        return TwoSampleTTestInput(sample_a=sample_a, sample_b=sample_b, alternative=alternative, alpha=alpha)

    def calculate_result(self, normalized_input: TwoSampleTTestInput) -> CalculationResult:
        sample_a = np.asarray(normalized_input.sample_a, dtype=float)
        sample_b = np.asarray(normalized_input.sample_b, dtype=float)
        n_a = int(sample_a.size)
        n_b = int(sample_b.size)
        mean_a = float(sample_a.mean())
        mean_b = float(sample_b.mean())
        var_a = float(sample_a.var(ddof=1))
        var_b = float(sample_b.var(ddof=1))
        pooled_variance = (((n_a - 1) * var_a) + ((n_b - 1) * var_b)) / (n_a + n_b - 2)
        pooled_std = sqrt(pooled_variance)
        standard_error = pooled_std * sqrt((1 / n_a) + (1 / n_b))
        statistic = (mean_a - mean_b) / standard_error
        df = n_a + n_b - 2
        if normalized_input.alternative is AlternativeHypothesis.TWO_SIDED:
            p_value = 2 * stats.t.sf(abs(statistic), df)
        elif normalized_input.alternative is AlternativeHypothesis.GREATER:
            p_value = stats.t.sf(statistic, df)
        else:
            p_value = stats.t.cdf(statistic, df)
        ci_low, ci_high = stats.t.interval(1 - normalized_input.alpha, df=df, loc=mean_a - mean_b, scale=standard_error)
        reject_null = p_value < normalized_input.alpha
        metrics = (
            ResultMetric("Group A size", str(n_a)),
            ResultMetric("Group B size", str(n_b)),
            ResultMetric("Group A mean", format_number(mean_a), emphasis=True),
            ResultMetric("Group B mean", format_number(mean_b), emphasis=True),
            ResultMetric("Pooled standard deviation", format_number(pooled_std)),
            ResultMetric("Mean difference", format_number(mean_a - mean_b)),
            ResultMetric("Standard error", format_number(standard_error)),
            ResultMetric("Degrees of freedom", str(df)),
            ResultMetric(
                f"{int((1 - normalized_input.alpha) * 100)}% CI for the mean difference",
                f"[{format_number(float(ci_low))}, {format_number(float(ci_high))}]",
            ),
        )
        return CalculationResult(
            slug=self.metadata.slug,
            test_name=self.metadata.name,
            statistic_name="t",
            statistic=display_number(float(statistic)),
            p_value=display_p_value(float(p_value)),
            metrics=metrics,
            decision=DecisionSummary(
                alpha=normalized_input.alpha,
                reject_null=reject_null,
                conclusion=_directional_decision(self.metadata.name, reject_null, normalized_input.alternative),
            ),
            interpretation=(
                f"Group A has mean {format_number(mean_a)} and Group B has mean {format_number(mean_b)} under a pooled-variance assumption."
            ),
        )


class WelchTTestCalculator(TwoIndependentSampleCalculator):
    metadata = CalculatorMetadata(
        catalog_position=5,
        slug="two-sample-t-test-welch",
        name="Two Sample T-Test (Welch's)",
        family=TestFamily.PARAMETRIC,
        description="Compare two independent sample means without assuming equal variances.",
        check="Whether two independent sample means differ when variances may be unequal.",
        statistic_formula="t = (x̄₁ - x̄₂) / √(s₁² / n₁ + s₂² / n₂)",
        assumptions=(
            "The two samples are independent.",
            "The outcome is approximately continuous.",
            "Each group is approximately normal for small samples.",
        ),
        required_sample_data=(
            "One numeric sample for group A.",
            "One numeric sample for group B.",
        ),
        input_fields=_two_sample_fields(),
    )

    def normalize(self, raw_data: Mapping[str, Any]) -> WelchTTestInput:
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
        return WelchTTestInput(sample_a=sample_a, sample_b=sample_b, alternative=alternative, alpha=alpha)

    def calculate_result(self, normalized_input: WelchTTestInput) -> CalculationResult:
        sample_a = np.asarray(normalized_input.sample_a, dtype=float)
        sample_b = np.asarray(normalized_input.sample_b, dtype=float)
        test_result = stats.ttest_ind(sample_a, sample_b, equal_var=False, alternative=normalized_input.alternative.value)
        group_a_stats = DescrStatsW(sample_a)
        group_b_stats = DescrStatsW(sample_b)
        compare_means = CompareMeans(group_a_stats, group_b_stats)
        ci_low, ci_high = compare_means.tconfint_diff(alpha=normalized_input.alpha, usevar="unequal")
        n_a = int(sample_a.size)
        n_b = int(sample_b.size)
        mean_a = float(sample_a.mean())
        mean_b = float(sample_b.mean())
        variance_a = float(sample_a.var(ddof=1))
        variance_b = float(sample_b.var(ddof=1))
        standard_error = sqrt((variance_a / n_a) + (variance_b / n_b))
        numerator = (variance_a / n_a + variance_b / n_b) ** 2
        denominator = ((variance_a / n_a) ** 2) / (n_a - 1) + ((variance_b / n_b) ** 2) / (n_b - 1)
        df = numerator / denominator
        statistic = float(test_result.statistic)
        p_value = float(test_result.pvalue)
        reject_null = p_value < normalized_input.alpha
        metrics = (
            ResultMetric("Group A size", str(n_a)),
            ResultMetric("Group B size", str(n_b)),
            ResultMetric("Group A mean", format_number(mean_a), emphasis=True),
            ResultMetric("Group B mean", format_number(mean_b), emphasis=True),
            ResultMetric("Mean difference", format_number(mean_a - mean_b)),
            ResultMetric("Standard error", format_number(standard_error)),
            ResultMetric("Welch degrees of freedom", format_number(df)),
            ResultMetric(
                f"{int((1 - normalized_input.alpha) * 100)}% CI for the mean difference",
                f"[{format_number(float(ci_low))}, {format_number(float(ci_high))}]",
            ),
        )
        return CalculationResult(
            slug=self.metadata.slug,
            test_name=self.metadata.name,
            statistic_name="t",
            statistic=display_number(statistic),
            p_value=display_p_value(p_value),
            metrics=metrics,
            decision=DecisionSummary(
                alpha=normalized_input.alpha,
                reject_null=reject_null,
                conclusion=_directional_decision(self.metadata.name, reject_null, normalized_input.alternative),
            ),
            interpretation=f"Group A has mean {format_number(mean_a)} and Group B has mean {format_number(mean_b)}.",
        )


class PairedTTestCalculator(PairedSampleCalculator):
    metadata = CalculatorMetadata(
        catalog_position=7,
        slug="paired-t-test",
        name="Paired T-Test",
        family=TestFamily.PARAMETRIC,
        description="Compare paired numeric measurements by testing whether their mean difference is zero.",
        check="Whether the average within-pair difference differs from zero.",
        statistic_formula="t = d̄ / (s_d / √n)",
        assumptions=(
            "The observations are paired by design.",
            "Pairs are independent of each other.",
            "The paired differences are approximately normal for small samples.",
        ),
        required_sample_data=(
            "One numeric sample for measurement A.",
            "One numeric sample for measurement B with the same length.",
        ),
        input_fields=(
            textarea_field("sample_a", "Measurement A values", "Enter comma-separated values for the first paired measurement.", placeholder="12, 14, 13, 15", rows=5),
            textarea_field("sample_b", "Measurement B values", "Enter comma-separated values for the second paired measurement.", placeholder="11, 13, 11, 14", rows=5),
            alternative_field(),
            alpha_field(),
        ),
    )

    def normalize(self, raw_data: Mapping[str, Any]) -> PairedTTestInput:
        sample_a, sample_b, issues = parse_paired_numeric_samples(
            raw_data.get("sample_a"),
            raw_data.get("sample_b"),
            field_a="sample_a",
            field_b="sample_b",
            label_a="Measurement A values",
            label_b="Measurement B values",
        )
        alternative, alternative_issues = parse_alternative(raw_data.get("alternative"))
        alpha, alpha_issues = parse_alpha(raw_data.get("alpha"))
        issues.extend(alternative_issues)
        issues.extend(alpha_issues)
        raise_if_issues(issues)
        return PairedTTestInput(sample_a=sample_a, sample_b=sample_b, alternative=alternative, alpha=alpha)

    def calculate_result(self, normalized_input: PairedTTestInput) -> CalculationResult:
        sample_a = np.asarray(normalized_input.sample_a, dtype=float)
        sample_b = np.asarray(normalized_input.sample_b, dtype=float)
        differences = sample_a - sample_b
        n = int(differences.size)
        mean_difference = float(differences.mean())
        std_difference = float(differences.std(ddof=1))
        standard_error = std_difference / sqrt(n)
        df = n - 1
        test_result = stats.ttest_rel(sample_a, sample_b, alternative=normalized_input.alternative.value)
        ci_low, ci_high = stats.t.interval(1 - normalized_input.alpha, df=df, loc=mean_difference, scale=standard_error)
        statistic = float(test_result.statistic)
        p_value = float(test_result.pvalue)
        reject_null = p_value < normalized_input.alpha
        metrics = (
            ResultMetric("Paired observations", str(n)),
            ResultMetric("Mean difference", format_number(mean_difference), emphasis=True),
            ResultMetric("Standard deviation of differences", format_number(std_difference)),
            ResultMetric("Standard error", format_number(standard_error)),
            ResultMetric("Degrees of freedom", str(df)),
            ResultMetric(
                f"{int((1 - normalized_input.alpha) * 100)}% CI for the mean difference",
                f"[{format_number(float(ci_low))}, {format_number(float(ci_high))}]",
            ),
        )
        return CalculationResult(
            slug=self.metadata.slug,
            test_name=self.metadata.name,
            statistic_name="t",
            statistic=display_number(statistic),
            p_value=display_p_value(p_value),
            metrics=metrics,
            decision=DecisionSummary(
                alpha=normalized_input.alpha,
                reject_null=reject_null,
                conclusion=_directional_decision(self.metadata.name, reject_null, normalized_input.alternative),
            ),
            interpretation=f"The average within-pair difference is {format_number(mean_difference)}.",
        )
