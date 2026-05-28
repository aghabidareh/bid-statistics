from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy import stats

from test_statistics.domain.enums import TestFamily
from test_statistics.domain.inputs import MannWhitneyInput, NamedGroupsInput, PairedWilcoxonInput, RepeatedMeasuresInput
from test_statistics.domain.metadata import CalculatorMetadata
from test_statistics.domain.results import CalculationResult, DecisionSummary, ResultMetric, display_number, display_p_value, format_number
from test_statistics.services.calculators.base import NamedGroupCalculator, PairedSampleCalculator, TwoIndependentSampleCalculator, alpha_field, alternative_field, textarea_field
from test_statistics.services.validators import parse_alpha, parse_alternative, parse_named_groups, parse_paired_numeric_samples, parse_repeated_measures_rows, parse_numeric_series, raise_if_issues



def _decision(test_name: str, reject_null: bool) -> str:
    if reject_null:
        return f"Reject the null hypothesis for the {test_name}."
    return f"Fail to reject the null hypothesis for the {test_name}."


class MannWhitneyCalculator(TwoIndependentSampleCalculator):
    metadata = CalculatorMetadata(
        catalog_position=6,
        slug="mann-whitney-u-test",
        name="Two Sample Mann-Whitney U-Test",
        family=TestFamily.NONPARAMETRIC,
        description="Compare two independent samples using a rank-based alternative to the two-sample t-test.",
        check="Whether two independent samples differ in location without assuming normality.",
        statistic_formula="U = min(U₁, U₂)",
        assumptions=(
            "The two samples are independent.",
            "The outcome is ordinal or continuous.",
            "The test compares stochastic ordering or location depending on assumptions.",
        ),
        required_sample_data=(
            "One numeric sample for group A.",
            "One numeric sample for group B.",
        ),
        input_fields=(
            textarea_field("sample_a", "Group A values", "Enter comma-separated values for the first sample.", placeholder="3, 4, 6, 7, 9", rows=5),
            textarea_field("sample_b", "Group B values", "Enter comma-separated values for the second sample.", placeholder="1, 2, 5, 5, 8", rows=5),
            alternative_field(),
            alpha_field(),
        ),
    )

    def normalize(self, raw_data: Mapping[str, Any]) -> MannWhitneyInput:
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
        return MannWhitneyInput(sample_a=sample_a, sample_b=sample_b, alternative=alternative, alpha=alpha)

    def calculate_result(self, normalized_input: MannWhitneyInput) -> CalculationResult:
        sample_a = np.asarray(normalized_input.sample_a, dtype=float)
        sample_b = np.asarray(normalized_input.sample_b, dtype=float)
        result = stats.mannwhitneyu(sample_a, sample_b, alternative=normalized_input.alternative.value, method="auto")
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
        reject_null = p_value < normalized_input.alpha
        metrics = (
            ResultMetric("Group A size", str(sample_a.size)),
            ResultMetric("Group B size", str(sample_b.size)),
            ResultMetric("Group A median", format_number(float(np.median(sample_a))), emphasis=True),
            ResultMetric("Group B median", format_number(float(np.median(sample_b))), emphasis=True),
            ResultMetric("Rank-sum comparison", format_number(float(sample_a.size * sample_b.size - statistic))),
        )
        return CalculationResult(
            slug=self.metadata.slug,
            test_name=self.metadata.name,
            statistic_name="U",
            statistic=display_number(statistic),
            p_value=display_p_value(p_value),
            metrics=metrics,
            decision=DecisionSummary(alpha=normalized_input.alpha, reject_null=reject_null, conclusion=_decision(self.metadata.name, reject_null)),
            interpretation="The result compares the rank distributions of the two independent samples.",
        )


class PairedWilcoxonCalculator(PairedSampleCalculator):
    metadata = CalculatorMetadata(
        catalog_position=8,
        slug="paired-wilcoxon-signed-rank-test",
        name="Paired Wilcoxon Sign Rank Test",
        family=TestFamily.NONPARAMETRIC,
        description="Compare paired numeric measurements using a signed-rank test when the paired t-test assumptions are not appropriate.",
        check="Whether the median paired difference differs from zero.",
        statistic_formula="W = sum of signed ranks",
        assumptions=(
            "The observations are paired by design.",
            "Pairs are independent of each other.",
            "The paired differences are symmetrically distributed around the median.",
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

    def normalize(self, raw_data: Mapping[str, Any]) -> PairedWilcoxonInput:
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
        return PairedWilcoxonInput(sample_a=sample_a, sample_b=sample_b, alternative=alternative, alpha=alpha)

    def calculate_result(self, normalized_input: PairedWilcoxonInput) -> CalculationResult:
        sample_a = np.asarray(normalized_input.sample_a, dtype=float)
        sample_b = np.asarray(normalized_input.sample_b, dtype=float)
        differences = sample_a - sample_b
        result = stats.wilcoxon(sample_a, sample_b, alternative=normalized_input.alternative.value, zero_method="wilcox")
        p_value = float(result.pvalue)
        reject_null = p_value < normalized_input.alpha
        warnings = ()
        if np.count_nonzero(differences) < differences.size:
            warnings = ("Zero paired differences were excluded from the signed-rank calculation.",)
        metrics = (
            ResultMetric("Paired observations", str(differences.size)),
            ResultMetric("Median difference", format_number(float(np.median(differences))), emphasis=True),
            ResultMetric("Non-zero differences", str(int(np.count_nonzero(differences)))),
        )
        return CalculationResult(
            slug=self.metadata.slug,
            test_name=self.metadata.name,
            statistic_name="W",
            statistic=display_number(float(result.statistic)),
            p_value=display_p_value(p_value),
            metrics=metrics,
            decision=DecisionSummary(alpha=normalized_input.alpha, reject_null=reject_null, conclusion=_decision(self.metadata.name, reject_null)),
            interpretation="The signed ranks summarize the direction and magnitude of the within-pair differences.",
            warnings=warnings,
        )


class KruskalWallisCalculator(NamedGroupCalculator):
    metadata = CalculatorMetadata(
        catalog_position=11,
        slug="kruskal-wallis-test",
        name="Kruskal-Wallis Test",
        family=TestFamily.NONPARAMETRIC,
        description="Compare three or more independent groups using a rank-based alternative to one-way ANOVA.",
        check="Whether at least one group's distribution differs from the others.",
        statistic_formula="H = rank-based omnibus statistic",
        assumptions=(
            "Observations are independent within and across groups.",
            "The outcome is ordinal or continuous.",
            "The test compares distributions across groups.",
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
        result = stats.kruskal(*arrays)
        group_summaries = "; ".join(
            f"{group_name}: median {format_number(float(np.median(np.asarray(group_values, dtype=float))))}"
            for group_name, group_values in normalized_input.groups
        )
        p_value = float(result.pvalue)
        reject_null = p_value < normalized_input.alpha
        metrics = (
            ResultMetric("Groups", str(len(arrays))),
            ResultMetric("Total observations", str(sum(array.size for array in arrays))),
            ResultMetric("Group medians", group_summaries, emphasis=True),
            ResultMetric("Degrees of freedom", str(len(arrays) - 1)),
        )
        return CalculationResult(
            slug=self.metadata.slug,
            test_name=self.metadata.name,
            statistic_name="H",
            statistic=display_number(float(result.statistic)),
            p_value=display_p_value(p_value),
            metrics=metrics,
            decision=DecisionSummary(alpha=normalized_input.alpha, reject_null=reject_null, conclusion=_decision(self.metadata.name, reject_null)),
            interpretation="The omnibus rank test assesses whether at least one group tends to have higher or lower values.",
        )


class FriedmanCalculator(NamedGroupCalculator):
    metadata = CalculatorMetadata(
        catalog_position=12,
        slug="friedman-test",
        name="Friedman Test",
        family=TestFamily.NONPARAMETRIC,
        description="Compare repeated measurements across conditions using a rank-based alternative to repeated-measures ANOVA.",
        check="Whether at least one repeated-measures condition differs from the others.",
        statistic_formula="Q = rank-based repeated-measures statistic",
        assumptions=(
            "The same subjects are measured under each condition.",
            "Subjects are independent of one another.",
            "The outcome is ordinal or continuous.",
        ),
        required_sample_data=(
            "Repeated-measures rows in 'subject, condition, value' format.",
            "Each subject must appear once for every condition.",
        ),
        input_fields=(
            textarea_field("rows", "Repeated-measures rows", "Enter one observation per line as 'subject, condition, value'.", placeholder="S1, Baseline, 4\nS1, Treatment, 6\nS2, Baseline, 5\nS2, Treatment, 7", rows=8),
            alpha_field(),
        ),
    )

    def normalize(self, raw_data: Mapping[str, Any]) -> RepeatedMeasuresInput:
        issues = []
        rows, row_issues = parse_repeated_measures_rows(raw_data.get("rows"), "rows")
        alpha, alpha_issues = parse_alpha(raw_data.get("alpha"))
        issues.extend(row_issues)
        issues.extend(alpha_issues)
        raise_if_issues(issues)
        return RepeatedMeasuresInput(rows=rows, alpha=alpha)

    def calculate_result(self, normalized_input: RepeatedMeasuresInput) -> CalculationResult:
        by_condition: defaultdict[str, list[tuple[str, float]]] = defaultdict(list)
        for row in normalized_input.rows:
            by_condition[row.condition].append((row.subject, row.value))
        ordered_conditions = tuple(sorted(by_condition))
        ordered_subjects = tuple(sorted({row.subject for row in normalized_input.rows}))
        arrays = []
        for condition in ordered_conditions:
            values_by_subject = {subject: value for subject, value in by_condition[condition]}
            arrays.append(np.asarray([values_by_subject[subject] for subject in ordered_subjects], dtype=float))
        result = stats.friedmanchisquare(*arrays)
        condition_medians = "; ".join(
            f"{condition}: median {format_number(float(np.median(array)))}"
            for condition, array in zip(ordered_conditions, arrays, strict=True)
        )
        p_value = float(result.pvalue)
        reject_null = p_value < normalized_input.alpha
        metrics = (
            ResultMetric("Subjects", str(len(ordered_subjects))),
            ResultMetric("Conditions", str(len(ordered_conditions))),
            ResultMetric("Condition medians", condition_medians, emphasis=True),
            ResultMetric("Degrees of freedom", str(len(ordered_conditions) - 1)),
        )
        return CalculationResult(
            slug=self.metadata.slug,
            test_name=self.metadata.name,
            statistic_name="Q",
            statistic=display_number(float(result.statistic)),
            p_value=display_p_value(p_value),
            metrics=metrics,
            decision=DecisionSummary(alpha=normalized_input.alpha, reject_null=reject_null, conclusion=_decision(self.metadata.name, reject_null)),
            interpretation="The repeated-measures rank test compares condition tendencies within the same subjects.",
        )
