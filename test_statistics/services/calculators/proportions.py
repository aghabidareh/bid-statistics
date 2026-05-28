from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from statsmodels.stats.proportion import confint_proportions_2indep, proportion_confint, proportions_ztest

from test_statistics.domain.enums import TestFamily
from test_statistics.domain.inputs import OneSampleProportionInput, TwoSampleProportionInput
from test_statistics.domain.metadata import CalculatorMetadata
from test_statistics.domain.results import CalculationResult, DecisionSummary, ResultMetric, display_number, display_p_value, format_number
from test_statistics.services.calculators.base import BaseCalculator, alpha_field, alternative_field, numeric_field
from test_statistics.services.validators import parse_alpha, parse_alternative, parse_count_trial_inputs, parse_probability, raise_if_issues



def _decision(test_name: str, reject_null: bool) -> str:
    if reject_null:
        return f"Reject the null hypothesis for the {test_name}."
    return f"Fail to reject the null hypothesis for the {test_name}."


class OneSampleProportionCalculator(BaseCalculator):
    metadata = CalculatorMetadata(
        catalog_position=15,
        slug="one-sample-proportion-test",
        name="One Sample Proportion Test",
        family=TestFamily.PROPORTIONS,
        description="Compare an observed success proportion against a hypothesized population proportion.",
        check="Whether one sample proportion differs from a reference proportion.",
        statistic_formula="z = (p̂ - p₀) / √(p₀(1 - p₀) / n)",
        assumptions=(
            "Trials are independent.",
            "Each trial has a binary outcome.",
            "The sample size is large enough for the normal approximation.",
        ),
        required_sample_data=(
            "A success count.",
            "A trial count.",
            "A hypothesized population proportion.",
        ),
        input_fields=(
            numeric_field("successes", "Successes", "Enter the number of successes.", placeholder="42", min_value="0"),
            numeric_field("trials", "Trials", "Enter the total number of trials.", placeholder="60", min_value="1"),
            numeric_field("null_proportion", "Hypothesized proportion", "Enter the proportion to test against.", placeholder="0.5", default_value="0.5", min_value="0", max_value="1"),
            alternative_field(),
            alpha_field(),
        ),
    )

    def normalize(self, raw_data: Mapping[str, Any]) -> OneSampleProportionInput:
        successes, trials, issues = parse_count_trial_inputs(
            raw_data.get("successes"),
            raw_data.get("trials"),
            successes_field="successes",
            trials_field="trials",
            label_prefix="Sample",
        )
        null_proportion = parse_probability(raw_data.get("null_proportion"), "null_proportion", "Hypothesized proportion", issues)
        alternative, alternative_issues = parse_alternative(raw_data.get("alternative"))
        alpha, alpha_issues = parse_alpha(raw_data.get("alpha"))
        issues.extend(alternative_issues)
        issues.extend(alpha_issues)
        raise_if_issues(issues)
        return OneSampleProportionInput(successes=successes, trials=trials, null_proportion=null_proportion, alternative=alternative, alpha=alpha)

    def calculate_result(self, normalized_input: OneSampleProportionInput) -> CalculationResult:
        statistic, p_value = proportions_ztest(
            count=normalized_input.successes,
            nobs=normalized_input.trials,
            value=normalized_input.null_proportion,
            alternative=normalized_input.alternative.value,
        )
        observed_proportion = normalized_input.successes / normalized_input.trials
        ci_low, ci_high = proportion_confint(
            normalized_input.successes,
            normalized_input.trials,
            alpha=normalized_input.alpha,
            method="normal",
        )
        reject_null = float(p_value) < normalized_input.alpha
        metrics = (
            ResultMetric("Observed proportion", format_number(observed_proportion), emphasis=True),
            ResultMetric("Hypothesized proportion", format_number(normalized_input.null_proportion)),
            ResultMetric("Successes", str(normalized_input.successes)),
            ResultMetric("Trials", str(normalized_input.trials)),
            ResultMetric(
                f"{int((1 - normalized_input.alpha) * 100)}% CI for the observed proportion",
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
            decision=DecisionSummary(alpha=normalized_input.alpha, reject_null=reject_null, conclusion=_decision(self.metadata.name, reject_null)),
            interpretation="The z-test compares the observed proportion against the reference proportion under a normal approximation.",
        )


class TwoSampleProportionCalculator(BaseCalculator):
    metadata = CalculatorMetadata(
        catalog_position=16,
        slug="two-sample-proportion-test",
        name="Two Sample Proportion Test",
        family=TestFamily.PROPORTIONS,
        description="Compare two independent success proportions.",
        check="Whether two groups have different success probabilities.",
        statistic_formula="z = (p̂₁ - p̂₂) / SE(p̂₁ - p̂₂)",
        assumptions=(
            "The two groups are independent.",
            "Each trial has a binary outcome.",
            "Both groups are large enough for the normal approximation.",
        ),
        required_sample_data=(
            "Successes and trials for group A.",
            "Successes and trials for group B.",
        ),
        input_fields=(
            numeric_field("successes_a", "Group A successes", "Enter the number of successes in group A.", placeholder="42", min_value="0"),
            numeric_field("trials_a", "Group A trials", "Enter the number of trials in group A.", placeholder="60", min_value="1"),
            numeric_field("successes_b", "Group B successes", "Enter the number of successes in group B.", placeholder="30", min_value="0"),
            numeric_field("trials_b", "Group B trials", "Enter the number of trials in group B.", placeholder="55", min_value="1"),
            alternative_field(),
            alpha_field(),
        ),
    )

    def normalize(self, raw_data: Mapping[str, Any]) -> TwoSampleProportionInput:
        successes_a, trials_a, issues = parse_count_trial_inputs(
            raw_data.get("successes_a"),
            raw_data.get("trials_a"),
            successes_field="successes_a",
            trials_field="trials_a",
            label_prefix="Group A",
        )
        successes_b, trials_b, group_b_issues = parse_count_trial_inputs(
            raw_data.get("successes_b"),
            raw_data.get("trials_b"),
            successes_field="successes_b",
            trials_field="trials_b",
            label_prefix="Group B",
        )
        issues.extend(group_b_issues)
        alternative, alternative_issues = parse_alternative(raw_data.get("alternative"))
        alpha, alpha_issues = parse_alpha(raw_data.get("alpha"))
        issues.extend(alternative_issues)
        issues.extend(alpha_issues)
        raise_if_issues(issues)
        return TwoSampleProportionInput(
            successes_a=successes_a,
            trials_a=trials_a,
            successes_b=successes_b,
            trials_b=trials_b,
            alternative=alternative,
            alpha=alpha,
        )

    def calculate_result(self, normalized_input: TwoSampleProportionInput) -> CalculationResult:
        statistic, p_value = proportions_ztest(
            count=[normalized_input.successes_a, normalized_input.successes_b],
            nobs=[normalized_input.trials_a, normalized_input.trials_b],
            alternative=normalized_input.alternative.value,
        )
        proportion_a = normalized_input.successes_a / normalized_input.trials_a
        proportion_b = normalized_input.successes_b / normalized_input.trials_b
        difference = proportion_a - proportion_b
        ci_low, ci_high = confint_proportions_2indep(
            normalized_input.successes_a,
            normalized_input.trials_a,
            normalized_input.successes_b,
            normalized_input.trials_b,
            alpha=normalized_input.alpha,
            method="wald",
        )
        reject_null = float(p_value) < normalized_input.alpha
        metrics = (
            ResultMetric("Group A proportion", format_number(proportion_a), emphasis=True),
            ResultMetric("Group B proportion", format_number(proportion_b), emphasis=True),
            ResultMetric("Difference", format_number(difference)),
            ResultMetric("Group A successes / trials", f"{normalized_input.successes_a} / {normalized_input.trials_a}"),
            ResultMetric("Group B successes / trials", f"{normalized_input.successes_b} / {normalized_input.trials_b}"),
            ResultMetric(
                f"{int((1 - normalized_input.alpha) * 100)}% CI for the difference",
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
            decision=DecisionSummary(alpha=normalized_input.alpha, reject_null=reject_null, conclusion=_decision(self.metadata.name, reject_null)),
            interpretation="The z-test compares the two observed proportions under an independent-samples normal approximation.",
        )
