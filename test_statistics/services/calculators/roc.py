from __future__ import annotations

from collections.abc import Mapping
from math import sqrt
from typing import Any

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

from test_statistics.domain.enums import TestFamily
from test_statistics.domain.inputs import IndependentDelongInput, PairedDelongInput
from test_statistics.domain.metadata import CalculatorMetadata
from test_statistics.domain.results import CalculationResult, DecisionSummary, ResultMetric, display_number, display_p_value, format_number
from test_statistics.services.calculators.base import RocComparisonCalculator, alpha_field, textarea_field
from test_statistics.services.validators import parse_alpha, parse_paired_roc_rows, parse_roc_rows, raise_if_issues



def _compute_midrank(values: np.ndarray) -> np.ndarray:
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    midranks = np.zeros(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        midranks[start:end] = 0.5 * (start + end - 1) + 1
        start = end
    result = np.empty(len(values), dtype=float)
    result[sorted_indices] = midranks
    return result



def _fast_delong(predictions_sorted_transposed: np.ndarray, positive_count: int) -> tuple[np.ndarray, np.ndarray]:
    negative_count = predictions_sorted_transposed.shape[1] - positive_count
    positive_examples = predictions_sorted_transposed[:, :positive_count]
    negative_examples = predictions_sorted_transposed[:, positive_count:]
    classifier_count = predictions_sorted_transposed.shape[0]

    tx = np.empty((classifier_count, positive_count), dtype=float)
    ty = np.empty((classifier_count, negative_count), dtype=float)
    tz = np.empty((classifier_count, positive_count + negative_count), dtype=float)
    for index in range(classifier_count):
        tx[index, :] = _compute_midrank(positive_examples[index, :])
        ty[index, :] = _compute_midrank(negative_examples[index, :])
        tz[index, :] = _compute_midrank(predictions_sorted_transposed[index, :])

    aucs = tz[:, :positive_count].sum(axis=1) / positive_count / negative_count - (positive_count + 1.0) / (2.0 * negative_count)
    v01 = (tz[:, :positive_count] - tx) / negative_count
    v10 = 1.0 - (tz[:, positive_count:] - ty) / positive_count
    sx = np.cov(v01)
    sy = np.cov(v10)
    covariance = sx / positive_count + sy / negative_count
    covariance = np.atleast_2d(covariance)
    return aucs, covariance



def _prepare_sorted_predictions(labels: np.ndarray, *score_arrays: np.ndarray) -> tuple[np.ndarray, int]:
    order = np.argsort(-labels)
    sorted_labels = labels[order]
    positive_count = int(sorted_labels.sum())
    stacked_scores = np.vstack([scores[order] for scores in score_arrays])
    return stacked_scores, positive_count



def _independent_auc_and_variance(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    predictions, positive_count = _prepare_sorted_predictions(labels, scores)
    aucs, covariance = _fast_delong(predictions, positive_count)
    return float(aucs[0]), float(covariance[0, 0])



def _paired_auc_and_covariance(labels: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray) -> tuple[float, float, np.ndarray]:
    predictions, positive_count = _prepare_sorted_predictions(labels, scores_a, scores_b)
    aucs, covariance = _fast_delong(predictions, positive_count)
    return float(aucs[0]), float(aucs[1]), covariance



def _z_test_from_difference(difference: float, variance: float) -> tuple[float, float]:
    standard_error = sqrt(max(variance, 0.0))
    if standard_error == 0:
        return 0.0, 1.0
    statistic = difference / standard_error
    p_value = 2 * stats.norm.sf(abs(statistic))
    return float(statistic), float(p_value)



def _decision(test_name: str, reject_null: bool) -> str:
    if reject_null:
        return f"Reject the null hypothesis for the {test_name}."
    return f"Fail to reject the null hypothesis for the {test_name}."


class IndependentDelongCalculator(RocComparisonCalculator):
    metadata = CalculatorMetadata(
        catalog_position=25,
        slug="delong-test-independent-curves",
        name="DeLong Test - Independent Curves",
        family=TestFamily.ROC,
        description="Compare the areas under two independent ROC curves.",
        check="Whether two independent AUC estimates differ.",
        statistic_formula="z = (AUC₁ - AUC₂) / √(Var(AUC₁) + Var(AUC₂))",
        assumptions=(
            "The two ROC datasets are independent.",
            "Each dataset includes both positive and negative labels.",
            "Scores are higher for stronger positive evidence.",
        ),
        required_sample_data=(
            "ROC rows for curve A in 'label, score' format.",
            "ROC rows for curve B in 'label, score' format.",
        ),
        input_fields=(
            textarea_field("curve_a", "Curve A rows", "Enter one observation per line as 'label, score'.", placeholder="1, 0.91\n1, 0.82\n0, 0.40\n0, 0.31", rows=8),
            textarea_field("curve_b", "Curve B rows", "Enter one observation per line as 'label, score'.", placeholder="1, 0.88\n1, 0.74\n0, 0.53\n0, 0.28", rows=8),
            alpha_field(),
        ),
    )

    def normalize(self, raw_data: Mapping[str, Any]) -> IndependentDelongInput:
        curve_a, curve_a_issues = parse_roc_rows(raw_data.get("curve_a"), "curve_a", "Curve A")
        curve_b, curve_b_issues = parse_roc_rows(raw_data.get("curve_b"), "curve_b", "Curve B")
        alpha, alpha_issues = parse_alpha(raw_data.get("alpha"))
        issues = [*curve_a_issues, *curve_b_issues, *alpha_issues]
        raise_if_issues(issues)
        return IndependentDelongInput(curve_a=curve_a, curve_b=curve_b, alpha=alpha)

    def calculate_result(self, normalized_input: IndependentDelongInput) -> CalculationResult:
        labels_a = np.asarray([row.label for row in normalized_input.curve_a], dtype=int)
        scores_a = np.asarray([row.score for row in normalized_input.curve_a], dtype=float)
        labels_b = np.asarray([row.label for row in normalized_input.curve_b], dtype=int)
        scores_b = np.asarray([row.score for row in normalized_input.curve_b], dtype=float)
        auc_a, variance_a = _independent_auc_and_variance(labels_a, scores_a)
        auc_b, variance_b = _independent_auc_and_variance(labels_b, scores_b)
        difference = auc_a - auc_b
        statistic, p_value = _z_test_from_difference(difference, variance_a + variance_b)
        reject_null = p_value < normalized_input.alpha
        metrics = (
            ResultMetric("Curve A AUC", format_number(float(roc_auc_score(labels_a, scores_a))), emphasis=True),
            ResultMetric("Curve B AUC", format_number(float(roc_auc_score(labels_b, scores_b))), emphasis=True),
            ResultMetric("AUC difference", format_number(difference)),
            ResultMetric("Curve A variance", format_number(variance_a)),
            ResultMetric("Curve B variance", format_number(variance_b)),
            ResultMetric("Standard error", format_number(sqrt(max(variance_a + variance_b, 0.0)))),
        )
        return CalculationResult(
            slug=self.metadata.slug,
            test_name=self.metadata.name,
            statistic_name="z",
            statistic=display_number(statistic),
            p_value=display_p_value(p_value),
            metrics=metrics,
            decision=DecisionSummary(alpha=normalized_input.alpha, reject_null=reject_null, conclusion=_decision(self.metadata.name, reject_null)),
            interpretation="The independent DeLong comparison treats the two AUC estimates as coming from separate ROC datasets.",
        )


class PairedDelongCalculator(RocComparisonCalculator):
    metadata = CalculatorMetadata(
        catalog_position=26,
        slug="delong-test-paired-curves",
        name="DeLong Test - Paired Curves",
        family=TestFamily.ROC,
        description="Compare the areas under two correlated ROC curves evaluated on the same cases.",
        check="Whether two paired AUC estimates differ.",
        statistic_formula="z = (AUC₁ - AUC₂) / √Var(AUC₁ - AUC₂)",
        assumptions=(
            "Both score columns are evaluated on the same observations.",
            "The labels are binary and include both positive and negative cases.",
            "Scores are higher for stronger positive evidence.",
        ),
        required_sample_data=(
            "Paired ROC rows in 'label, score_a, score_b' format.",
        ),
        input_fields=(
            textarea_field("rows", "Paired ROC rows", "Enter one observation per line as 'label, score_a, score_b'.", placeholder="1, 0.91, 0.88\n1, 0.82, 0.74\n0, 0.40, 0.53\n0, 0.31, 0.28", rows=8),
            alpha_field(),
        ),
    )

    def normalize(self, raw_data: Mapping[str, Any]) -> PairedDelongInput:
        observations, observation_issues = parse_paired_roc_rows(raw_data.get("rows"), "rows")
        alpha, alpha_issues = parse_alpha(raw_data.get("alpha"))
        issues = [*observation_issues, *alpha_issues]
        raise_if_issues(issues)
        return PairedDelongInput(observations=observations, alpha=alpha)

    def calculate_result(self, normalized_input: PairedDelongInput) -> CalculationResult:
        labels = np.asarray([row.label for row in normalized_input.observations], dtype=int)
        scores_a = np.asarray([row.score_a for row in normalized_input.observations], dtype=float)
        scores_b = np.asarray([row.score_b for row in normalized_input.observations], dtype=float)
        auc_a, auc_b, covariance = _paired_auc_and_covariance(labels, scores_a, scores_b)
        difference = auc_a - auc_b
        variance_difference = covariance[0, 0] + covariance[1, 1] - 2 * covariance[0, 1]
        statistic, p_value = _z_test_from_difference(difference, variance_difference)
        reject_null = p_value < normalized_input.alpha
        metrics = (
            ResultMetric("Curve A AUC", format_number(float(roc_auc_score(labels, scores_a))), emphasis=True),
            ResultMetric("Curve B AUC", format_number(float(roc_auc_score(labels, scores_b))), emphasis=True),
            ResultMetric("AUC difference", format_number(difference)),
            ResultMetric("Curve A variance", format_number(float(covariance[0, 0]))),
            ResultMetric("Curve B variance", format_number(float(covariance[1, 1]))),
            ResultMetric("Covariance", format_number(float(covariance[0, 1]))),
            ResultMetric("Standard error", format_number(sqrt(max(variance_difference, 0.0)))),
        )
        return CalculationResult(
            slug=self.metadata.slug,
            test_name=self.metadata.name,
            statistic_name="z",
            statistic=display_number(statistic),
            p_value=display_p_value(p_value),
            metrics=metrics,
            decision=DecisionSummary(alpha=normalized_input.alpha, reject_null=reject_null, conclusion=_decision(self.metadata.name, reject_null)),
            interpretation="The paired DeLong comparison uses the covariance between AUC estimates because both ROC curves are evaluated on the same cases.",
        )
