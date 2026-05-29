from django.test import SimpleTestCase

from services.calculators.registry import calculate_test_statistic
from services.validators import ValidationIssues, errors_by_field


class ValidationParserTests(SimpleTestCase):
    def test_paired_samples_require_equal_length(self):
        with self.assertRaises(ValidationIssues) as context:
            calculate_test_statistic(
                "paired-t-test",
                {
                    "sample_a": "1, 2, 3",
                    "sample_b": "1, 2",
                    "alternative": "two-sided",
                    "alpha": "0.05",
                },
            )

        self.assertEqual(
            errors_by_field(context.exception.issues),
            {"sample_b": ["Measurement A values and Measurement B values must have the same length."]},
        )

    def test_goodness_of_fit_requires_matching_lengths(self):
        with self.assertRaises(ValidationIssues) as context:
            calculate_test_statistic(
                "chi-squared-goodness-of-fit-test",
                {
                    "observed": "10, 12, 14",
                    "expected": "0.5, 0.5",
                    "alpha": "0.05",
                },
            )

        self.assertEqual(
            errors_by_field(context.exception.issues),
            {"expected": ["Observed counts and expected values must have the same length."]},
        )

    def test_repeated_measures_rows_require_complete_subject_condition_grid(self):
        with self.assertRaises(ValidationIssues) as context:
            calculate_test_statistic(
                "repeated-measures-anova",
                {
                    "rows": "S1, Baseline, 4\nS1, Final, 6\nS2, Baseline, 5",
                    "alpha": "0.05",
                },
            )

        self.assertEqual(
            errors_by_field(context.exception.issues),
            {"rows": ["Each subject must have exactly one observation for every condition."]},
        )

    def test_paired_roc_rows_require_both_binary_labels(self):
        with self.assertRaises(ValidationIssues) as context:
            calculate_test_statistic(
                "delong-test-paired-curves",
                {
                    "rows": "1, 0.9, 0.8\n1, 0.7, 0.6\n1, 0.5, 0.4",
                    "alpha": "0.05",
                },
            )

        self.assertEqual(
            errors_by_field(context.exception.issues),
            {"rows": ["Paired ROC rows must include both positive and negative labels."]},
        )

    def test_one_sample_ks_distribution_parameters_require_positive_scale(self):
        with self.assertRaises(ValidationIssues) as context:
            calculate_test_statistic(
                "one-sample-kolmogorov-smirnov-test",
                {
                    "sample": "1, 2, 3",
                    "distribution": "norm",
                    "distribution_parameters": "0, 0",
                    "alternative": "two-sided",
                    "alpha": "0.05",
                },
            )

        self.assertEqual(
            errors_by_field(context.exception.issues),
            {"distribution_parameters": ["The scale parameter must be greater than zero."]},
        )
