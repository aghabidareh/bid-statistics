from django.test import SimpleTestCase

from domain.enums import TestFamily
from domain.metadata import CalculatorMetadata
from domain.results import CalculationResult
from services.calculators.base import BaseCalculator, DuplicateCalculatorSlugError, UnknownCalculatorError
from services.calculators.registry import get_calculator, list_calculators


class CalculatorRegistryTests(SimpleTestCase):
    def test_catalog_contains_exact_strict_26_calculators_in_order(self):
        calculators = list_calculators()

        self.assertEqual(len(calculators), 26)
        self.assertEqual(
            [calculator.slug for calculator in calculators],
            [
                "one-sample-z-test",
                "one-sample-t-test",
                "two-sample-z-test",
                "two-sample-t-test-pooled",
                "two-sample-t-test-welch",
                "mann-whitney-u-test",
                "paired-t-test",
                "paired-wilcoxon-signed-rank-test",
                "one-way-anova",
                "repeated-measures-anova",
                "kruskal-wallis-test",
                "friedman-test",
                "two-way-anova",
                "one-way-manova",
                "one-sample-proportion-test",
                "two-sample-proportion-test",
                "chi-squared-variance-test",
                "f-test-for-variances",
                "levene-test-for-variances",
                "chi-squared-goodness-of-fit-test",
                "shapiro-wilk-test",
                "one-sample-kolmogorov-smirnov-test",
                "two-sample-kolmogorov-smirnov-test",
                "kaplan-meier-survival-analysis",
                "delong-test-independent-curves",
                "delong-test-paired-curves",
            ],
        )

    def test_registry_dispatches_to_expected_subclass(self):
        calculator = get_calculator("two-sample-t-test-welch")

        self.assertEqual(calculator.metadata.name, "Two Sample T-Test (Welch's)")

    def test_unknown_slug_raises_lookup_error(self):
        with self.assertRaises(UnknownCalculatorError):
            get_calculator("missing-calculator")

    def test_duplicate_slug_registration_is_rejected(self):
        with self.assertRaises(DuplicateCalculatorSlugError):

            class DuplicateSlugCalculator(BaseCalculator):
                metadata = CalculatorMetadata(
                    catalog_position=999,
                    slug="one-sample-t-test",
                    name="Duplicate",
                    family=TestFamily.PARAMETRIC,
                    description="Duplicate",
                    check="Duplicate",
                    statistic_formula="Duplicate",
                    assumptions=("Duplicate",),
                    required_sample_data=("Duplicate",),
                )

                def normalize(self, raw_data):
                    return raw_data

                def calculate_result(self, normalized_input):
                    return CalculationResult(slug="duplicate", test_name="Duplicate")
