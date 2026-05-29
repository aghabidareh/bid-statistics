from django.test import SimpleTestCase

from services.calculators.registry import calculate_test_statistic


class DistributionCalculatorTests(SimpleTestCase):
    def test_goodness_of_fit_returns_expected_values(self):
        result = calculate_test_statistic(
            "chi-squared-goodness-of-fit-test",
            {
                "observed": "18, 22, 20",
                "expected": "20, 20, 20",
                "alpha": "0.05",
            },
        )

        self.assertAlmostEqual(result.statistic.raw, 0.4)
        self.assertAlmostEqual(result.p_value.raw, 0.8187307530779818)
        self.assertEqual(len(result.tables[0].rows), 3)

    def test_shapiro_wilk_returns_expected_values(self):
        result = calculate_test_statistic(
            "shapiro-wilk-test",
            {
                "sample": "12, 15, 14, 13, 16",
                "alpha": "0.05",
            },
        )

        self.assertAlmostEqual(result.statistic.raw, 0.986762155211559)
        self.assertAlmostEqual(result.p_value.raw, 0.9671739349728582)

    def test_one_sample_ks_returns_expected_values(self):
        result = calculate_test_statistic(
            "one-sample-kolmogorov-smirnov-test",
            {
                "sample": "-0.5, 0.1, 0.2, 0.8, 1.1",
                "distribution": "norm",
                "distribution_parameters": "0, 1",
                "alternative": "two-sided",
                "alpha": "0.05",
            },
        )

        self.assertAlmostEqual(result.statistic.raw, 0.339827837277029)
        self.assertAlmostEqual(result.p_value.raw, 0.5079160738313431)

    def test_two_sample_ks_returns_expected_values(self):
        result = calculate_test_statistic(
            "two-sample-kolmogorov-smirnov-test",
            {
                "sample_a": "1, 2, 3, 4, 5",
                "sample_b": "2, 3, 4, 5, 6",
                "alternative": "two-sided",
                "alpha": "0.05",
            },
        )

        self.assertAlmostEqual(result.statistic.raw, 0.2)
        self.assertAlmostEqual(result.p_value.raw, 1.0)
