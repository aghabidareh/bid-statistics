from django.test import SimpleTestCase

from services.calculators.registry import calculate_test_statistic


class NonparametricCalculatorTests(SimpleTestCase):
    def test_mann_whitney_returns_expected_values(self):
        result = calculate_test_statistic(
            "mann-whitney-u-test",
            {
                "sample_a": "3, 4, 6, 7, 9",
                "sample_b": "1, 2, 5, 5, 8",
                "alternative": "two-sided",
                "alpha": "0.05",
            },
        )

        self.assertAlmostEqual(result.statistic.raw, 17.0)
        self.assertAlmostEqual(result.p_value.raw, 0.4019653583567354)

    def test_paired_wilcoxon_returns_expected_values(self):
        result = calculate_test_statistic(
            "paired-wilcoxon-signed-rank-test",
            {
                "sample_a": "12, 14, 13, 15",
                "sample_b": "11, 13, 11, 14",
                "alternative": "two-sided",
                "alpha": "0.05",
            },
        )

        self.assertAlmostEqual(result.statistic.raw, 0.0)
        self.assertAlmostEqual(result.p_value.raw, 0.125)

    def test_kruskal_wallis_returns_expected_values(self):
        result = calculate_test_statistic(
            "kruskal-wallis-test",
            {
                "groups": "Control: 4, 5, 6\nTreatment A: 6, 7, 8\nTreatment B: 8, 9, 10",
                "alpha": "0.05",
            },
        )

        self.assertAlmostEqual(result.statistic.raw, 6.531073446327686)
        self.assertAlmostEqual(result.p_value.raw, 0.038176439404577586)

    def test_friedman_returns_expected_values(self):
        result = calculate_test_statistic(
            "friedman-test",
            {
                "rows": "S1, Baseline, 4\nS1, Mid, 6\nS1, Final, 7\nS2, Baseline, 5\nS2, Mid, 6\nS2, Final, 8\nS3, Baseline, 6\nS3, Mid, 7\nS3, Final, 9\nS4, Baseline, 5\nS4, Mid, 7\nS4, Final, 8",
                "alpha": "0.05",
            },
        )

        self.assertAlmostEqual(result.statistic.raw, 8.0)
        self.assertAlmostEqual(result.p_value.raw, 0.018315638888734182)

    def test_paired_wilcoxon_adds_warning_when_zero_differences_exist(self):
        result = calculate_test_statistic(
            "paired-wilcoxon-signed-rank-test",
            {
                "sample_a": "10, 10, 12, 14",
                "sample_b": "10, 9, 12, 10",
                "alternative": "two-sided",
                "alpha": "0.05",
            },
        )

        self.assertEqual(len(result.warnings), 1)
        self.assertIn("Zero paired differences", result.warnings[0])
