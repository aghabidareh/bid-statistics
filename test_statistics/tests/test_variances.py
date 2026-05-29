from django.test import SimpleTestCase

from services.calculators.registry import calculate_test_statistic


class VarianceCalculatorTests(SimpleTestCase):
    def test_chi_squared_variance_returns_expected_values(self):
        result = calculate_test_statistic(
            "chi-squared-variance-test",
            {
                "sample": "12, 15, 14, 13, 16",
                "null_variance": "4",
                "alternative": "two-sided",
                "alpha": "0.05",
            },
        )

        self.assertAlmostEqual(result.statistic.raw, 2.5)
        self.assertAlmostEqual(result.p_value.raw, 0.7107284141291443)

    def test_f_test_for_variances_returns_expected_values(self):
        result = calculate_test_statistic(
            "f-test-for-variances",
            {
                "sample_a": "10, 12, 13, 11, 9",
                "sample_b": "16, 18, 15, 17, 23",
                "alternative": "two-sided",
                "alpha": "0.05",
            },
        )

        self.assertAlmostEqual(result.statistic.raw, 0.2577319587628866)
        self.assertAlmostEqual(result.p_value.raw, 0.21752922050744342)

    def test_levene_returns_expected_values(self):
        result = calculate_test_statistic(
            "levene-test-for-variances",
            {
                "groups": "Control: 4, 5, 6\nTreatment A: 6, 7, 12\nTreatment B: 8, 9, 10",
                "alpha": "0.05",
            },
        )

        self.assertAlmostEqual(result.statistic.raw, 0.6956521739130435)
        self.assertAlmostEqual(result.p_value.raw, 0.5349220435579076)
