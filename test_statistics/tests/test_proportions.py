from django.test import SimpleTestCase

from services.calculators.registry import calculate_test_statistic


class ProportionCalculatorTests(SimpleTestCase):
    def test_one_sample_proportion_returns_expected_values(self):
        result = calculate_test_statistic(
            "one-sample-proportion-test",
            {
                "successes": "42",
                "trials": "60",
                "null_proportion": "0.5",
                "alternative": "two-sided",
                "alpha": "0.05",
            },
        )

        self.assertAlmostEqual(result.statistic.raw, 3.3806170189140654)
        self.assertAlmostEqual(result.p_value.raw, 0.0007232327164301953)

    def test_two_sample_proportion_returns_expected_values(self):
        result = calculate_test_statistic(
            "two-sample-proportion-test",
            {
                "successes_a": "42",
                "trials_a": "60",
                "successes_b": "30",
                "trials_b": "55",
                "alternative": "two-sided",
                "alpha": "0.05",
            },
        )

        self.assertAlmostEqual(result.statistic.raw, 1.7110458621009623)
        self.assertAlmostEqual(result.p_value.raw, 0.08707264691294045)
