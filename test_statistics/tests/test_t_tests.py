from django.test import SimpleTestCase

from services.calculators.registry import calculate_test_statistic


class TTestCalculatorTests(SimpleTestCase):
    def test_one_sample_z_test_returns_expected_values(self):
        result = calculate_test_statistic(
            "one-sample-z-test",
            {
                "sample": "12, 15, 14, 13, 16",
                "population_mean": "10",
                "known_std": "2",
                "alternative": "two-sided",
                "alpha": "0.05",
            },
        )

        self.assertAlmostEqual(result.statistic.raw, 4.47213595499958)
        self.assertAlmostEqual(result.p_value.raw, 7.74421643104407e-06)

    def test_one_sample_t_test_returns_expected_values(self):
        result = calculate_test_statistic(
            "one-sample-t-test",
            {
                "sample": "12, 15, 14, 13, 16",
                "population_mean": "10",
                "alternative": "two-sided",
                "alpha": "0.05",
            },
        )

        self.assertAlmostEqual(result.statistic.raw, 5.65685424949238)
        self.assertAlmostEqual(result.p_value.raw, 0.004812678330044224)

    def test_two_sample_z_test_returns_expected_values(self):
        result = calculate_test_statistic(
            "two-sample-z-test",
            {
                "sample_a": "10, 12, 13, 11, 9",
                "sample_b": "16, 18, 15, 17, 19",
                "known_std_a": "2",
                "known_std_b": "2",
                "alternative": "two-sided",
                "alpha": "0.05",
            },
        )

        self.assertAlmostEqual(result.statistic.raw, -4.743416490252569)
        self.assertAlmostEqual(result.p_value.raw, 2.1014359560124373e-06)

    def test_two_sample_pooled_t_test_returns_expected_values(self):
        result = calculate_test_statistic(
            "two-sample-t-test-pooled",
            {
                "sample_a": "10, 12, 13, 11, 9",
                "sample_b": "16, 18, 15, 17, 19",
                "alternative": "two-sided",
                "alpha": "0.05",
            },
        )

        self.assertAlmostEqual(result.statistic.raw, -6.0)
        self.assertAlmostEqual(result.p_value.raw, 0.0003233932218851489)

    def test_welch_t_test_returns_expected_values(self):
        result = calculate_test_statistic(
            "two-sample-t-test-welch",
            {
                "sample_a": "10, 12, 13, 11, 9",
                "sample_b": "16, 18, 15, 17, 19",
                "alternative": "two-sided",
                "alpha": "0.05",
            },
        )

        self.assertAlmostEqual(result.statistic.raw, -6.0)
        self.assertAlmostEqual(result.p_value.raw, 0.0003233932218851489)

    def test_paired_t_test_returns_expected_values(self):
        result = calculate_test_statistic(
            "paired-t-test",
            {
                "sample_a": "12, 14, 13, 15",
                "sample_b": "11, 13, 11, 14",
                "alternative": "two-sided",
                "alpha": "0.05",
            },
        )

        self.assertAlmostEqual(result.statistic.raw, 5.0)
        self.assertAlmostEqual(result.p_value.raw, 0.015392438073302294)

    def test_directional_decision_branch_for_less_alternative(self):
        result = calculate_test_statistic(
            "one-sample-z-test",
            {
                "sample": "1, 2, 3, 4, 5",
                "population_mean": "10",
                "known_std": "1",
                "alternative": "less",
                "alpha": "0.05",
            },
        )

        self.assertTrue(result.decision.reject_null)
        self.assertIn("'less' direction", result.decision.conclusion)

    def test_one_sample_z_test_greater_branch_can_fail_to_reject(self):
        result = calculate_test_statistic(
            "one-sample-z-test",
            {
                "sample": "10, 10, 10, 10, 10",
                "population_mean": "10",
                "known_std": "1",
                "alternative": "greater",
                "alpha": "0.05",
            },
        )

        self.assertFalse(result.decision.reject_null)
        self.assertIn("Fail to reject", result.decision.conclusion)

    def test_two_sample_z_test_less_branch(self):
        result = calculate_test_statistic(
            "two-sample-z-test",
            {
                "sample_a": "1, 1, 1, 1, 1",
                "sample_b": "9, 9, 9, 9, 9",
                "known_std_a": "1",
                "known_std_b": "1",
                "alternative": "less",
                "alpha": "0.05",
            },
        )

        self.assertLess(result.p_value.raw, 0.05)

    def test_two_sample_z_test_greater_branch(self):
        result = calculate_test_statistic(
            "two-sample-z-test",
            {
                "sample_a": "9, 9, 9, 9, 9",
                "sample_b": "1, 1, 1, 1, 1",
                "known_std_a": "1",
                "known_std_b": "1",
                "alternative": "greater",
                "alpha": "0.05",
            },
        )

        self.assertLess(result.p_value.raw, 0.05)

    def test_two_sample_pooled_t_test_directional_branches(self):
        greater = calculate_test_statistic(
            "two-sample-t-test-pooled",
            {
                "sample_a": "9, 10, 11, 9, 10",
                "sample_b": "1, 2, 3, 1, 2",
                "alternative": "greater",
                "alpha": "0.05",
            },
        )
        less = calculate_test_statistic(
            "two-sample-t-test-pooled",
            {
                "sample_a": "1, 2, 3, 1, 2",
                "sample_b": "9, 10, 11, 9, 10",
                "alternative": "less",
                "alpha": "0.05",
            },
        )

        self.assertLess(greater.p_value.raw, 0.05)
        self.assertLess(less.p_value.raw, 0.05)
