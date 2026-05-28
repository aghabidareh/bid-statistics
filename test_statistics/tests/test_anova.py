from django.test import SimpleTestCase

from test_statistics.services.calculators.registry import calculate_test_statistic


class AnovaCalculatorTests(SimpleTestCase):
    def test_one_way_anova_returns_expected_values(self):
        result = calculate_test_statistic(
            "one-way-anova",
            {
                "groups": "Control: 4, 5, 6\nTreatment A: 6, 7, 8\nTreatment B: 8, 9, 10",
                "alpha": "0.05",
            },
        )

        self.assertAlmostEqual(result.statistic.raw, 12.0)
        self.assertAlmostEqual(result.p_value.raw, 0.008000000000000002)

    def test_repeated_measures_anova_returns_expected_values(self):
        result = calculate_test_statistic(
            "repeated-measures-anova",
            {
                "rows": "S1, Baseline, 4\nS1, Mid, 6\nS1, Final, 7\nS2, Baseline, 5\nS2, Mid, 6\nS2, Final, 8\nS3, Baseline, 6\nS3, Mid, 7\nS3, Final, 9\nS4, Baseline, 5\nS4, Mid, 7\nS4, Final, 8",
                "alpha": "0.05",
            },
        )

        self.assertAlmostEqual(result.statistic.raw, 81.0)
        self.assertAlmostEqual(result.p_value.raw, 4.55539358600583e-05)
        self.assertEqual(result.tables[0].title, "Repeated-measures ANOVA table")

    def test_two_way_anova_returns_expected_values(self):
        result = calculate_test_statistic(
            "two-way-anova",
            {
                "rows": "Low, Control, 4\nLow, Control, 5\nLow, Treatment, 7\nLow, Treatment, 9\nHigh, Control, 6\nHigh, Control, 7\nHigh, Treatment, 10\nHigh, Treatment, 12",
                "alpha": "0.05",
            },
        )

        self.assertAlmostEqual(result.statistic.raw, 0.3999999999999981)
        self.assertAlmostEqual(result.p_value.raw, 0.5614380442505266)
        self.assertEqual(len(result.tables), 2)

    def test_one_way_manova_returns_expected_values(self):
        result = calculate_test_statistic(
            "one-way-manova",
            {
                "variable_names": "score_1, score_2",
                "rows": "Control, 10, 15\nControl, 11, 14\nControl, 9, 16\nTreatment, 15, 18\nTreatment, 16, 19\nTreatment, 17, 20",
                "alpha": "0.05",
            },
        )

        self.assertAlmostEqual(result.statistic.raw, 0.9512195121951117)
        self.assertAlmostEqual(result.p_value.raw, 0.010773807421939646)
        self.assertEqual(result.statistic_name, "Pillai's trace")
