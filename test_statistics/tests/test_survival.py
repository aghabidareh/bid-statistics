from django.test import SimpleTestCase

from services.calculators.registry import calculate_test_statistic


class SurvivalCalculatorTests(SimpleTestCase):
    def test_kaplan_meier_returns_expected_summary(self):
        result = calculate_test_statistic(
            "kaplan-meier-survival-analysis",
            {
                "rows": "5, 1\n8, 0\n12, 1\n15, 1\n20, 0",
                "alpha": "0.05",
            },
        )

        self.assertEqual(result.statistic_name, "Median survival time")
        self.assertEqual(result.statistic.raw, 15.0)
        self.assertIsNone(result.p_value)
        self.assertEqual(len(result.tables), 2)
        self.assertEqual(result.sections[0].title, "Survival summary")
