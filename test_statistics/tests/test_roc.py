from django.test import SimpleTestCase

from services.calculators.registry import calculate_test_statistic
from services.calculators.roc import _decision, _z_test_from_difference


class RocCalculatorTests(SimpleTestCase):
    def test_independent_delong_returns_expected_values(self):
        result = calculate_test_statistic(
            "delong-test-independent-curves",
            {
                "curve_a": "1, 0.90\n1, 0.80\n1, 0.65\n0, 0.70\n0, 0.45\n0, 0.20",
                "curve_b": "1, 0.88\n1, 0.72\n1, 0.60\n0, 0.75\n0, 0.50\n0, 0.25",
                "alpha": "0.05",
            },
        )

        self.assertAlmostEqual(result.statistic.raw, 0.3779644730092274)
        self.assertAlmostEqual(result.p_value.raw, 0.7054569861112732)

    def test_paired_delong_returns_expected_values(self):
        result = calculate_test_statistic(
            "delong-test-paired-curves",
            {
                "rows": "1, 0.90, 0.88\n1, 0.80, 0.72\n1, 0.65, 0.60\n0, 0.70, 0.75\n0, 0.45, 0.50\n0, 0.20, 0.25",
                "alpha": "0.05",
            },
        )

        self.assertAlmostEqual(result.statistic.raw, 0.7071067811865479)
        self.assertAlmostEqual(result.p_value.raw, 0.47950012218695326)

    def test_zero_variance_difference_returns_neutral_z_test_result(self):
        statistic, p_value = _z_test_from_difference(0.5, 0.0)

        self.assertEqual(statistic, 0.0)
        self.assertEqual(p_value, 1.0)

    def test_decision_helper_returns_fail_to_reject_text(self):
        conclusion = _decision("ROC comparison", reject_null=False)

        self.assertIn("Fail to reject", conclusion)

    def test_decision_helper_returns_reject_text(self):
        conclusion = _decision("ROC comparison", reject_null=True)

        self.assertIn("Reject", conclusion)
