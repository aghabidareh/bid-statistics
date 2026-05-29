from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase
import pandas as pd

from domain.inputs import KaplanMeierInput, SurvivalObservation
from services.calculators.registry import calculate_test_statistic
from services.calculators.survival import KaplanMeierCalculator


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

    def test_kaplan_meier_reports_infinite_median_when_no_events_observed(self):
        result = calculate_test_statistic(
            "kaplan-meier-survival-analysis",
            {
                "rows": "5, 0\n8, 0\n12, 0\n15, 0\n20, 0",
                "alpha": "0.05",
            },
        )

        self.assertEqual(result.statistic.raw, float("inf"))
        self.assertEqual(result.notes, ())

    def test_kaplan_meier_adds_note_when_median_is_nan(self):
        calculator = KaplanMeierCalculator.instance()
        fake_fitter = MagicMock()
        fake_fitter.fit.return_value = fake_fitter
        fake_fitter.confidence_interval_ = pd.DataFrame({"Overall survival_lower_0.95": [0.8], "Overall survival_upper_0.95": [1.0]}, index=[0.0])
        fake_fitter.survival_function_ = pd.DataFrame({"Overall survival": [1.0]}, index=[0.0])
        fake_fitter.event_table = pd.DataFrame({"at_risk": [1], "observed": [0], "censored": [1], "removed": [1]}, index=pd.Index([0.0], name="event_at"))
        fake_fitter.median_survival_time_ = float("nan")

        with patch("services.calculators.survival.KaplanMeierFitter", return_value=fake_fitter):
            result = calculator.calculate_result(
                KaplanMeierInput(
                    observations=(SurvivalObservation(time=5.0, event=0),),
                    alpha=0.05,
                )
            )

        self.assertEqual(result.statistic.display, "Not reached")
        self.assertEqual(result.notes, ("Median survival was not reached within the observed follow-up window.",))
