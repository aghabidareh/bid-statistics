from django.test import SimpleTestCase

from domain.enums import DatasetColumnRole
from domain.regression_inputs import RegressionDataset
from domain.results import DecisionSummary, ValidationIssue, display_number, display_p_value


class ResultHelpersTests(SimpleTestCase):
    def test_validation_issue_to_dict(self):
        issue = ValidationIssue(field="alpha", message="must be between 0 and 1")
        self.assertEqual(issue.to_dict(), {"field": "alpha", "message": "must be between 0 and 1"})

    def test_decision_summary_to_dict_uses_reject_null_key(self):
        summary = DecisionSummary(alpha=0.05, reject_null=True, conclusion="Reject H0")
        self.assertEqual(
            summary.to_dict(),
            {"alpha": 0.05, "rejectNull": True, "conclusion": "Reject H0"},
        )

    def test_display_helpers_handle_none(self):
        self.assertEqual(display_number(None).to_dict(), {"raw": None, "display": "—"})
        self.assertEqual(display_p_value(None).to_dict(), {"raw": None, "display": "—"})


class RegressionDatasetTests(SimpleTestCase):
    def test_from_dict_applies_defaults_and_normalizes_cells(self):
        dataset = RegressionDataset.from_dict(
            {
                "columns": [{"label": " X "}, {"key": "y", "label": "Y", "role": DatasetColumnRole.TARGET.value}],
                "rows": [{"cells": [" 1 ", None]}, {"cells": [2, " 3 "]}],
                "sourceMode": None,
                "filename": None,
            }
        )

        self.assertEqual(dataset.columns[0].key, "column_1")
        self.assertEqual(dataset.columns[0].label, "X")
        self.assertEqual(dataset.columns[0].role, DatasetColumnRole.PREDICTOR)
        self.assertEqual(dataset.columns[1].role, DatasetColumnRole.TARGET)
        self.assertEqual(dataset.rows[0].cells, ("1", ""))
        self.assertEqual(dataset.rows[1].cells, ("2", "3"))
        self.assertEqual(dataset.source_mode, "grid")
        self.assertEqual(dataset.filename, "")