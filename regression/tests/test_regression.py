from django.test import SimpleTestCase

from services.calculators.registry import calculate_test_statistic


class RegressionCalculatorTests(SimpleTestCase):
    def test_simple_linear_regression_fills_blank_prediction_rows(self):
        result = calculate_test_statistic(
            "simple-linear-regression",
            {
                "dataset": {
                    "columns": [
                        {"key": "column_1", "label": "x", "role": "predictor"},
                        {"key": "column_2", "label": "y", "role": "target"},
                    ],
                    "rows": [
                        {"cells": ["1", "15"]},
                        {"cells": ["2", "25"]},
                        {"cells": ["3", "35"]},
                        {"cells": ["4", "45"]},
                        {"cells": ["5", "55"]},
                        {"cells": ["6", ""]},
                    ],
                    "sourceMode": "grid",
                    "filename": "",
                }
            },
        )

        self.assertEqual(result.statistic_name, "R²")
        self.assertEqual(result.statistic.display, "1")
        self.assertEqual(result.dataset["rows"][-1]["cells"][1], "65")
        self.assertEqual(result.tables[1].title, "Predicted rows")

    def test_multiple_linear_regression_returns_coefficients_and_predictions(self):
        result = calculate_test_statistic(
            "multiple-linear-regression",
            {
                "dataset": {
                    "columns": [
                        {"key": "column_1", "label": "x1", "role": "predictor"},
                        {"key": "column_2", "label": "x2", "role": "predictor"},
                        {"key": "column_3", "label": "y", "role": "target"},
                    ],
                    "rows": [
                        {"cells": ["1", "1", "15"]},
                        {"cells": ["2", "1", "17"]},
                        {"cells": ["1", "2", "18"]},
                        {"cells": ["2", "2", "20"]},
                        {"cells": ["3", "2", "22"]},
                        {"cells": ["3", "3", ""]},
                    ],
                    "sourceMode": "grid",
                    "filename": "",
                }
            },
        )

        self.assertEqual(result.tables[0].title, "Coefficients")
        self.assertEqual(result.tables[1].title, "Predicted rows")
        self.assertEqual(result.dataset["rows"][-1]["cells"][2], "25")

    def test_bulk_linear_regression_predicts_multiple_rows(self):
        result = calculate_test_statistic(
            "bulk-linear-regression",
            {
                "dataset": {
                    "columns": [
                        {"key": "column_1", "label": "x1", "role": "predictor"},
                        {"key": "column_2", "label": "x2", "role": "predictor"},
                        {"key": "column_3", "label": "y", "role": "target"},
                    ],
                    "rows": [
                        {"cells": ["1", "1", "12"]},
                        {"cells": ["2", "1", "14"]},
                        {"cells": ["1", "2", "15"]},
                        {"cells": ["2", "2", "17"]},
                        {"cells": ["3", "2", "19"]},
                        {"cells": ["3", "3", ""]},
                        {"cells": ["4", "3", ""]},
                    ],
                    "sourceMode": "grid",
                    "filename": "",
                }
            },
        )

        self.assertEqual(result.metrics[2].value, "2")
        self.assertEqual(result.tables[1].rows[0][-1], "22")
        self.assertEqual(result.tables[1].rows[1][-1], "24")

    def test_binary_logistic_regression_returns_class_predictions(self):
        result = calculate_test_statistic(
            "binary-logistic-regression",
            {
                "dataset": {
                    "columns": [
                        {"key": "column_1", "label": "age", "role": "predictor"},
                        {"key": "column_2", "label": "income", "role": "predictor"},
                        {"key": "column_3", "label": "label", "role": "target"},
                    ],
                    "rows": [
                        {"cells": ["20", "30", "No"]},
                        {"cells": ["24", "35", "No"]},
                        {"cells": ["29", "41", "No"]},
                        {"cells": ["36", "55", "Yes"]},
                        {"cells": ["42", "63", "Yes"]},
                        {"cells": ["48", "72", "Yes"]},
                        {"cells": ["45", "68", ""]},
                    ],
                    "sourceMode": "grid",
                    "filename": "",
                }
            },
        )

        self.assertEqual(result.statistic_name, "Accuracy")
        self.assertEqual(result.tables[0].title, "Coefficients")
        self.assertEqual(result.tables[1].title, "Predicted rows")
        self.assertEqual(result.dataset["rows"][-1]["cells"][2], "Yes")

    def test_multinomial_logistic_regression_returns_probability_table(self):
        result = calculate_test_statistic(
            "multinomial-logistic-regression",
            {
                "dataset": {
                    "columns": [
                        {"key": "column_1", "label": "tenure", "role": "predictor"},
                        {"key": "column_2", "label": "usage", "role": "predictor"},
                        {"key": "column_3", "label": "plan", "role": "target"},
                    ],
                    "rows": [
                        {"cells": ["2", "10", "Basic"]},
                        {"cells": ["4", "15", "Basic"]},
                        {"cells": ["8", "28", "Standard"]},
                        {"cells": ["10", "34", "Standard"]},
                        {"cells": ["14", "50", "Premium"]},
                        {"cells": ["18", "62", "Premium"]},
                        {"cells": ["11", "39", "Standard"]},
                        {"cells": ["16", "57", ""]},
                    ],
                    "sourceMode": "grid",
                    "filename": "",
                }
            },
        )

        self.assertEqual(result.tables[1].title, "Predicted rows")
        self.assertEqual(result.dataset["rows"][-1]["cells"][2], "Premium")
        self.assertIn("P(Basic)", result.tables[1].columns)
        self.assertIn("P(Premium)", result.tables[1].columns)

    def test_propensity_score_matching_returns_pairs_and_balance(self):
        result = calculate_test_statistic(
            "propensity-score-matching",
            {
                "dataset": {
                    "columns": [
                        {"key": "column_1", "label": "id", "role": "id"},
                        {"key": "column_2", "label": "age", "role": "predictor"},
                        {"key": "column_3", "label": "risk", "role": "predictor"},
                        {"key": "column_4", "label": "treatment", "role": "treatment"},
                        {"key": "column_5", "label": "outcome", "role": "outcome"},
                    ],
                    "rows": [
                        {"cells": ["T1", "44", "0.82", "Treated", "78"]},
                        {"cells": ["T2", "39", "0.74", "Treated", "73"]},
                        {"cells": ["T3", "47", "0.91", "Treated", "81"]},
                        {"cells": ["T4", "41", "0.69", "Treated", "76"]},
                        {"cells": ["C1", "43", "0.80", "Control", "69"]},
                        {"cells": ["C2", "37", "0.71", "Control", "68"]},
                        {"cells": ["C3", "49", "0.89", "Control", "72"]},
                        {"cells": ["C4", "40", "0.67", "Control", "70"]},
                    ],
                    "sourceMode": "grid",
                    "filename": "",
                }
            },
        )

        self.assertEqual(result.statistic_name, "Matched pairs")
        self.assertEqual(result.statistic.display, "4")
        self.assertEqual(result.tables[1].title, "Matched pairs")
        self.assertEqual(result.tables[2].title, "Balance summary")
        self.assertEqual(len(result.tables[1].rows), 4)
