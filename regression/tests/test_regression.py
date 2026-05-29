import pandas as pd
from django.test import SimpleTestCase

from domain.enums import DatasetColumnRole
from domain.regression_inputs import PreparedMatchingDataset, PreparedRegressionDataset, RegressionColumn, RegressionDataset, RegressionRow
from sklearn.linear_model import LinearRegression

from services.calculators.regression import (
    _balance_table,
    _build_linear_result,
    _class_balance_warning,
    _classification_prediction_table,
    _prediction_table,
    _select_treated_and_control_labels,
    _standardized_mean_difference,
)
from services.calculators.registry import calculate_test_statistic
from services.calculators.regression import BulkLinearRegressionCalculator


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

    def test_simple_linear_without_prediction_rows_has_no_prediction_table(self):
        result = calculate_test_statistic(
            "simple-linear-regression",
            {
                "dataset": {
                    "columns": [
                        {"key": "column_1", "label": "x", "role": "predictor"},
                        {"key": "column_2", "label": "y", "role": "target"},
                    ],
                    "rows": [
                        {"cells": ["1", "10"]},
                        {"cells": ["2", "20"]},
                        {"cells": ["3", "30"]},
                    ],
                    "sourceMode": "grid",
                    "filename": "",
                }
            },
        )

        self.assertEqual(len(result.tables), 1)

    def test_binary_logistic_without_prediction_rows_has_no_prediction_table(self):
        result = calculate_test_statistic(
            "binary-logistic-regression",
            {
                "dataset": {
                    "columns": [
                        {"key": "column_1", "label": "x", "role": "predictor"},
                        {"key": "column_2", "label": "y", "role": "target"},
                    ],
                    "rows": [
                        {"cells": ["1", "No"]},
                        {"cells": ["2", "No"]},
                        {"cells": ["3", "Yes"]},
                        {"cells": ["4", "Yes"]},
                    ],
                    "sourceMode": "grid",
                    "filename": "",
                }
            },
        )

        self.assertEqual(len(result.tables), 1)

    def test_propensity_score_matching_warns_about_unmatched_treated_rows(self):
        result = calculate_test_statistic(
            "propensity-score-matching",
            {
                "dataset": {
                    "columns": [
                        {"key": "column_1", "label": "id", "role": "id"},
                        {"key": "column_2", "label": "age", "role": "predictor"},
                        {"key": "column_3", "label": "treatment", "role": "treatment"},
                        {"key": "column_4", "label": "outcome", "role": "outcome"},
                    ],
                    "rows": [
                        {"cells": ["T1", "40", "treated", "10"]},
                        {"cells": ["T2", "42", "treated", "11"]},
                        {"cells": ["T3", "44", "treated", "12"]},
                        {"cells": ["C1", "41", "control", "9"]},
                        {"cells": ["C2", "43", "control", "8"]},
                    ],
                    "sourceMode": "grid",
                    "filename": "",
                }
            },
        )

        self.assertTrue(any("could not be matched" in warning for warning in result.warnings))

    def test_class_balance_warning_handles_empty_and_imbalanced_training_frames(self):
        dataset = RegressionDataset(
            columns=(
                RegressionColumn("x", "X", DatasetColumnRole.PREDICTOR),
                RegressionColumn("y", "Y", DatasetColumnRole.TARGET),
            ),
            rows=(RegressionRow(("1", "A")),),
        )
        prepared = PreparedRegressionDataset(
            dataset=dataset,
            predictor_columns=(dataset.columns[0],),
            target_column=dataset.columns[1],
            id_column=None,
            training_frame=pd.DataFrame(columns=["x", "y"]),
            prediction_frame=pd.DataFrame(),
            training_row_indices=(),
            prediction_row_indices=(),
        )
        self.assertEqual(_class_balance_warning(prepared), ())

        prepared_imbalanced = PreparedRegressionDataset(
            dataset=dataset,
            predictor_columns=(dataset.columns[0],),
            target_column=dataset.columns[1],
            id_column=None,
            training_frame=pd.DataFrame({"x": [1, 2, 3, 4, 5, 6], "y": ["A", "A", "A", "A", "A", "B"]}),
            prediction_frame=pd.DataFrame(),
            training_row_indices=(0, 1, 2, 3, 4, 5),
            prediction_row_indices=(),
        )
        self.assertTrue(_class_balance_warning(prepared_imbalanced))

    def test_prediction_helpers_return_none_when_no_prediction_rows(self):
        dataset = RegressionDataset(
            columns=(
                RegressionColumn("x", "X", DatasetColumnRole.PREDICTOR),
                RegressionColumn("y", "Y", DatasetColumnRole.TARGET),
            ),
            rows=(RegressionRow(("1", "A")),),
        )
        prepared = PreparedRegressionDataset(
            dataset=dataset,
            predictor_columns=(dataset.columns[0],),
            target_column=dataset.columns[1],
            id_column=None,
            training_frame=pd.DataFrame({"x": [1], "y": ["A"]}),
            prediction_frame=pd.DataFrame(),
            training_row_indices=(0,),
            prediction_row_indices=(),
        )

        self.assertIsNone(_prediction_table(prepared, pd.Series(dtype=float)))
        self.assertIsNone(
            _classification_prediction_table(
                prepared,
                pd.Series(dtype=object),
                pd.DataFrame(columns=["A"]).to_numpy(),
                pd.Series(["A"]).to_numpy(),
            )
        )

    def test_matching_helpers_cover_fallback_and_empty_pairs(self):
        self.assertEqual(_select_treated_and_control_labels(pd.Series(["Beta", "Alpha"]).to_numpy()), ("Beta", "Alpha"))
        self.assertEqual(_standardized_mean_difference(pd.Series([1.0, 1.0]).to_numpy(), pd.Series([1.0, 1.0]).to_numpy()), 0.0)

        dataset = RegressionDataset(
            columns=(
                RegressionColumn("x", "X", DatasetColumnRole.PREDICTOR),
                RegressionColumn("t", "T", DatasetColumnRole.TREATMENT),
                RegressionColumn("o", "O", DatasetColumnRole.OUTCOME),
            ),
            rows=(RegressionRow(("1", "A", "10")), RegressionRow(("2", "B", "11"))),
        )
        prepared = PreparedMatchingDataset(
            dataset=dataset,
            predictor_columns=(dataset.columns[0],),
            treatment_column=dataset.columns[1],
            outcome_column=dataset.columns[2],
            id_column=None,
            dataframe=pd.DataFrame(),
        )
        treated = pd.DataFrame({"__row_id__": ["T1"], "x": [1.0]})
        control = pd.DataFrame({"__row_id__": ["C1"], "x": [1.0]})
        table = _balance_table(prepared, treated, control, ())
        self.assertEqual(table.rows[0][2], "0")

    def test_build_linear_result_requires_prediction_rows_when_requested(self):
        dataset = RegressionDataset(
            columns=(
                RegressionColumn("x", "X", DatasetColumnRole.PREDICTOR),
                RegressionColumn("y", "Y", DatasetColumnRole.TARGET),
            ),
            rows=(RegressionRow(("1", "2")), RegressionRow(("2", "4")), RegressionRow(("3", "6"))),
        )
        prepared = PreparedRegressionDataset(
            dataset=dataset,
            predictor_columns=(dataset.columns[0],),
            target_column=dataset.columns[1],
            id_column=None,
            training_frame=pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]}),
            prediction_frame=pd.DataFrame(),
            training_row_indices=(0, 1, 2),
            prediction_row_indices=(),
        )
        model = LinearRegression().fit(prepared.training_frame[["x"]], prepared.training_frame["y"])

        with self.assertRaises(ValueError):
            _build_linear_result(
                prepared,
                metadata=BulkLinearRegressionCalculator.metadata,
                model=model,
                r_squared=1.0,
                rmse=0.0,
                interpretation="test",
                notes=(),
                metrics=(),
                require_prediction_rows=True,
            )
