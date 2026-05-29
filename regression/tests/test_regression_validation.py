from django.test import SimpleTestCase

from domain.enums import DatasetColumnRole
from domain.regression_inputs import RegressionColumn, RegressionDataset, RegressionRow
from services.calculators.registry import calculate_test_statistic
from services.regression_validators import (
    parse_regression_dataset,
    prepare_matching_dataset,
    prepare_supervised_dataset,
    require_binary_target,
    require_multiclass_target,
    validate_role_counts,
)
from services.validators import ValidationIssues, errors_by_field


class RegressionValidationTests(SimpleTestCase):
    def test_dataset_parsing_rejects_duplicate_column_labels(self):
        with self.assertRaises(ValidationIssues) as context:
            calculate_test_statistic(
                "simple-linear-regression",
                {
                    "dataset": {
                        "columns": [
                            {"key": "column_1", "label": "Value", "role": "predictor"},
                            {"key": "column_2", "label": "Value", "role": "target"},
                        ],
                        "rows": [
                            {"cells": ["1", "2"]},
                            {"cells": ["2", "4"]},
                            {"cells": ["3", "6"]},
                        ],
                        "sourceMode": "grid",
                        "filename": "",
                    }
                },
            )

        self.assertEqual(
            errors_by_field(context.exception.issues),
            {"dataset.columns.1.label": ["Column name 'Value' is duplicated."]},
        )

    def test_parse_regression_dataset_parses_json_string_file_mode(self):
        dataset, issues = parse_regression_dataset(
            '{"columns":[{"key":"x","label":"X","role":"predictor"},{"key":"y","label":"Y","role":"target"}],"rows":[{"cells":["1","2"]}],"sourceMode":"file","filename":"sample.csv"}'
        )

        self.assertEqual(issues, [])
        assert dataset is not None
        self.assertEqual(dataset.source_mode, "file")
        self.assertEqual(dataset.filename, "sample.csv")

    def test_parse_regression_dataset_reports_column_role_and_row_shape_issues(self):
        dataset, issues = parse_regression_dataset(
            {
                "columns": [
                    {"key": "a", "label": "A", "role": "predictor"},
                    {"key": "b", "label": "B", "role": "bad-role"},
                    "not-a-column",
                ],
                "rows": [
                    "not-a-row",
                    {"cells": "not-a-list"},
                    {"cells": ["1"]},
                ],
            }
        )

        self.assertIsNone(dataset)
        self.assertEqual(
            errors_by_field(issues),
            {
                "dataset.columns.1.role": ["Choose a valid column role."],
                "dataset.columns.2": ["Each column must be an object."],
                "dataset.rows.0": ["Each row must be an object."],
                "dataset.rows.1.cells": ["Each row must include a cell list."],
                "dataset.rows.2.cells": ["Row 3 must contain exactly 2 cells."],
            },
        )

    def test_validate_role_counts_reports_missing_and_excess_roles(self):
        dataset = RegressionDataset(
            columns=(
                RegressionColumn("x1", "X1", DatasetColumnRole.PREDICTOR),
                RegressionColumn("x2", "X2", DatasetColumnRole.PREDICTOR),
                RegressionColumn("y1", "Y1", DatasetColumnRole.TARGET),
                RegressionColumn("y2", "Y2", DatasetColumnRole.TARGET),
            ),
            rows=(RegressionRow(("1", "2", "3", "4")),),
        )

        issues = validate_role_counts(
            dataset,
            required_roles={DatasetColumnRole.TREATMENT: 1},
            maximum_roles={DatasetColumnRole.TARGET: 1},
        )

        self.assertEqual(
            errors_by_field(issues),
            {
                "dataset.columns": [
                    "Select 1 column(s) with the 'treatment' role.",
                    "Use at most 1 column(s) with the 'target' role.",
                ]
            },
        )

    def test_prepare_supervised_dataset_validates_shape_numeric_cells_and_class_count(self):
        dataset = RegressionDataset(
            columns=(
                RegressionColumn("id", "ID", DatasetColumnRole.ID),
                RegressionColumn("x", "X", DatasetColumnRole.PREDICTOR),
                RegressionColumn("y", "Y", DatasetColumnRole.TARGET),
            ),
            rows=(
                RegressionRow(("", "", "")),
                RegressionRow(("A", "", "yes")),
                RegressionRow(("B", "bad", "yes")),
                RegressionRow(("C", "1.0", "yes", "extra")),
                RegressionRow(("D", "2.0", "yes")),
                RegressionRow(("E", "3.0", "yes")),
            ),
        )

        with self.assertRaises(ValidationIssues) as context:
            prepare_supervised_dataset(
                dataset,
                target_kind="categorical",
                minimum_classes=2,
                min_training_rows=2,
            )

        self.assertEqual(
            errors_by_field(context.exception.issues),
            {
                "dataset.rows.1.cells.1": ["X is required on row 2."],
                "dataset.rows.2.cells.1": ["X must be numeric on row 3."],
                "dataset.rows.3.cells": ["Row 4 has the wrong number of cells."],
                "dataset.rows": ["Y must contain at least 2 classes in the training rows."],
            },
        )

    def test_prepare_matching_dataset_reports_missing_and_nonnumeric_cells(self):
        dataset = RegressionDataset(
            columns=(
                RegressionColumn("id", "ID", DatasetColumnRole.ID),
                RegressionColumn("x", "X", DatasetColumnRole.PREDICTOR),
                RegressionColumn("t", "Treatment", DatasetColumnRole.TREATMENT),
                RegressionColumn("o", "Outcome", DatasetColumnRole.OUTCOME),
            ),
            rows=(
                RegressionRow(("1", "", "A", "10")),
                RegressionRow(("2", "bad", "B", "11")),
                RegressionRow(("3", "2", "", "12")),
                RegressionRow(("4", "3", "A", "bad")),
            ),
        )

        with self.assertRaises(ValidationIssues) as context:
            prepare_matching_dataset(dataset, min_rows=1)

        self.assertEqual(
            errors_by_field(context.exception.issues),
            {
                "dataset.rows.0.cells.1": ["X is required on row 1."],
                "dataset.rows.1.cells.1": ["X must be numeric on row 2."],
                "dataset.rows.2.cells.2": ["Treatment is required on row 3."],
                "dataset.rows.3.cells.3": ["Outcome must be numeric on row 4."],
                "dataset.rows": [
                    "Provide at least 1 populated rows for propensity score matching.",
                    "Treatment must contain exactly two treatment groups.",
                ],
            },
        )

    def test_blank_target_rows_are_allowed_for_prediction(self):
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
                        {"cells": ["4", ""]},
                    ],
                    "sourceMode": "grid",
                    "filename": "",
                }
            },
        )

        self.assertEqual(result.dataset["rows"][-1]["cells"][1], "40")

    def test_binary_logistic_requires_exactly_two_training_classes(self):
        with self.assertRaises(ValidationIssues) as context:
            calculate_test_statistic(
                "binary-logistic-regression",
                {
                    "dataset": {
                        "columns": [
                            {"key": "column_1", "label": "x", "role": "predictor"},
                            {"key": "column_2", "label": "y", "role": "target"},
                        ],
                        "rows": [
                            {"cells": ["1", "A"]},
                            {"cells": ["2", "B"]},
                            {"cells": ["3", "C"]},
                            {"cells": ["4", "A"]},
                        ],
                        "sourceMode": "grid",
                        "filename": "",
                    }
                },
            )

        self.assertEqual(
            errors_by_field(context.exception.issues),
            {"dataset.rows": ["y must contain no more than 2 classes in the training rows."]},
        )

    def test_multinomial_logistic_requires_three_classes(self):
        with self.assertRaises(ValidationIssues) as context:
            calculate_test_statistic(
                "multinomial-logistic-regression",
                {
                    "dataset": {
                        "columns": [
                            {"key": "column_1", "label": "x", "role": "predictor"},
                            {"key": "column_2", "label": "y", "role": "target"},
                        ],
                        "rows": [
                            {"cells": ["1", "A"]},
                            {"cells": ["2", "A"]},
                            {"cells": ["3", "B"]},
                            {"cells": ["4", "B"]},
                            {"cells": ["5", "A"]},
                            {"cells": ["6", "B"]},
                        ],
                        "sourceMode": "grid",
                        "filename": "",
                    }
                },
            )

        self.assertEqual(
            errors_by_field(context.exception.issues),
            {"dataset.rows": ["y must contain at least 3 classes in the training rows."]},
        )

    def test_propensity_score_matching_requires_two_treatment_groups(self):
        with self.assertRaises(ValidationIssues) as context:
            calculate_test_statistic(
                "propensity-score-matching",
                {
                    "dataset": {
                        "columns": [
                            {"key": "column_1", "label": "id", "role": "id"},
                            {"key": "column_2", "label": "x", "role": "predictor"},
                            {"key": "column_3", "label": "treatment", "role": "treatment"},
                            {"key": "column_4", "label": "outcome", "role": "outcome"},
                        ],
                        "rows": [
                            {"cells": ["1", "0.2", "Treated", "10"]},
                            {"cells": ["2", "0.3", "Treated", "11"]},
                            {"cells": ["3", "0.4", "Treated", "12"]},
                            {"cells": ["4", "0.5", "Treated", "13"]},
                        ],
                        "sourceMode": "grid",
                        "filename": "",
                    }
                },
            )

        self.assertEqual(
            errors_by_field(context.exception.issues),
            {"dataset.rows": ["treatment must contain exactly two treatment groups."]},
        )

    def test_parse_regression_dataset_rejects_empty_and_invalid_payload_shapes(self):
        dataset, issues = parse_regression_dataset("   ")
        self.assertIsNone(dataset)
        self.assertEqual(errors_by_field(issues), {"dataset": ["Dataset payload is required."]})

        dataset, issues = parse_regression_dataset("{bad json}")
        self.assertIsNone(dataset)
        self.assertEqual(errors_by_field(issues), {"dataset": ["Dataset payload must be valid JSON."]})

        dataset, issues = parse_regression_dataset([1, 2, 3])
        self.assertIsNone(dataset)
        self.assertEqual(errors_by_field(issues), {"dataset": ["Dataset payload must be an object."]})

    def test_parse_regression_dataset_requires_column_label(self):
        dataset, issues = parse_regression_dataset(
            {
                "columns": [{"key": "x", "label": "", "role": "predictor"}],
                "rows": [{"cells": [None]}],
            }
        )

        self.assertIsNone(dataset)
        self.assertEqual(errors_by_field(issues), {"dataset.columns.0.label": ["Column name is required."]})

    def test_prepare_supervised_dataset_validates_numeric_target_and_prediction_row_minimum(self):
        dataset = RegressionDataset(
            columns=(
                RegressionColumn("x", "X", DatasetColumnRole.PREDICTOR),
                RegressionColumn("y", "Y", DatasetColumnRole.TARGET),
            ),
            rows=(
                RegressionRow(("1", "10")),
                RegressionRow(("2", "bad")),
                RegressionRow(("3", "30")),
            ),
        )

        with self.assertRaises(ValidationIssues) as context:
            prepare_supervised_dataset(dataset, target_kind="numeric", min_training_rows=2, min_prediction_rows=1)

        self.assertEqual(
            errors_by_field(context.exception.issues),
            {
                "dataset.rows.1.cells.1": ["Y must be numeric on row 2."],
                "dataset.rows": ["Provide at least 1 prediction row(s) with a blank target value."],
            },
        )

    def test_prepare_matching_dataset_checks_role_counts_and_group_sizes(self):
        with self.assertRaises(ValidationIssues):
            prepare_matching_dataset(
                RegressionDataset(
                    columns=(
                        RegressionColumn("x", "X", DatasetColumnRole.PREDICTOR),
                        RegressionColumn("o", "Outcome", DatasetColumnRole.OUTCOME),
                    ),
                    rows=(RegressionRow(("1", "2")),),
                )
            )

        with self.assertRaises(ValidationIssues) as context:
            prepare_matching_dataset(
                RegressionDataset(
                    columns=(
                        RegressionColumn("id", "ID", DatasetColumnRole.ID),
                        RegressionColumn("x", "X", DatasetColumnRole.PREDICTOR),
                        RegressionColumn("t", "Treatment", DatasetColumnRole.TREATMENT),
                        RegressionColumn("o", "Outcome", DatasetColumnRole.OUTCOME),
                    ),
                    rows=(
                        RegressionRow(("", "", "", "")),
                        RegressionRow(("T1", "1", "Treated", "10")),
                        RegressionRow(("T2", "2", "Treated", "11")),
                        RegressionRow(("C1", "3", "Control", "12")),
                    ),
                ),
                min_rows=3,
            )

        self.assertEqual(
            errors_by_field(context.exception.issues),
            {"dataset.rows": ["Each treatment group in Treatment must contain at least two rows."]},
        )

    def test_binary_and_multiclass_target_requirements(self):
        prepared_binary = prepare_supervised_dataset(
            RegressionDataset(
                columns=(
                    RegressionColumn("x", "X", DatasetColumnRole.PREDICTOR),
                    RegressionColumn("y", "Y", DatasetColumnRole.TARGET),
                ),
                rows=(
                    RegressionRow(("1", "A")),
                    RegressionRow(("2", "A")),
                    RegressionRow(("3", "A")),
                ),
            ),
            target_kind="categorical",
            min_training_rows=3,
        )
        with self.assertRaises(ValidationIssues):
            require_binary_target(prepared_binary)

        prepared_multi = prepare_supervised_dataset(
            RegressionDataset(
                columns=(
                    RegressionColumn("x", "X", DatasetColumnRole.PREDICTOR),
                    RegressionColumn("y", "Y", DatasetColumnRole.TARGET),
                ),
                rows=(
                    RegressionRow(("1", "A")),
                    RegressionRow(("2", "B")),
                    RegressionRow(("3", "B")),
                ),
            ),
            target_kind="categorical",
            min_training_rows=3,
        )
        with self.assertRaises(ValidationIssues):
            require_multiclass_target(prepared_multi)

    def test_parse_regression_dataset_requires_columns_and_rows_lists(self):
        dataset, issues = parse_regression_dataset({"columns": "bad", "rows": "bad"})
        self.assertIsNone(dataset)
        self.assertEqual(
            errors_by_field(issues),
            {
                "dataset.columns": ["Add at least one dataset column."],
                "dataset.rows": ["Add at least one dataset row."],
            },
        )

    def test_prepare_supervised_dataset_raises_for_missing_target_role_and_too_few_training_rows(self):
        with self.assertRaises(ValidationIssues):
            prepare_supervised_dataset(
                RegressionDataset(
                    columns=(RegressionColumn("x", "X", DatasetColumnRole.PREDICTOR),),
                    rows=(RegressionRow(("1",)),),
                )
            )

        with self.assertRaises(ValidationIssues) as context:
            prepare_supervised_dataset(
                RegressionDataset(
                    columns=(
                        RegressionColumn("x", "X", DatasetColumnRole.PREDICTOR),
                        RegressionColumn("y", "Y", DatasetColumnRole.TARGET),
                    ),
                    rows=(RegressionRow(("1", "10")), RegressionRow(("2", ""))),
                ),
                min_training_rows=2,
                min_prediction_rows=0,
            )

        self.assertEqual(
            errors_by_field(context.exception.issues),
            {"dataset.rows": ["Provide at least 2 training row(s) with a filled target value."]},
        )

    def test_prepare_matching_dataset_requires_outcome_value(self):
        with self.assertRaises(ValidationIssues) as context:
            prepare_matching_dataset(
                RegressionDataset(
                    columns=(
                        RegressionColumn("x", "X", DatasetColumnRole.PREDICTOR),
                        RegressionColumn("t", "Treatment", DatasetColumnRole.TREATMENT),
                        RegressionColumn("o", "Outcome", DatasetColumnRole.OUTCOME),
                    ),
                    rows=(
                        RegressionRow(("1", "A", "")),
                        RegressionRow(("2", "A", "10")),
                        RegressionRow(("3", "B", "11")),
                        RegressionRow(("4", "B", "12")),
                        RegressionRow(("5", "A", "13")),
                    ),
                )
            )

        self.assertEqual(
            errors_by_field(context.exception.issues),
            {"dataset.rows.0.cells.2": ["Outcome is required on row 1."]},
        )

    def test_binary_and_multiclass_requirement_helpers_allow_valid_class_counts(self):
        prepared_binary = prepare_supervised_dataset(
            RegressionDataset(
                columns=(
                    RegressionColumn("x", "X", DatasetColumnRole.PREDICTOR),
                    RegressionColumn("y", "Y", DatasetColumnRole.TARGET),
                ),
                rows=(RegressionRow(("1", "A")), RegressionRow(("2", "B")), RegressionRow(("3", "A"))),
            ),
            target_kind="categorical",
            min_training_rows=3,
        )
        require_binary_target(prepared_binary)

        prepared_multi = prepare_supervised_dataset(
            RegressionDataset(
                columns=(
                    RegressionColumn("x", "X", DatasetColumnRole.PREDICTOR),
                    RegressionColumn("y", "Y", DatasetColumnRole.TARGET),
                ),
                rows=(
                    RegressionRow(("1", "A")),
                    RegressionRow(("2", "B")),
                    RegressionRow(("3", "C")),
                ),
            ),
            target_kind="categorical",
            min_training_rows=3,
        )
        require_multiclass_target(prepared_multi)
