from django.test import SimpleTestCase

from services.calculators.registry import calculate_test_statistic
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
