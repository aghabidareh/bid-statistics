import json

from django.test import SimpleTestCase


class RegressionViewTests(SimpleTestCase):
    def inertia_get(self, path: str):
        return self.client.get(path, HTTP_X_INERTIA="true")

    def inertia_post(self, path: str, data: dict[str, str]):
        return self.client.post(path, data, HTTP_X_INERTIA="true")

    def test_regression_index_returns_six_calculator_catalog(self):
        response = self.inertia_get("/regression/")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["component"], "Regression/Index")
        self.assertEqual(len(payload["props"]["catalog"]), 6)
        self.assertEqual(payload["props"]["catalog"][0]["slug"], "simple-linear-regression")
        self.assertEqual(payload["props"]["catalog"][-1]["slug"], "propensity-score-matching")

    def test_regression_show_page_returns_dataset_defaults(self):
        response = self.inertia_get("/regression/simple-linear-regression/")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["component"], "Regression/Show")
        self.assertEqual(payload["props"]["calculator"]["slug"], "simple-linear-regression")
        self.assertEqual(payload["props"]["calculator"]["workflowKind"], "dataset")
        self.assertEqual(len(payload["props"]["form"]["values"]["dataset"]["columns"]), 2)
        self.assertEqual(payload["props"]["form"]["values"]["dataset"]["columns"][1]["role"], "target")
        self.assertIsNone(payload["props"]["result"])

    def test_regression_calculate_returns_structured_prediction_result(self):
        dataset = {
            "columns": [
                {"key": "column_1", "label": "Hours studied", "role": "predictor"},
                {"key": "column_2", "label": "Exam score", "role": "target"},
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
        response = self.inertia_post(
            "/regression/simple-linear-regression/calculate/",
            {"dataset": json.dumps(dataset)},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["component"], "Regression/Show")
        self.assertEqual(payload["props"]["calculator"]["slug"], "simple-linear-regression")
        self.assertEqual(payload["props"]["result"]["statisticName"], "R²")
        self.assertEqual(payload["props"]["result"]["tables"][1]["title"], "Predicted rows")
        self.assertEqual(payload["props"]["result"]["dataset"]["rows"][-1]["cells"][1], "65")

    def test_regression_validation_errors_are_row_aware(self):
        dataset = {
            "columns": [
                {"key": "column_1", "label": "Age", "role": "predictor"},
                {"key": "column_2", "label": "Income", "role": "predictor"},
                {"key": "column_3", "label": "Subscribed", "role": "target"},
            ],
            "rows": [
                {"cells": ["24", "35", "No"]},
                {"cells": ["29", "bad", "Yes"]},
                {"cells": ["31", "44", ""]},
            ],
            "sourceMode": "grid",
            "filename": "",
        }
        response = self.inertia_post(
            "/regression/binary-logistic-regression/calculate/",
            {"dataset": json.dumps(dataset)},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(
            payload["props"]["form"]["errors"],
            {
                "dataset.rows.1.cells.1": ["Income must be numeric on row 2."],
                "dataset.rows": [
                    "Provide at least 4 training row(s) with a filled target value.",
                    "Subscribed must contain at least 2 classes in the training rows.",
                ],
            },
        )
        self.assertIsNone(payload["props"]["result"])
