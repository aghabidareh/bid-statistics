from django.core.cache import cache
from django.test import SimpleTestCase


class TestStatisticsViewTests(SimpleTestCase):
    def setUp(self):
        super().setUp()
        cache.clear()

    def tearDown(self):
        cache.clear()
        super().tearDown()

    def inertia_get(self, path: str):
        return self.client.get(path, HTTP_X_INERTIA="true")

    def inertia_post(self, path: str, data: dict[str, str]):
        return self.client.post(path, data, HTTP_X_INERTIA="true")

    def test_home_page_returns_both_section_overviews(self):
        response = self.inertia_get("/")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["component"], "Home")
        self.assertEqual(len(payload["props"]["sections"]), 3)
        self.assertEqual(payload["props"]["sections"][0]["slug"], "test-statistics")
        self.assertEqual(payload["props"]["sections"][0]["href"], "/test-statistics/")
        self.assertEqual(payload["props"]["sections"][0]["itemCount"], 26)
        self.assertEqual(payload["props"]["sections"][1]["slug"], "regression")
        self.assertEqual(payload["props"]["sections"][1]["href"], "/regression/")
        self.assertEqual(payload["props"]["sections"][1]["itemCount"], 6)
        self.assertEqual(payload["props"]["sections"][2]["slug"], "statistical-tables")
        self.assertEqual(payload["props"]["sections"][2]["href"], "/statistical-tables/")
        self.assertEqual(payload["props"]["sections"][2]["itemCount"], 2)

    def test_test_statistics_index_returns_exact_26_calculator_catalog(self):
        response = self.inertia_get("/test-statistics/")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["component"], "TestStatistics/Index")
        self.assertEqual(len(payload["props"]["catalog"]), 26)
        self.assertEqual(payload["props"]["catalog"][0]["slug"], "one-sample-z-test")
        self.assertEqual(payload["props"]["catalog"][-1]["slug"], "delong-test-paired-curves")

    def test_calculator_page_returns_metadata_and_defaults(self):
        response = self.inertia_get("/test-statistics/one-sample-z-test/")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["component"], "TestStatistics/Show")
        self.assertEqual(payload["props"]["calculator"]["slug"], "one-sample-z-test")
        self.assertEqual(payload["props"]["form"]["values"]["alpha"], "0.05")
        self.assertEqual(payload["props"]["form"]["values"]["alternative"], "two-sided")
        self.assertIsNone(payload["props"]["result"])

    def test_valid_calculation_post_returns_structured_result_props(self):
        response = self.inertia_post(
            "/test-statistics/kaplan-meier-survival-analysis/calculate/",
            {
                "rows": "5, 1\n8, 0\n12, 1\n15, 1\n20, 0",
                "alpha": "0.05",
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["component"], "TestStatistics/Show")
        self.assertEqual(payload["props"]["calculator"]["slug"], "kaplan-meier-survival-analysis")
        self.assertEqual(payload["props"]["result"]["statisticName"], "Median survival time")
        self.assertIsNone(payload["props"]["result"]["pValue"])
        self.assertEqual(len(payload["props"]["result"]["tables"]), 2)

    def test_invalid_calculation_post_returns_structured_validation_errors(self):
        response = self.inertia_post(
            "/test-statistics/paired-t-test/calculate/",
            {
                "sample_a": "1, 2, 3",
                "sample_b": "1, 2",
                "alternative": "two-sided",
                "alpha": "0.05",
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(
            payload["props"]["form"]["errors"],
            {"sample_b": ["Measurement A values and Measurement B values must have the same length."]},
        )
        self.assertIsNone(payload["props"]["result"])

    def test_unknown_calculator_returns_not_found(self):
        response = self.inertia_get("/test-statistics/not-a-real-calculator/")

        self.assertEqual(response.status_code, 404)
