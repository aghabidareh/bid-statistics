import json
from types import SimpleNamespace
from unittest.mock import patch

from django.http import Http404
from django.test import RequestFactory, SimpleTestCase

from bid_statistics import section_views
from domain.results import ValidationIssue
from services.validators import ValidationIssues


class SectionViewsTests(SimpleTestCase):
    def setUp(self):
        self.factory = RequestFactory()

    def test_get_request_data_reads_json_and_normalizes_values(self):
        request = self.factory.post(
            "/test-statistics/fake/calculate/",
            data=json.dumps({"a": None, "b": True, "c": [1], "d": {"nested": 1}}),
            content_type="application/json",
        )

        data = section_views.get_request_data(request)

        self.assertEqual(data["a"], "")
        self.assertEqual(data["b"], True)
        self.assertEqual(data["c"], [1])
        self.assertEqual(data["d"], {"nested": 1})

    def test_normalize_request_value_casts_unknown_objects_to_string(self):
        self.assertIsInstance(section_views.normalize_request_value(object()), str)

    def test_get_request_data_falls_back_to_post_and_skips_csrf(self):
        request = self.factory.post(
            "/test-statistics/fake/calculate/",
            data={"x": "1", "csrfmiddlewaretoken": "token"},
        )

        data = section_views.get_request_data(request)

        self.assertEqual(data, {"x": "1"})

    def test_get_request_data_parses_dataset_json_from_standard_post(self):
        request = self.factory.post(
            "/regression/fake/calculate/",
            data={
                "dataset": json.dumps({
                    "columns": [{"key": "column_1", "label": "x", "role": "predictor"}],
                    "rows": [{"cells": ["1"]}],
                    "sourceMode": "grid",
                    "filename": "",
                }),
                "csrfmiddlewaretoken": "token",
            },
        )

        data = section_views.get_request_data(request)

        self.assertIsInstance(data["dataset"], dict)
        self.assertEqual(data["dataset"]["rows"][0]["cells"], ["1"])

    def test_build_section_uses_catalog_length(self):
        calculators = [SimpleNamespace(), SimpleNamespace(), SimpleNamespace()]

        with patch("bid_statistics.section_views.list_calculators", return_value=calculators):
            section = section_views.build_section("test-statistics")

        self.assertEqual(section["slug"], "test-statistics")
        self.assertEqual(section["itemCount"], 3)

    def test_build_section_uses_statistical_tables_catalog_length(self):
        tables = [SimpleNamespace(), SimpleNamespace()]

        with patch("bid_statistics.section_views.list_tables", return_value=tables):
            section = section_views.build_section("statistical-tables")

        self.assertEqual(section["slug"], "statistical-tables")
        self.assertEqual(section["itemCount"], 2)

    def test_render_catalog_renders_index_component(self):
        request = self.factory.get("/test-statistics/")
        calculator = SimpleNamespace(to_dict=lambda: {"slug": "calc-1"})

        with (
            patch("bid_statistics.section_views.list_calculators", return_value=[calculator]),
            patch("bid_statistics.section_views.render", return_value=SimpleNamespace(status_code=200)) as render_mock,
        ):
            response = section_views.render_catalog(request, section_slug="test-statistics")

        self.assertEqual(response.status_code, 200)
        _, component, props = render_mock.call_args.args
        self.assertEqual(component, section_views.SECTION_CONFIG["test-statistics"]["index_component"])
        self.assertEqual(props["catalog"], [{"slug": "calc-1"}])

    def test_render_show_renders_show_component_with_default_props(self):
        request = self.factory.get("/test-statistics/fake/")
        metadata = SimpleNamespace(
            slug="fake",
            section_slug="test-statistics",
            default_values={"alpha": "0.05"},
            to_dict=lambda: {"slug": "fake"},
        )

        with (
            patch("bid_statistics.section_views.get_metadata_or_404", return_value=metadata),
            patch("bid_statistics.section_views.get_token", return_value="csrf-token"),
            patch("bid_statistics.section_views.render", return_value=SimpleNamespace(status_code=200)) as render_mock,
        ):
            response = section_views.render_show(request, slug="fake", section_slug="test-statistics")

        self.assertEqual(response.status_code, 200)
        _, component, props = render_mock.call_args.args
        self.assertEqual(component, section_views.SECTION_CONFIG["test-statistics"]["show_component"])
        self.assertEqual(props["form"]["values"], {"alpha": "0.05"})

    def test_get_metadata_or_404_raises_when_calculator_unknown(self):
        with patch("bid_statistics.section_views.get_calculator_metadata", side_effect=section_views.UnknownCalculatorError("nope")):
            with self.assertRaises(Http404):
                section_views.get_metadata_or_404("missing", section_slug="test-statistics")

    def test_get_metadata_or_404_raises_for_wrong_section(self):
        metadata = SimpleNamespace(section_slug="regression")

        with patch("bid_statistics.section_views.get_calculator_metadata", return_value=metadata):
            with self.assertRaises(Http404):
                section_views.get_metadata_or_404("slug", section_slug="test-statistics")

    def test_get_metadata_or_404_returns_metadata_for_matching_section(self):
        metadata = SimpleNamespace(section_slug="test-statistics")

        with patch("bid_statistics.section_views.get_calculator_metadata", return_value=metadata):
            result = section_views.get_metadata_or_404("slug", section_slug="test-statistics")

        self.assertIs(result, metadata)

    def test_render_calculation_handles_validation_errors(self):
        metadata = SimpleNamespace(
            slug="fake-calc",
            section_slug="test-statistics",
            default_values={"alpha": "0.05"},
            to_dict=lambda: {"slug": "fake-calc"},
        )
        request = self.factory.post("/test-statistics/fake-calc/calculate/", data={"alpha": "bad"})

        with (
            patch("bid_statistics.section_views.get_metadata_or_404", return_value=metadata),
            patch(
                "bid_statistics.section_views.calculate_test_statistic",
                side_effect=ValidationIssues([ValidationIssue(field="alpha", message="Alpha is invalid")]),
            ),
            patch("bid_statistics.section_views.get_token", return_value="csrf-token"),
            patch("bid_statistics.section_views.render", return_value=SimpleNamespace(status_code=200)) as render_mock,
        ):
            response = section_views.render_calculation(request, slug="fake-calc", section_slug="test-statistics")

        self.assertEqual(response.status_code, 200)
        _, component, props = render_mock.call_args.args
        self.assertEqual(component, section_views.SECTION_CONFIG["test-statistics"]["show_component"])
        self.assertEqual(props["form"]["errors"], {"alpha": ["Alpha is invalid"]})
        self.assertIsNone(props["result"])

    def test_render_calculation_returns_result_dict_on_success(self):
        metadata = SimpleNamespace(
            slug="fake-calc",
            section_slug="test-statistics",
            default_values={"alpha": "0.05", "x": "1"},
            to_dict=lambda: {"slug": "fake-calc"},
        )
        request = self.factory.post(
            "/test-statistics/fake-calc/calculate/",
            data=json.dumps({"x": "2"}),
            content_type="application/json",
        )
        result = SimpleNamespace(to_dict=lambda: {"value": 123})

        with (
            patch("bid_statistics.section_views.get_metadata_or_404", return_value=metadata),
            patch("bid_statistics.section_views.calculate_test_statistic", return_value=result),
            patch("bid_statistics.section_views.get_token", return_value="csrf-token"),
            patch("bid_statistics.section_views.render", return_value=SimpleNamespace(status_code=200)) as render_mock,
        ):
            response = section_views.render_calculation(request, slug="fake-calc", section_slug="test-statistics")

        self.assertEqual(response.status_code, 200)
        _, _, props = render_mock.call_args.args
        self.assertEqual(props["form"]["values"]["x"], "2")
        self.assertEqual(props["result"], {"value": 123})