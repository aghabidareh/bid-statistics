import importlib
import os
import sys
from unittest.mock import patch

from django.test import RequestFactory, SimpleTestCase
from django.urls import resolve, reverse

from bid_statistics import views


class ProjectViewsAndUrlsTests(SimpleTestCase):
    def setUp(self):
        self.factory = RequestFactory()

    def test_home_get_renders_sections(self):
        request = self.factory.get("/")

        with (
            patch("bid_statistics.views.build_section", side_effect=[{"slug": "test-statistics"}, {"slug": "regression"}]) as build_section_mock,
            patch("bid_statistics.views.render") as render_mock,
        ):
            views.home(request)

        build_section_mock.assert_any_call("test-statistics")
        build_section_mock.assert_any_call("regression")
        _, component, props = render_mock.call_args.args
        self.assertEqual(component, "Home")
        self.assertEqual(props["sections"], [{"slug": "test-statistics"}, {"slug": "regression"}])

    def test_home_post_not_allowed_by_require_get(self):
        response = self.client.post(reverse("home"))

        self.assertEqual(response.status_code, 405)

    def test_root_url_resolves_to_home(self):
        match = resolve("/")

        self.assertEqual(match.func, views.home)


class AsgiAndWsgiModuleTests(SimpleTestCase):
    def _reload_entrypoint(self, module_name: str):
        sys.modules.pop(module_name, None)
        return importlib.import_module(module_name)

    def test_asgi_module_sets_settings_and_application(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("django.core.asgi.get_asgi_application", return_value="asgi-app") as app_factory:
                module = self._reload_entrypoint("bid_statistics.asgi")

        self.assertEqual(os.environ["DJANGO_SETTINGS_MODULE"], "bid_statistics.settings")
        app_factory.assert_called_once_with()
        self.assertEqual(module.application, "asgi-app")

    def test_wsgi_module_sets_settings_and_application(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("django.core.wsgi.get_wsgi_application", return_value="wsgi-app") as app_factory:
                module = self._reload_entrypoint("bid_statistics.wsgi")

        self.assertEqual(os.environ["DJANGO_SETTINGS_MODULE"], "bid_statistics.settings")
        app_factory.assert_called_once_with()
        self.assertEqual(module.application, "wsgi-app")