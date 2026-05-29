from django.test import SimpleTestCase

from services.calculators.registry import list_calculators


class RegressionRegistryTests(SimpleTestCase):
    def test_regression_catalog_can_be_listed_by_section(self):
        calculators = list_calculators(section_slug="regression")

        self.assertEqual(
            [calculator.slug for calculator in calculators],
            [
                "simple-linear-regression",
                "multiple-linear-regression",
                "bulk-linear-regression",
                "binary-logistic-regression",
                "multinomial-logistic-regression",
                "propensity-score-matching",
            ],
        )
