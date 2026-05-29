from django.test import SimpleTestCase

from domain.enums import TestFamily
from domain.metadata import CalculatorMetadata
from domain.results import CalculationResult
from services.calculators.base import BaseCalculator
from services.calculators.base import DuplicateCalculatorSlugError, UnknownCalculatorError
from services.calculators.registry import (
    get_calculator,
    get_calculator_metadata,
    list_all_calculators,
)


class BaseAndRegistryCoverageTests(SimpleTestCase):
    def test_registry_returns_singleton_calculator_instance(self):
        first = get_calculator("one-sample-t-test")
        second = get_calculator("one-sample-t-test")

        self.assertIs(first, second)

    def test_get_calculator_metadata_and_list_all_calculators(self):
        metadata = get_calculator_metadata("one-sample-t-test")
        all_metadata = list_all_calculators()

        self.assertEqual(metadata.slug, "one-sample-t-test")
        self.assertGreater(len(all_metadata), 26)

    def test_default_values_are_available_from_calculator(self):
        calculator = get_calculator("one-sample-z-test")

        self.assertEqual(calculator.default_values()["alpha"], "0.05")

    def test_subclass_without_metadata_raises_type_error(self):
        with self.assertRaises(TypeError):

            class MissingMetadataCalculator(BaseCalculator):
                def normalize(self, raw_data):
                    return raw_data

                def calculate_result(self, normalized_input):
                    return CalculationResult(slug="missing-metadata", test_name="Missing metadata")

    def test_subclass_with_register_false_is_not_registered(self):
        class NotRegisteredCalculator(BaseCalculator, register=False):
            metadata = CalculatorMetadata(
                catalog_position=998,
                slug="not-registered-calculator",
                name="Not Registered",
                family=TestFamily.PARAMETRIC,
                description="Not Registered",
                check="Not Registered",
                statistic_formula="N/A",
                assumptions=("N/A",),
                required_sample_data=("N/A",),
            )

            def normalize(self, raw_data):
                return raw_data

            def calculate_result(self, normalized_input):
                return CalculationResult(slug="not-registered-calculator", test_name="Not Registered")

        slugs = {metadata.slug for metadata in list_all_calculators()}
        self.assertNotIn(NotRegisteredCalculator.metadata.slug, slugs)

    def test_base_abstract_methods_raise_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            BaseCalculator.normalize(object(), {})

        with self.assertRaises(NotImplementedError):
            BaseCalculator.calculate_result(object(), {})

    def test_for_slug_raises_unknown_calculator_error(self):
        with self.assertRaises(UnknownCalculatorError):
            BaseCalculator.for_slug("does-not-exist")

    def test_duplicate_slug_registration_raises(self):
        with self.assertRaises(DuplicateCalculatorSlugError):

            class DuplicateSlugCalculator(BaseCalculator):
                metadata = CalculatorMetadata(
                    catalog_position=999,
                    slug="one-sample-t-test",
                    name="Duplicate",
                    family=TestFamily.PARAMETRIC,
                    description="Duplicate",
                    check="Duplicate",
                    statistic_formula="N/A",
                    assumptions=("N/A",),
                    required_sample_data=("N/A",),
                )

                def normalize(self, raw_data):
                    return raw_data

                def calculate_result(self, normalized_input):
                    return CalculationResult(slug="duplicate", test_name="Duplicate")

    def test_all_calculators_can_filter_by_section_slug(self):
        calculators = BaseCalculator.all_calculators(section_slug="nonexistent-section")
        self.assertEqual(calculators, ())