from django.test import SimpleTestCase

from statistical_tables.tables import build_table_payload, get_table_metadata, list_tables


class StatisticalTableDataTests(SimpleTestCase):
    def test_catalog_contains_z_and_t_tables(self):
        tables = list_tables()

        self.assertEqual([table.slug for table in tables], ["z-table", "t-table"])
        self.assertEqual(tables[0].href, "/statistical-tables/z-table/")
        self.assertEqual(get_table_metadata("t-table").name, "T Table")
        self.assertIsNone(get_table_metadata("missing"))

    def test_z_table_payload_contains_probabilities_and_inverse_rows(self):
        payload = build_table_payload("z-table")

        self.assertEqual(payload["kind"], "z")
        self.assertEqual(payload["probability"]["columns"][:3], ["0.00", "0.01", "0.02"])
        self.assertEqual(payload["probability"]["defaultCell"], {"z": "1.96", "value": "0.9750021"})
        self.assertEqual(payload["probability"]["rows"][0]["z"], "-3.4")
        self.assertEqual(payload["probability"]["rows"][-1]["z"], "3.4")
        self.assertEqual(payload["inverse"]["rows"][5]["alpha"], "0.05")
        self.assertEqual(payload["inverse"]["rows"][5]["zOneMinusAlphaOverTwo"], "1.959964")

    def test_t_table_payload_contains_critical_values(self):
        payload = build_table_payload("t-table")

        self.assertEqual(payload["kind"], "t")
        self.assertEqual(payload["criticalValues"]["oneTailColumns"], ["0.1", "0.05", "0.025", "0.01", "0.005"])
        self.assertEqual(payload["criticalValues"]["twoTailColumns"], ["0.2", "0.1", "0.05", "0.02", "0.01"])
        self.assertEqual(payload["criticalValues"]["defaultCell"], {"df": "10", "alpha": "0.05", "value": "1.812461"})
        self.assertEqual(payload["criticalValues"]["rows"][-1]["df"], "∞")
        self.assertEqual(payload["criticalValues"]["rows"][-1]["cells"][1]["value"], "1.644854")

    def test_unknown_table_payload_returns_none(self):
        self.assertIsNone(build_table_payload("missing"))
