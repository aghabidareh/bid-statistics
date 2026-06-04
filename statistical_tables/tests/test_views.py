from django.test import SimpleTestCase


class StatisticalTableViewTests(SimpleTestCase):
    def inertia_get(self, path: str):
        return self.client.get(path, HTTP_X_INERTIA="true")

    def test_index_lists_two_tables(self):
        response = self.inertia_get("/statistical-tables/")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["component"], "StatisticalTables/Index")
        self.assertEqual([table["slug"] for table in payload["props"]["tables"]], ["z-table", "t-table"])

    def test_z_table_page_returns_table_payload(self):
        response = self.inertia_get("/statistical-tables/z-table/")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["component"], "StatisticalTables/Show")
        self.assertEqual(payload["props"]["metadata"]["name"], "Z Table")
        self.assertEqual(payload["props"]["table"]["kind"], "z")
        self.assertEqual(payload["props"]["table"]["probability"]["defaultCell"]["z"], "1.96")

    def test_t_table_page_returns_table_payload(self):
        response = self.inertia_get("/statistical-tables/t-table/")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["component"], "StatisticalTables/Show")
        self.assertEqual(payload["props"]["metadata"]["name"], "T Table")
        self.assertEqual(payload["props"]["table"]["kind"], "t")
        self.assertEqual(payload["props"]["table"]["criticalValues"]["defaultCell"]["df"], "10")

    def test_unknown_table_returns_not_found(self):
        response = self.inertia_get("/statistical-tables/not-real/")

        self.assertEqual(response.status_code, 404)
