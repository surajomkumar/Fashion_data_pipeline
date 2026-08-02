"""
Unit tests for output/fashion_intel.db (trend_signals + the 3 pass-through
tables written by src/pipeline/build_storage.py).

Run the pipeline first so the DB exists:
    python src/pipeline/matching.py
    python src/pipeline/build_storage.py

Then run these tests:
    python -m unittest test.test_sql_db -v
    (or just: python test/test_sql_db.py)
"""
import json
import sqlite3
import unittest
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "output" / "fashion_intel.db"

# One SELECT per table, run in this order by preview_tables() when the file
# is executed directly. Pass a different list of 4 (or more) SQL strings to
# preview_tables(queries=[...]) to inspect something else.
DEFAULT_PREVIEW_QUERIES = [
    "SELECT source, source_id, category, style_tag, momentum, pct_change, "
    "record_count, oos_weeks_out_of_4 FROM trend_signals ORDER BY source, style_tag LIMIT 10",

    "SELECT id, category, style_tag, json_extract(data, '$.brand') AS brand, "
    "json_extract(data, '$.title') AS title, json_extract(data, '$.price') AS price "
    "FROM competitor_pdp LIMIT 10",

    "SELECT id, category, style_tag, json_extract(data, '$.style_name') AS style_name, "
    "json_extract(data, '$.price') AS price FROM self_catalog LIMIT 10",

    "SELECT id, category, style_tag, json_extract(data, '$.caption_text') AS caption, "
    "json_extract(data, '$.likes') AS likes FROM ig_posts LIMIT 10",
]

VALID_MOMENTUM = {"rising", "flat", "declining"}
EXPECTED_TABLES = {"trend_signals", "competitor_pdp", "self_catalog", "ig_posts"}
PASSTHROUGH_ID_FIELD = {
    "competitor_pdp": "competitor_sku",
    "self_catalog": "sku",
    "ig_posts": "post_id",
}


class TestFashionIntelDB(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not DB_PATH.exists():
            raise unittest.SkipTest(f"{DB_PATH} does not exist - run the pipeline first")
        cls.conn = sqlite3.connect(DB_PATH)
        cls.conn.row_factory = sqlite3.Row

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()

    def test_all_tables_exist(self):
        tables = {r[0] for r in self.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertTrue(EXPECTED_TABLES <= tables, f"missing tables: {EXPECTED_TABLES - tables}")

    def test_trend_signals_row_count(self):
        n = self.conn.execute("SELECT COUNT(*) FROM trend_signals").fetchone()[0]
        self.assertGreater(n, 0, "trend_signals is empty")

    def test_trend_signals_source_breakdown(self):
        rows = self.conn.execute("SELECT source, COUNT(*) FROM trend_signals GROUP BY source").fetchall()
        by_source = {r[0]: r[1] for r in rows}
        self.assertEqual(set(by_source), {"competitor", "self", "instagram"})
        for source, count in by_source.items():
            self.assertGreater(count, 0, f"no rows for source={source}")

    def test_primary_key_is_unique(self):
        total = self.conn.execute("SELECT COUNT(*) FROM trend_signals").fetchone()[0]
        distinct = self.conn.execute(
            "SELECT COUNT(DISTINCT source || ':' || source_id) FROM trend_signals"
        ).fetchone()[0]
        self.assertEqual(total, distinct, "duplicate (source, source_id) pairs found")

    def test_momentum_values_are_valid(self):
        bad = self.conn.execute(
            "SELECT source, source_id, momentum FROM trend_signals WHERE momentum NOT IN (?, ?, ?)",
            tuple(VALID_MOMENTUM),
        ).fetchall()
        self.assertEqual(bad, [], f"invalid momentum values: {[dict(r) for r in bad]}")

    def test_pct_change_matches_momentum_thresholds(self):
        for row in self.conn.execute("SELECT source, source_id, pct_change, momentum FROM trend_signals"):
            pct, momentum = row["pct_change"], row["momentum"]
            if pct > 0.10:
                expected = "rising"
            elif pct < -0.10:
                expected = "declining"
            else:
                expected = "flat"
            self.assertEqual(
                momentum, expected,
                f"{row['source']}/{row['source_id']}: pct_change={pct} implies {expected}, got {momentum}",
            )

    def test_no_nulls_in_required_columns(self):
        n = self.conn.execute(
            "SELECT COUNT(*) FROM trend_signals WHERE category IS NULL OR style_tag IS NULL OR momentum IS NULL"
        ).fetchone()[0]
        self.assertEqual(n, 0)

    def test_oos_fields_null_only_for_instagram(self):
        n = self.conn.execute(
            "SELECT COUNT(*) FROM trend_signals WHERE "
            "(source = 'instagram' AND oos_weeks_out_of_4 IS NOT NULL) OR "
            "(source != 'instagram' AND oos_weeks_out_of_4 IS NULL)"
        ).fetchone()[0]
        self.assertEqual(n, 0, "oos_weeks_out_of_4 should be set for competitor/self, NULL for instagram")

    def test_oos_weeks_within_range(self):
        rows = self.conn.execute(
            "SELECT source, source_id, oos_weeks_out_of_4 FROM trend_signals WHERE oos_weeks_out_of_4 IS NOT NULL"
        ).fetchall()
        for row in rows:
            self.assertTrue(0 <= row["oos_weeks_out_of_4"] <= 4, dict(row))

    def test_passthrough_tables_have_rows(self):
        for table in PASSTHROUGH_ID_FIELD:
            n = self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            self.assertGreater(n, 0, f"{table} is empty")

    def test_passthrough_data_is_valid_json(self):
        for table in PASSTHROUGH_ID_FIELD:
            for (blob,) in self.conn.execute(f"SELECT data FROM {table}"):
                try:
                    json.loads(blob)
                except json.JSONDecodeError as e:
                    self.fail(f"{table}: invalid JSON blob ({e})")

    def test_passthrough_ids_are_unique(self):
        for table in PASSTHROUGH_ID_FIELD:
            total = self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            distinct = self.conn.execute(f"SELECT COUNT(DISTINCT id) FROM {table}").fetchone()[0]
            self.assertEqual(total, distinct, f"{table}.id is not unique")

    def test_competitor_and_self_trend_rows_resolve_to_passthrough(self):
        table_by_source = {"competitor": "competitor_pdp", "self": "self_catalog"}
        for source, table in table_by_source.items():
            orphans = self.conn.execute(
                f"SELECT t.source_id FROM trend_signals t "
                f"WHERE t.source = ? AND NOT EXISTS (SELECT 1 FROM {table} p WHERE p.id = t.source_id)",
                (source,),
            ).fetchall()
            self.assertEqual(orphans, [], f"trend_signals rows with no matching {table} record: {orphans}")

    def test_instagram_trend_rows_are_tag_clusters_not_posts(self):
        """IG rows in trend_signals represent (category, style_tag) clusters, so
        there should be far fewer than the number of raw ig_posts rows."""
        ig_trend_rows = self.conn.execute(
            "SELECT COUNT(*) FROM trend_signals WHERE source = 'instagram'"
        ).fetchone()[0]
        ig_raw_rows = self.conn.execute("SELECT COUNT(*) FROM ig_posts").fetchone()[0]
        self.assertLess(ig_trend_rows, ig_raw_rows)

    def test_secondary_metric_matches_source(self):
        expected = {
            "competitor": "rating_avg",
            "self": "units_available_current",
            "instagram": "avg_follower_count",
        }
        for source, metric_name in expected.items():
            bad = self.conn.execute(
                "SELECT COUNT(*) FROM trend_signals WHERE source = ? AND secondary_metric_name != ?",
                (source, metric_name),
            ).fetchone()[0]
            self.assertEqual(bad, 0, f"{source} rows with unexpected secondary_metric_name")


def preview_tables(queries: list = None, db_path: Path = DB_PATH) -> None:
    """
    Executes the given SQL statements one by one and prints the results as a
    simple text table. Defaults to one SELECT per table (trend_signals,
    competitor_pdp, self_catalog, ig_posts) so you can see real rows, not
    just pass/fail.
    """
    queries = queries if queries is not None else DEFAULT_PREVIEW_QUERIES

    if not db_path.exists():
        print(f"{db_path} does not exist - run the pipeline first.")
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        for i, sql in enumerate(queries, start=1):
            print(f"\n=== Query {i} ===")
            print(sql.strip())
            print()

            try:
                cur = conn.execute(sql)
            except sqlite3.Error as e:
                print(f"ERROR: {e}")
                continue

            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()

            if not rows:
                print("(no rows)")
                continue

            widths = [
                max(len(str(c)), max((len(str(r[c])) for r in rows), default=0))
                for c in cols
            ]
            header = " | ".join(c.ljust(w) for c, w in zip(cols, widths))
            print(header)
            print("-+-".join("-" * w for w in widths))
            for r in rows:
                print(" | ".join(str(r[c]).ljust(w) for c, w in zip(cols, widths)))
            print(f"({len(rows)} row(s))")
    finally:
        conn.close()


if __name__ == "__main__":
    print("#" * 70)
    print("# Running test cases")
    print("#" * 70)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(unittest.defaultTestLoader.loadTestsFromTestCase(TestFashionIntelDB))

    print("\n" + "#" * 70)
    print("# Previewing table data")
    print("#" * 70)
    preview_tables()
