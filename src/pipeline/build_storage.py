"""
Step 2 + 3 - Metric normalization and the structured DB (SQLite).

Reads the *_matched.json files written by matching.py, computes
week-over-week momentum per record (competitor/self) or per tag-cluster
(instagram), and writes:

  - trend_signals            one row per competitor SKU, self SKU, or
                              (category, style_tag) IG cluster
  - competitor_pdp / self_catalog / ig_posts   raw pass-through tables,
    full matched record as a JSON blob keyed by its natural ID - this is
    what backs evidence lookups later; trend_signals intentionally does not
    carry every display field (bullets, captions, full price/rating detail).

Full rebuild every run: this drops and recreates all 4 tables each time.
"""
import json
import logging
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output"
DB_PATH = OUTPUT_DIR / "fashion_intel.db"

WEEK_ORDER = ["week_now-3", "week_now-2", "week_now-1", "week_now"]


def week_over_week_trend(series):
    start, end = series[0], series[-1]
    pct_change = 0.0 if start == 0 and end == 0 else (1.0 if start == 0 else (end - start) / start)
    momentum = "rising" if pct_change > 0.10 else "declining" if pct_change < -0.10 else "flat"
    return round(pct_change, 3), momentum


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def build_competitor_rows(competitor_matched: list) -> list:
    rows = []
    for r in competitor_matched:
        series = [r[f"rating_velocity_week_now-3"], r["rating_velocity_week_now-2"],
                  r["rating_velocity_week_now-1"], r["rating_velocity_week_now"]]
        pct_change, momentum = week_over_week_trend(series)
        oos_weeks = sum(bool(r[f"oos_flag_week_now-{n}"]) for n in (3, 2, 1)) + int(bool(r["oos_flag_week_now"]))
        rows.append({
            "source": "competitor",
            "source_id": r["competitor_sku"],
            "category": r["category"],
            "style_tag": r["style_tag"],
            "brand_or_handle": r["brand"],
            "metric_name": "rating_velocity",
            "metric_week_now_3": series[0],
            "metric_week_now_2": series[1],
            "metric_week_now_1": series[2],
            "metric_week_now": series[3],
            "pct_change": pct_change,
            "momentum": momentum,
            "oos_weeks_out_of_4": oos_weeks,
            "oos_status_current": int(bool(r["oos_status_current"])),
            "price": r["price"],
            "secondary_metric_name": "rating_avg",
            "secondary_metric_value": r["rating_avg"],
            "record_count": 1,
            "thumbnail_url": r.get("thumbnail_url"),
        })
    return rows


def build_self_rows(self_matched: list) -> list:
    rows = []
    for r in self_matched:
        series = [r["sell_through_rate_week_now-3"], r["sell_through_rate_week_now-2"],
                  r["sell_through_rate_week_now-1"], r["sell_through_rate_week_now"]]
        pct_change, momentum = week_over_week_trend(series)
        oos_weeks = sum(bool(r[f"oos_flag_week_now-{n}"]) for n in (3, 2, 1)) + int(bool(r["oos_flag_week_now"]))
        rows.append({
            "source": "self",
            "source_id": r["sku"],
            "category": r["category"],
            "style_tag": r["style_tag"],
            "brand_or_handle": "self",
            "metric_name": "sell_through_rate",
            "metric_week_now_3": series[0],
            "metric_week_now_2": series[1],
            "metric_week_now_1": series[2],
            "metric_week_now": series[3],
            "pct_change": pct_change,
            "momentum": momentum,
            "oos_weeks_out_of_4": oos_weeks,
            "oos_status_current": int(bool(r["oos_status_current"])),
            "price": r["price"],
            "secondary_metric_name": "units_available_current",
            "secondary_metric_value": r["units_available_current"],
            "record_count": 1,
            "thumbnail_url": r.get("thumbnail_url"),
        })
    return rows


def build_instagram_rows(ig_matched: list) -> list:
    """
    IG rows represent a (category, style_tag) tag-week cluster, never a
    single post's raw numbers: aggregate likes+comments_count per
    (category, style_tag, week_label) BEFORE trending.
    """
    engagement = defaultdict(lambda: defaultdict(int))
    follower_sum = defaultdict(int)
    record_count = defaultdict(int)
    sample_thumbnail = {}

    for post in ig_matched:
        if post.get("tag_source") == "unmatched":
            continue
        key = (post["category"], post["style_tag"])
        engagement[key][post["week_label"]] += (post.get("likes") or 0) + (post.get("comments_count") or 0)
        follower_sum[key] += post.get("follower_count") or 0
        record_count[key] += 1
        sample_thumbnail.setdefault(key, post.get("thumbnail_url"))

    rows = []
    for key, weeks in engagement.items():
        category, style_tag = key
        series = [weeks.get(w, 0) for w in WEEK_ORDER]
        pct_change, momentum = week_over_week_trend(series)
        n = record_count[key]
        rows.append({
            "source": "instagram",
            "source_id": f"{category}::{style_tag}",
            "category": category,
            "style_tag": style_tag,
            "brand_or_handle": None,
            "metric_name": "ig_engagement",
            "metric_week_now_3": series[0],
            "metric_week_now_2": series[1],
            "metric_week_now_1": series[2],
            "metric_week_now": series[3],
            "pct_change": pct_change,
            "momentum": momentum,
            "oos_weeks_out_of_4": None,
            "oos_status_current": None,
            "price": None,
            "secondary_metric_name": "avg_follower_count",
            "secondary_metric_value": round(follower_sum[key] / n, 1) if n else None,
            "record_count": n,
            "thumbnail_url": sample_thumbnail.get(key),
        })
    return rows


CREATE_TREND_SIGNALS = """
CREATE TABLE trend_signals (
    source TEXT NOT NULL,
    source_id TEXT NOT NULL,
    category TEXT NOT NULL,
    style_tag TEXT NOT NULL,
    brand_or_handle TEXT,
    metric_name TEXT NOT NULL,
    metric_week_now_3 REAL,
    metric_week_now_2 REAL,
    metric_week_now_1 REAL,
    metric_week_now REAL,
    pct_change REAL,
    momentum TEXT NOT NULL,
    oos_weeks_out_of_4 INTEGER,
    oos_status_current INTEGER,
    price REAL,
    secondary_metric_name TEXT,
    secondary_metric_value REAL,
    record_count INTEGER NOT NULL,
    thumbnail_url TEXT,
    computed_at TEXT NOT NULL,
    PRIMARY KEY (source, source_id)
)
"""

PASSTHROUGH_TABLES = {
    "competitor_pdp": "competitor_sku",
    "self_catalog": "sku",
    "ig_posts": "post_id",
}

CREATE_PASSTHROUGH = """
CREATE TABLE {table} (
    id TEXT PRIMARY KEY,
    category TEXT,
    style_tag TEXT,
    data TEXT NOT NULL
)
"""


def write_sqlite(trend_rows: list, competitor_matched: list, self_matched: list, ig_matched: list,
                  db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()

        cur.execute("DROP TABLE IF EXISTS trend_signals")
        cur.execute(CREATE_TREND_SIGNALS)

        computed_at = datetime.now(timezone.utc).isoformat()
        cur.executemany(
            f"INSERT INTO trend_signals ({', '.join(k for k in trend_rows[0].keys())}, computed_at) "
            f"VALUES ({', '.join(['?'] * (len(trend_rows[0]) + 1))})",
            [tuple(row.values()) + (computed_at,) for row in trend_rows],
        )

        passthrough_data = {
            "competitor_pdp": competitor_matched,
            "self_catalog": self_matched,
            "ig_posts": ig_matched,
        }
        for table, id_field in PASSTHROUGH_TABLES.items():
            cur.execute(f"DROP TABLE IF EXISTS {table}")
            cur.execute(CREATE_PASSTHROUGH.format(table=table))
            records = passthrough_data[table]
            cur.executemany(
                f"INSERT INTO {table} (id, category, style_tag, data) VALUES (?, ?, ?, ?)",
                [(r[id_field], r.get("category"), r.get("style_tag"), json.dumps(r)) for r in records],
            )

        conn.commit()
    finally:
        conn.close()


def run_build_storage(output_dir: Path = OUTPUT_DIR, db_path: Path = DB_PATH) -> dict:
    competitor_matched = load_json(output_dir / "competitor_pdp_matched.json")
    self_matched = load_json(output_dir / "self_catalog_matched.json")
    ig_matched = load_json(output_dir / "ig_posts_matched.json")

    trend_rows = (
        build_competitor_rows(competitor_matched)
        + build_self_rows(self_matched)
        + build_instagram_rows(ig_matched)
    )

    write_sqlite(trend_rows, competitor_matched, self_matched, ig_matched, db_path=db_path)

    momentum_dist = defaultdict(int)
    source_dist = defaultdict(int)
    for row in trend_rows:
        momentum_dist[row["momentum"]] += 1
        source_dist[row["source"]] += 1

    return {
        "trend_signals_rows": len(trend_rows),
        "by_source": dict(source_dist),
        "momentum_distribution": dict(momentum_dist),
        "db_path": str(db_path),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    summary = run_build_storage()
    print(json.dumps(summary, indent=2))
