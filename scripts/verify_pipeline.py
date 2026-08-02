"""
Sanity checks for the built pipeline outputs (fashion_intel.db + chroma_db/).
Run after matching.py -> build_storage.py -> build_vector_store.py.

Usage: python3 scripts/verify_pipeline.py
"""
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chromadb

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
DB_PATH = OUTPUT_DIR / "fashion_intel.db"
CHROMA_DIR = OUTPUT_DIR / "chroma_db"

failures = []


def check(label, condition, detail=""):
    status = "OK  " if condition else "FAIL"
    print(f"[{status}] {label}" + (f" - {detail}" if detail and not condition else ""))
    if not condition:
        failures.append(label)


def verify_sqlite():
    print("\n== SQLite: fashion_intel.db ==")
    if not DB_PATH.exists():
        check("db file exists", False, str(DB_PATH))
        return

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    tables = {r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    check("tables present", {"trend_signals", "competitor_pdp", "self_catalog", "ig_posts"} <= tables, tables)

    total = cur.execute("SELECT COUNT(*) FROM trend_signals").fetchone()[0]
    distinct_pk = cur.execute("SELECT COUNT(DISTINCT source || ':' || source_id) FROM trend_signals").fetchone()[0]
    check("trend_signals has rows", total > 0, total)
    check("PRIMARY KEY (source, source_id) is unique", total == distinct_pk, f"{total} rows, {distinct_pk} distinct")

    by_source = dict(cur.execute("SELECT source, COUNT(*) FROM trend_signals GROUP BY source"))
    check("all 3 sources present", set(by_source) == {"competitor", "self", "instagram"}, by_source)

    bad_momentum = cur.execute(
        "SELECT COUNT(*) FROM trend_signals WHERE momentum NOT IN ('rising','flat','declining')"
    ).fetchone()[0]
    check("momentum values are valid", bad_momentum == 0, f"{bad_momentum} invalid rows")

    null_required = cur.execute(
        "SELECT COUNT(*) FROM trend_signals WHERE category IS NULL OR style_tag IS NULL OR momentum IS NULL"
    ).fetchone()[0]
    check("no NULLs in required columns", null_required == 0, f"{null_required} rows")

    # oos fields should be populated for competitor/self, NULL for instagram
    oos_mismatch = cur.execute(
        "SELECT COUNT(*) FROM trend_signals WHERE (source='instagram' AND oos_weeks_out_of_4 IS NOT NULL) "
        "OR (source!='instagram' AND oos_weeks_out_of_4 IS NULL)"
    ).fetchone()[0]
    check("oos_weeks_out_of_4 null only for instagram", oos_mismatch == 0, f"{oos_mismatch} rows")

    for table, id_field in [("competitor_pdp", "competitor_sku"), ("self_catalog", "sku"), ("ig_posts", "post_id")]:
        n = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        check(f"{table} has rows", n > 0, n)
        bad_json = 0
        for (blob,) in cur.execute(f"SELECT data FROM {table}"):
            try:
                json.loads(blob)
            except json.JSONDecodeError:
                bad_json += 1
        check(f"{table}.data is valid JSON for every row", bad_json == 0, f"{bad_json} bad rows")

        # every trend_signals source_id for competitor/self should resolve to a passthrough row
        source_name = {"competitor_pdp": "competitor", "self_catalog": "self"}.get(table)
        if source_name:
            missing = cur.execute(
                f"SELECT COUNT(*) FROM trend_signals t "
                f"WHERE t.source = ? AND NOT EXISTS (SELECT 1 FROM {table} p WHERE p.id = t.source_id)",
                (source_name,),
            ).fetchone()[0]
            check(f"every {source_name} trend_signals row resolves to {table}", missing == 0, f"{missing} orphaned")

    conn.close()


def verify_chroma():
    print("\n== Chroma: chroma_db/ ==")
    if not CHROMA_DIR.exists():
        check("chroma_db dir exists", False, str(CHROMA_DIR))
        return

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        collection = client.get_collection(name="fashion_intel")
    except Exception as e:
        check("collection 'fashion_intel' exists", False, str(e))
        return

    count = collection.count()
    check("collection has vectors", count > 0, count)

    got = collection.get(include=["embeddings", "metadatas"])
    dims = {len(e) for e in got["embeddings"]}
    check("all embeddings share one dimension", len(dims) == 1, dims)

    by_source = {}
    for m in got["metadatas"]:
        by_source[m["source"]] = by_source.get(m["source"], 0) + 1
    check("all 3 sources present in metadata", set(by_source) == {"competitor", "self", "instagram"}, by_source)

    missing_meta = sum(1 for m in got["metadatas"] if not m.get("category") or not m.get("style_tag"))
    check("every vector has category + style_tag metadata", missing_meta == 0, f"{missing_meta} incomplete")

    # retrieval sanity: a style-specific query's #1 hit should carry that style_tag
    probes = {
        "oversized hoodie streetwear": "oversized-hoodie",
        "graphic print tee": "graphic-print",
        "cropped zip up jacket": "cropped-zip-up",
        "boxy t-shirt": "boxy-tee",
    }
    from src.pipeline.build_vector_store import semantic_search
    for query, expected_tag in probes.items():
        res = semantic_search(query, top_k=1, chroma_dir=CHROMA_DIR)
        top_tag = res["metadatas"][0][0]["style_tag"] if res["metadatas"][0] else None
        check(f"top hit for '{query}' has style_tag={expected_tag}", top_tag == expected_tag, top_tag)


if __name__ == "__main__":
    verify_sqlite()
    verify_chroma()

    print(f"\n{len(failures)} failure(s)" if failures else "\nAll checks passed")
    sys.exit(1 if failures else 0)
