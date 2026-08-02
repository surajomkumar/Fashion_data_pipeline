"""
SQL node - schema-RAG SQL generation, replacing the earlier idea of fixed
parameterized query functions (query_trends/compare_across_sources/
find_whitespace) entirely, per the decision to make this general rather
than a closed set of pre-written shapes.

Pipeline: schema_link() -> generate_sql() -> validate_sql() -> execute_sql().
schema_link() narrows what the LLM sees to the tables/columns/joins
relevant to this question (knowledge_enrichment/schema_link.py); that
retrieved slice is also what validate_sql() checks the generated SQL
against, so retrieval isn't just a prompt-shaping nicety, it's the basis
the safety check works from.

Safety, since this is genuinely "model writes SQL, system executes it"
now, not fixed code:
  - single statement only, must start with SELECT or WITH
  - blacklist of write/DDL/pragma keywords, checked as whole words
  - every FROM/JOIN table must be one of the 4 known tables (not just the
    retrieved slice - a legitimate query can reference a table schema_link
    missed if retrieval under-matched, so validation is generous about
    tables but strict about statement shape/keywords)
  - executed on a READ-ONLY sqlite connection (uri mode=ro) as defense in
    depth beneath the keyword check, and a LIMIT is appended if missing
  - column existence is NOT statically validated (would need a real SQL
    parser) - an invalid column surfaces naturally as a sqlite
    OperationalError at execution time, which is safe to just report
    since the connection is read-only and non-destructive either way

One real failure mode found via the LLM-as-judge eval (question: "Why is
engagement rising for graphic-print on Instagram?"): the model correctly
wrote json_extract(p.data, '$.field') for five ig_posts fields in a row,
then slipped and wrote `p.thumbnail_url` directly for the sixth -
probably primed by trend_signals.thumbnail_url (a real, directly
selectable column) sitting right next to it in the same schema context.
The schema doc was correct; this was a generation slip, not a retrieval
gap, so MAX_SQL_RETRIES adds one bounded self-correction pass: on a
sqlite3.Error, the exact failed SQL and DB error message are fed back to
generate_sql() and the model gets one chance to fix its own query. Still
goes through the same validate_sql()/read-only execute_sql() path -
this doesn't loosen any safety check, it just gives the model the same
kind of feedback a human would get from running the query themselves.
"""
import json
import logging
import re
import sqlite3
from pathlib import Path

from src.agent.knowledge_enrichment.schema_link import render_schema_context, schema_link
from src.agent.nodes.state import AgentState
from src.agent.observability import traced_bedrock_converse, traced_span
from src.pipeline.build_storage import DB_PATH
from src.utils.aws_cred import call_cred

logger = logging.getLogger(__name__)

MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
KNOWN_TABLES = {"trend_signals", "competitor_pdp", "self_catalog", "ig_posts"}
FORBIDDEN_KEYWORDS = (
    "insert", "update", "delete", "drop", "alter", "attach", "detach", "pragma",
    "create", "replace", "vacuum", "reindex", "trigger", "transaction",
    "commit", "rollback", "grant", "revoke", "exec", "execute",
)
DEFAULT_LIMIT = 200
MAX_SQL_RETRIES = 1

SYSTEM_PROMPT = """You are a SQL generator for a read-only SQLite analytics database.
Using ONLY the schema information given below, write a single valid SQLite SELECT statement that answers the question.

Rules:
- Only reference tables and columns listed in the schema below - never invent one.
- Only write ONE statement - a single SELECT (or WITH ... SELECT). No INSERT/UPDATE/DELETE/DROP/PRAGMA/ATTACH, no multiple statements separated by ";".
- Pass-through tables (competitor_pdp, self_catalog, ig_posts) store most fields inside a JSON column called `data` - use json_extract(data, '$.fieldname') to read a specific field, never SELECT data directly unless the full record was asked for.
- Instagram rows in trend_signals are (category, style_tag) clusters - trend_signals.source_id for instagram is NOT a post_id and cannot be joined to ig_posts.id. Filter ig_posts by category/style_tag instead if you need individual posts.
- Some columns only make sense paired with a sibling column that names what they mean - e.g. secondary_metric_value is meaningless without secondary_metric_name (it's a rating average, units-available, or follower count depending on the row's source). If you SELECT a *_value column that has a corresponding *_name column, always SELECT both together.
- Include a LIMIT clause for anything that isn't already a single aggregate value.
- Output ONLY the raw SQL statement - no markdown code fences, no explanation, no trailing semicolon commentary.
"""


class SQLGenerationError(Exception):
    pass


def generate_sql(question: str, entities: dict, schema_context: str, repair: dict = None) -> str:
    prompt = (
        f"{SYSTEM_PROMPT}\n\nSchema:\n{schema_context}\n\n"
        f"Already-resolved filter values (use these literal values in WHERE clauses where relevant, "
        f"don't re-derive them): {json.dumps(entities)}\n\n"
        f"Question: {question}\n\nSQL:"
    )
    if repair:
        prompt += (
            f"\n\nYour previous attempt failed when executed against the real database:\n"
            f"SQL:\n{repair['sql']}\n\nDatabase error: {repair['error']}\n\n"
            f"Fix the SQL so it executes successfully. Output ONLY the corrected SQL statement."
        )

    client = call_cred()
    response = traced_bedrock_converse(
        client, MODEL_ID, [{"role": "user", "content": [{"text": prompt}]}],
        name="sql_generation", inference_config={"temperature": 0},
    )
    raw = response["output"]["message"]["content"][0]["text"]
    return _strip_code_fences(raw).strip()


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(sql)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text


def validate_sql(sql: str, known_tables: set = KNOWN_TABLES) -> str:
    """Returns the validated (and LIMIT-padded if needed) SQL, or raises SQLGenerationError."""
    sql = sql.strip().rstrip(";").strip()
    if not sql:
        raise SQLGenerationError("Generated SQL is empty.")

    statements = [s.strip() for s in sql.split(";") if s.strip()]
    if len(statements) != 1:
        raise SQLGenerationError(f"Expected exactly one statement, got {len(statements)}.")
    sql = statements[0]

    first_word = sql.strip().split(None, 1)[0].lower()
    if first_word not in ("select", "with"):
        raise SQLGenerationError(f"Statement must start with SELECT or WITH, got {first_word!r}.")

    lowered = sql.lower()
    for kw in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{kw}\b", lowered):
            raise SQLGenerationError(f"Forbidden keyword {kw!r} found in generated SQL.")

    referenced = set(re.findall(r"\b(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*)", sql, flags=re.IGNORECASE))
    unknown = {t for t in referenced if t.lower() not in known_tables}
    if unknown:
        raise SQLGenerationError(f"Unknown table(s) referenced: {unknown}. Known tables: {known_tables}.")

    if not re.search(r"\blimit\s+\d+", lowered):
        sql = f"{sql}\nLIMIT {DEFAULT_LIMIT}"

    return sql


def execute_sql(sql: str, db_path: Path = DB_PATH) -> list:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(sql).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def sql_node(state: AgentState) -> dict:
    question = state["question"]
    entities = {k: v for k, v in state.get("entities", {}).items() if v is not None}

    with traced_span("schema_retrieval", as_type="retriever", input=question) as span:
        linked = schema_link(question)
        span.update(output=linked)
    schema_context = render_schema_context(linked)

    raw_sql = None
    repair = None
    result = None
    for attempt in range(MAX_SQL_RETRIES + 1):
        try:
            raw_sql = generate_sql(question, entities, schema_context, repair=repair)
            validated_sql = validate_sql(raw_sql)
            with traced_span("sql_execution", input=validated_sql) as span:
                rows = execute_sql(validated_sql)
                span.update(output={"row_count": len(rows), "rows": rows[:30]})
            result = {"sql": validated_sql, "rows": rows, "row_count": len(rows), "error": None}
            break
        except (SQLGenerationError, sqlite3.Error) as e:
            logger.warning(f"sql_node: attempt {attempt + 1} failed for question={question!r}: {e}")
            result = {"sql": raw_sql, "rows": [], "row_count": 0, "error": str(e)}
            repair = {"sql": raw_sql, "error": str(e)}

    return {"sql_result": result}
