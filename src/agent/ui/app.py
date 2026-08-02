"""
Streamlit UI over run_agent() (lang_graph.py) - a chat box on the left,
a per-turn trace/step breakdown on the right. Run with:

    streamlit run src/agent/ui/app.py

Left side is a standard st.chat_input/st.chat_message thread, one
thread_id per browser session (st.session_state, generated once) so
MemorySaver's last-3-turn conversation memory actually accumulates the
way it does in the CLI (lang_graph.py's __main__ block reusing one
thread across questions).

Each assistant turn renders as three stacked boxes, in this fixed order:
  1. Enriched Query - what enrichment_node actually resolved the question
     to (see enrich.build_enriched_question) - shown first since it's the
     input everything downstream acted on, useful for spotting a bad
     resolution (e.g. the wrong momentum/source getting picked up) before
     even reading the answer.
  2. Data - the raw SQL rows as a real table (optional: only rendered
     when sql_result has rows), so the row-level detail doesn't have to
     be squeezed into the answer text itself. A thumbnail_url column (a
     path relative to data/, e.g. "final_data_v3/images/streetwear_tops__
     oversized-hoodie__self1.jpg" - see build_storage.py) is rendered as
     an actual image gallery instead of a raw path string; the column is
     dropped from the table itself since showing the same thing twice
     (path text + image) is noise, not signal.
  3. Answer - the final answer text (bullet-point narrative for
     explanatory/semantic/strategic, summary+table-text for structured).
Skipped boxes (no enriched_question - blocked/memory-hit turns; no rows -
non-SQL or empty-result turns) are simply omitted, not shown empty.

Right side shows the LATEST turn's pipeline steps (guardrail outcome,
resolved intent/entities, the enriched question text, generated SQL,
vector hits) plus a link out to the full Langfuse trace for that turn -
the in-app view is for "what happened at a glance", Langfuse is for
deep-dives (token usage, latency breakdown, judge scores).
"""
import uuid
from pathlib import Path

import streamlit as st

from src.agent.lang_graph import run_agent

st.set_page_config(page_title="Fashion Intelligence Agent", layout="wide")

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
IMAGE_COLUMNS_PER_ROW = 4

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state.history = []  # list of {"question": str, "result": dict}


def _resolve_image_path(thumbnail_url: str) -> Path:
    path = DATA_DIR / thumbnail_url
    return path if path.is_file() else None


def _row_caption(row: dict) -> str:
    return row.get("style_name") or row.get("title") or row.get("style_tag") or ""


def _render_answer(result: dict) -> None:
    if result.get("enriched_question"):
        with st.container(border=True):
            st.caption("Enriched Query")
            st.write(result["enriched_question"])

    rows = (result.get("sql_result") or {}).get("rows") or []
    if rows:
        with st.container(border=True):
            st.caption("Data")
            has_thumbnails = any(row.get("thumbnail_url") for row in rows)
            table_rows = [{k: v for k, v in row.items() if k != "thumbnail_url"} for row in rows] if has_thumbnails else rows
            st.dataframe(table_rows, width="stretch")

            if has_thumbnails:
                image_rows = [(row, _resolve_image_path(row["thumbnail_url"]))
                              for row in rows if row.get("thumbnail_url")]
                image_rows = [(row, path) for row, path in image_rows if path is not None]
                if image_rows:
                    for start in range(0, len(image_rows), IMAGE_COLUMNS_PER_ROW):
                        chunk = image_rows[start:start + IMAGE_COLUMNS_PER_ROW]
                        cols = st.columns(IMAGE_COLUMNS_PER_ROW)
                        for col, (row, path) in zip(cols, chunk):
                            col.image(str(path), caption=_row_caption(row), width="stretch")

    with st.container(border=True):
        st.caption("Answer")
        st.markdown(result["answer"])


def _render_trace_panel(result: dict) -> None:
    st.subheader("Trace")

    if result.get("memory_hit") is not None:
        st.info("Answered from conversation memory (exact repeat of a recent question) - no pipeline steps ran.")

    st.markdown("**1. Guardrail**")
    if result.get("guardrail_passed"):
        st.success("Passed")
    else:
        st.error(f"Blocked: {result.get('guardrail_message')}")

    if result.get("guardrail_passed"):
        st.markdown("**2. Resolved intent**")
        st.code(result.get("intent") or "-", language=None)

        st.markdown("**3. Resolved entities**")
        entities = {k: v for k, v in (result.get("entities") or {}).items() if v is not None}
        st.json(entities or {})

        st.markdown("**4. Enriched question**")
        st.caption(result.get("enriched_question") or "-")

        sql_result = result.get("sql_result")
        if sql_result:
            st.markdown("**5. SQL executed**")
            st.code(sql_result.get("sql") or "-", language="sql")
            if sql_result.get("error"):
                st.warning(f"Error: {sql_result['error']}")
            else:
                st.caption(f"{sql_result.get('row_count', 0)} row(s) returned")

        vector_result = result.get("vector_result")
        if vector_result:
            st.markdown("**6. Vector matches**")
            for hit in vector_result:
                meta = hit.get("metadata", {})
                st.caption(f"{meta.get('style_tag', '?')} ({meta.get('source', '?')}) - distance {hit.get('distance')}")

    if result.get("trace_url"):
        st.markdown("---")
        st.link_button("Open full trace in Langfuse", result["trace_url"])


st.title("Fashion Intelligence Agent")

left, right = st.columns([2, 1])

with left:
    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            _render_answer(turn["result"])

    question = st.chat_input("Ask about trends, competitors, or your own catalog...")
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = run_agent(question, thread_id=st.session_state.thread_id)
            _render_answer(result)
        st.session_state.history.append({"question": question, "result": result})
        st.rerun()

with right:
    if st.session_state.history:
        _render_trace_panel(st.session_state.history[-1]["result"])
    else:
        st.caption("Ask a question to see its trace here.")
