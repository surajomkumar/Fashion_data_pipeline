"""
Schema linking: retrieves the tables/columns/joins relevant to a question
from fashion_knowledge, to narrow what sql_node's LLM generation step
sees. Unlike enrich_query() (which picks one winning value per type via
voting), this returns a SET of matching docs per type - SQL generation
usually needs several columns, not one.

Narrowing matters even though the schema is tiny (4 tables, ~35 docs):
handing the LLM only the schema slice actually relevant to the question
(rather than the whole schema every time) is what later lets
sql_validation reject any table/column reference outside that slice -
the retrieval step is what makes the validation step meaningful, not
just a formality.
"""
from pathlib import Path

import chromadb

from src.agent.knowledge_enrichment.build_knowledge_index import KNOWLEDGE_COLLECTION_NAME
from src.pipeline.build_vector_store import CHROMA_DIR, embed_texts, load_embedding_model

TOP_K_BY_TYPE = {"table": 4, "column": 32, "join": 3}


def schema_link(question: str, chroma_dir: Path = CHROMA_DIR, model=None,
                 top_k_by_type: dict = TOP_K_BY_TYPE) -> dict:
    """
    Returns {"table": [...], "column": [...], "join": [...]}, each a list
    of {"id","text","distance"}, ordered by similarity but NOT distance-
    filtered for any type.

    Measured directly (see git history / conversation): at this schema's
    size (4 tables, 32 columns, 3 joins - a few thousand tokens of text
    total), distance-based narrowing actively hurts more than it helps.
    A real test case ("why is the cropped zip-up jacket declining?") had
    pct_change rank 22nd of 32 candidates and source/source_id rank near
    the bottom - a plausible max_distance threshold silently dropped them
    from the prompt, and the LLM hallucinated plausible-sounding column
    names to fill the gap (caught by validate_sql's execution-time check,
    but that's a safety net, not something to rely on routinely). The
    embedding model just doesn't discriminate well between dry technical
    column descriptions and natural question phrasing. Below this scale,
    "show everything, let the LLM's own judgment pick what's relevant" is
    more reliable than "trust cosine distance to decide what's relevant."
    top_k_by_type still caps each type at its total count, so this is
    genuinely retrieval (ranked, deduped, type-scoped), just not filtered -
    if the schema ever grows meaningfully past this size, distance
    filtering (or a higher top_k with a wide threshold) is worth
    revisiting, since showing an unbounded full schema stops being cheap.
    """
    model = model or load_embedding_model()
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_collection(KNOWLEDGE_COLLECTION_NAME)
    query_embedding = embed_texts([question], model=model)

    linked = {}
    for t, top_k in top_k_by_type.items():
        res = collection.query(query_embeddings=query_embedding, n_results=top_k, where={"type": t})
        ids, documents, distances = res["ids"][0], res["documents"][0], res["distances"][0]
        linked[t] = [
            {"id": id_, "text": doc, "distance": round(dist, 3)}
            for id_, doc, dist in zip(ids, documents, distances)
        ]
    return linked


def render_schema_context(linked: dict) -> str:
    """Flattens a schema_link() result into the text block handed to the SQL-generation prompt."""
    sections = []
    for t, label in (("table", "Tables"), ("column", "Columns"), ("join", "Joins")):
        docs = linked.get(t, [])
        if docs:
            sections.append(f"-- {label} --\n" + "\n".join(d["text"] for d in docs))
    return "\n\n".join(sections)


def known_tables(linked: dict) -> set:
    return {d["id"].split(":", 1)[1] for d in linked.get("table", [])}
