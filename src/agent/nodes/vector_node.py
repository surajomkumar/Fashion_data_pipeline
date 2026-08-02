"""
Vector node - semantic search against the fashion_intel collection.
Dispatched by the orchestrator for "semantic" and "strategic" intents
(see orchestrator.PLAN_BY_INTENT).

Uses enriched_question (raw question + resolved glossary descriptions,
see enrichment_node) as the embedding query, not the raw question -
richer text specifically helps aesthetic-language questions (category F)
where the user's own phrasing is often terser than what actually
describes the concept well.

Two modes:
  - "similar to X but not tagged that way" (question bank F25): detected
    by a small local keyword check ("similar to"/"untagged"/"not tagged")
    combined with a resolved style_tag. Same kind of narrow, low-risk
    local heuristic as the SQL sort/aggregate inference discussed earlier -
    it's a binary choice within an already-resolved "semantic" intent, not
    a case where RAG resolution adds much. Uses find_similar().
  - everything else: plain semantic_search(), where-filtered by whatever
    of category/momentum resolved. style_tag is deliberately NOT used as
    a filter here - narrowing to the exact tag would defeat the point of
    a semantic/vibe search reaching beyond the fixed taxonomy.
"""
from src.agent.nodes.state import AgentState
from src.agent.observability import traced_span
from src.pipeline.build_vector_store import build_where, find_similar, semantic_search

SIMILAR_PHRASES = ("similar to", "untagged", "haven't tagged", "have not tagged", "not tagged", "not yet tagged")
TOP_K = 5


def _format_hits(result: dict) -> list:
    if not result["ids"][0]:
        return []
    return [
        {"id": id_, "distance": round(dist, 3), "metadata": meta, "text": doc}
        for id_, meta, dist, doc in zip(
            result["ids"][0], result["metadatas"][0], result["distances"][0], result["documents"][0]
        )
    ]


def vector_node(state: AgentState) -> dict:
    question = state["question"]
    enriched_question = state.get("enriched_question") or question
    entities = state.get("entities", {})

    is_similar_query = bool(entities.get("style_tag")) and any(p in question.lower() for p in SIMILAR_PHRASES)

    with traced_span("vector_retrieval", as_type="retriever", input=enriched_question) as span:
        if is_similar_query:
            extra_where = build_where(category=entities.get("category"))
            result = find_similar(entities["style_tag"], top_k=TOP_K, exclude_same_tag=True, extra_where=extra_where)
        else:
            where = build_where(category=entities.get("category"), momentum=entities.get("momentum"))
            result = semantic_search(enriched_question, top_k=TOP_K, where=where)
        hits = _format_hits(result)
        span.update(output=hits)

    return {"vector_result": hits}
