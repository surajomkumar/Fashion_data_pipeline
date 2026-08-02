"""
Enrichment node - does both jobs implied by "enrichment": structured
extraction (intent + entities, via RAG lookup against fashion_knowledge,
replacing what would otherwise be hardcoded keyword lists) AND query
rewriting (enriched_question, the raw question with resolved types'
glossary descriptions appended - see enrich.build_enriched_question()).

category is derived from the resolved style_tag via TAG_TO_CATEGORY, not
a separately-resolved knowledge type - style_tag -> category is a fixed,
known-at-dev-time mapping (see matching.py), no need to duplicate it as
its own RAG lookup.

Defaults to intent="structured" when nothing resolves confidently - the
cheapest, LLM-free path, which is the safe failure mode if the question
doesn't clearly match any category.
"""
from src.agent.knowledge_enrichment.enrich import build_enriched_question, enrich_query
from src.agent.nodes.state import AgentState
from src.agent.observability import traced_span
from src.pipeline.matching import TAG_TO_CATEGORY

DEFAULT_INTENT = "structured"


def enrichment_node(state: AgentState) -> dict:
    question = state["question"]

    with traced_span("knowledge_retrieval", as_type="retriever", input=question) as span:
        matches = enrich_query(question)
        span.update(output=matches)

    style_tag = matches.get("style_tag", {}).get("value")
    entities = {
        "style_tag": style_tag,
        "category": TAG_TO_CATEGORY.get(style_tag) if style_tag else None,
        "source": matches.get("source", {}).get("value"),
        "momentum": matches.get("momentum", {}).get("value"),
    }

    with traced_span("query_rewrite", input={"question": question, "matches": matches}) as span:
        enriched_question = build_enriched_question(question, matches)
        span.update(output=enriched_question)

    return {
        "knowledge_matches": matches,
        "intent": matches.get("intent", {}).get("value", DEFAULT_INTENT),
        "entities": entities,
        "enriched_question": enriched_question,
    }
