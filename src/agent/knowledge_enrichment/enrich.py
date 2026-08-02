"""
RAG enrichment: resolves a raw user question against the fashion_knowledge
collection instead of hardcoded keyword lists. Called by the enrichment
node early in the graph; its output (per-type resolved value + distance)
is what the orchestrator/router logic uses to set intent and entities.

Resolution is top-k inverse-distance-weighted voting per type, not a
single nearest-neighbor lookup. This matters specifically for `intent`:
each intent category is backed by several example-phrasing documents
(see knowledge/intents.json), not one description each, since a single
"1-NN over one doc per category" match proved unreliable in testing -
short, abstract questions could land closer to the wrong intent's lone
description than the right one. Voting rewards a category with several
close matches over one with a single lucky hit; taking the plain nearest
neighbor within a candidate pool is mathematically identical regardless
of pool size (the global minimum distance is always the top-1 result),
so only voting actually benefits from having more documents per value.

A value is only returned if at least one of its supporting matches is
within max_distance; anything weaker is dropped rather than forced -
callers treat a missing type as "unresolved", not "no filter for this
dimension".
"""
from pathlib import Path

import chromadb

from src.agent.knowledge_enrichment.build_knowledge_index import KNOWLEDGE_COLLECTION_NAME
from src.pipeline.build_vector_store import CHROMA_DIR, embed_texts, load_embedding_model

DEFAULT_TYPES = ("style_tag", "source", "intent", "momentum", "aesthetic_concept")
MAX_DISTANCE = 0.65
TOP_K = 5

# Which metadata key holds the actual resolved value for each knowledge type -
# fixed at development time, same reasoning as the SQL schema mapping: we
# control the knowledge base content, so this isn't inferred at runtime.
VALUE_KEY_BY_TYPE = {
    "style_tag": "style_tag",
    "source": "source",
    "intent": "intent",
    "momentum": "momentum",
    "aesthetic_concept": "concept",
    "metric": "metric",
}


def enrich_query(question: str, types: tuple = DEFAULT_TYPES, top_k: int = TOP_K,
                  max_distance: float = MAX_DISTANCE, chroma_dir: Path = CHROMA_DIR, model=None) -> dict:
    """
    Returns {type: {"value", "distance", "votes"}} for every type with a
    confident match. `distance` is the best (minimum) distance actually
    observed for the winning value; `votes` is its aggregate inverse-
    distance weight, useful for debugging close calls.
    """
    model = model or load_embedding_model()
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_collection(KNOWLEDGE_COLLECTION_NAME)
    query_embedding = embed_texts([question], model=model)

    resolved = {}
    for t in types:
        value_key = VALUE_KEY_BY_TYPE[t]
        res = collection.query(query_embeddings=query_embedding, n_results=top_k, where={"type": t})
        metadatas, distances = res["metadatas"][0], res["distances"][0]
        if not metadatas:
            continue

        scores, best_distance = {}, {}
        for meta, dist in zip(metadatas, distances):
            if dist > max_distance:
                continue
            value = meta[value_key]
            scores[value] = scores.get(value, 0.0) + 1.0 / (dist + 1e-6)
            best_distance[value] = min(best_distance.get(value, dist), dist)

        if not scores:
            continue

        winner = max(scores, key=scores.get)
        resolved[t] = {
            "value": winner,
            "distance": round(best_distance[winner], 3),
            "votes": round(scores[winner], 2),
        }

    return resolved


# Types whose canonical description is worth appending to the query text.
# "intent" is deliberately excluded - it's a routing signal (which plan to
# run), not content that should shape what gets embedded/searched.
REWRITE_TYPES = ("style_tag", "source", "momentum", "aesthetic_concept")


def build_enriched_question(question: str, knowledge_matches: dict, types: tuple = REWRITE_TYPES,
                             chroma_dir: Path = CHROMA_DIR) -> str:
    """
    Query rewriting: appends the canonical glossary description for each
    resolved type to the raw question, so downstream embedding (vector_node)
    searches against richer text than the user's often-terse phrasing -
    e.g. "Y2K-adjacent" alone embeds less precisely than "Y2K-adjacent
    early-2000s revival aesthetic - low-rise silhouettes, metallics,
    logomania, butterfly motifs, bold graphics".

    Each knowledge JSON file's main description entry uses the id pattern
    "{type}:{value}" (distinct from the ":ex1".. example-phrasing docs), so
    the canonical text is a direct id lookup, not another similarity search.
    Falls back to the raw question unchanged if nothing resolved.
    """
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_collection(KNOWLEDGE_COLLECTION_NAME)

    canonical_ids = [f"{t}:{knowledge_matches[t]['value']}" for t in types if t in knowledge_matches]
    if not canonical_ids:
        return question

    got = collection.get(ids=canonical_ids, include=["documents"])
    return " ".join([question] + got["documents"])
