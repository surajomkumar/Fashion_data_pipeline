"""
Builds the fashion_knowledge Chroma collection - a RAG-queryable glossary
(style tags, sources, intents, aesthetic concepts, metrics) used to
resolve a raw user question into structured entities/intent before
routing, instead of hardcoded keyword lists in the router node.

Lives in the SAME persistent store as the data collection
(output/chroma_db/) but as its own collection ("fashion_knowledge",
separate from "fashion_intel"), so the two can be rebuilt independently -
see build_vector_store.py, which now only deletes its own collection on
rebuild rather than the whole chroma_dir, precisely so this collection
survives a data pipeline re-run.

Source content: src/agent/knowledge/*.json - each file is a flat list of
entries: {"id", "type", "text", "metadata"}. Add new vocabulary (slang,
synonyms, new aesthetic concepts) by editing those files and re-running
this script; no code change needed here.

Uses the same embedding model and the same fail-fast policy as
build_vector_store.py (load_embedding_model() raises if the model can't
be loaded - no fallback).
"""
import json
import logging
from pathlib import Path

import chromadb

from src.pipeline.build_vector_store import CHROMA_DIR, embed_texts, load_embedding_model

logger = logging.getLogger(__name__)

KNOWLEDGE_DIR = Path(__file__).resolve().parents[1] / "knowledge"
KNOWLEDGE_COLLECTION_NAME = "fashion_knowledge"
REQUIRED_FIELDS = {"id", "type", "text", "metadata"}


def load_entries(knowledge_dir: Path = KNOWLEDGE_DIR) -> list:
    entries = []
    for path in sorted(knowledge_dir.glob("*.json")):
        with open(path, "r") as f:
            file_entries = json.load(f)
        for e in file_entries:
            missing = REQUIRED_FIELDS - e.keys()
            if missing:
                raise ValueError(f"{path.name}: entry {e.get('id')!r} missing fields {missing}")
        entries.extend(file_entries)

    ids = [e["id"] for e in entries]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"Duplicate knowledge entry ids across {knowledge_dir}/*.json: {dupes}")

    return entries


def build_knowledge_index(entries: list, model=None, chroma_dir: Path = CHROMA_DIR) -> None:
    """Full rebuild of just the fashion_knowledge collection - fashion_intel is untouched."""
    model = model or load_embedding_model()
    chroma_dir.mkdir(parents=True, exist_ok=True)

    embeddings = embed_texts([e["text"] for e in entries], model=model)

    client = chromadb.PersistentClient(path=str(chroma_dir))
    try:
        client.delete_collection(KNOWLEDGE_COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=KNOWLEDGE_COLLECTION_NAME,
        embedding_function=None,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        ids=[e["id"] for e in entries],
        embeddings=embeddings,
        metadatas=[e["metadata"] for e in entries],
        documents=[e["text"] for e in entries],
    )


def run_build_knowledge_index(knowledge_dir: Path = KNOWLEDGE_DIR, chroma_dir: Path = CHROMA_DIR) -> dict:
    entries = load_entries(knowledge_dir)
    build_knowledge_index(entries, chroma_dir=chroma_dir)

    by_type = {}
    for e in entries:
        by_type[e["type"]] = by_type.get(e["type"], 0) + 1

    return {"entries": len(entries), "by_type": by_type, "chroma_dir": str(chroma_dir)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    summary = run_build_knowledge_index()
    print(json.dumps(summary, indent=2))
