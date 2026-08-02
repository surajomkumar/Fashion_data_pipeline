"""
Step 4 - Vector store (Chroma at output/chroma_db/).

Embeds with sentence-transformers' all-MiniLM-L6-v2 - a small, fast,
local 384-dim model. This is a hard dependency: if the model can't be
loaded (package missing, weights not downloadable, no local HF cache),
load_embedding_model() raises and the run fails. There is no fallback
embedding path.

One embedding per raw record (not per IG tag-cluster): 32 competitor +
12 self + 46 individual IG posts = 90 vectors, so semantic search can
surface specific evidence, not just tag-level summaries, and so a record
can be found by similarity even when its style_tag isn't the one being
searched for (see find_similar()). Unmatched IG posts (no category/
style_tag) are skipped since their metadata would be incomplete - none
occur in the current run.

Chroma's own embedding function is bypassed entirely
(embedding_function=None at collection creation) so this fixed, known
model computes the vectors and collection.add() receives them directly.
The model is identified by name and re-downloaded (or read from the
local HF cache) at query time, so the same vector space is reproduced as
long as MODEL_NAME doesn't change.

Design note (why the metadata is richer than just source/category/tag):
most of the question bank this store needs to support is answered by SQL
against trend_signals (momentum, velocity, price, stock) - the vector
store's real job is (a) semantic reach outside the fixed style_tag
vocabulary ("Y2K-adjacent", "elevated basics"), and (b) fetching evidence
records to back a structured answer. Both jobs are cheaper as ONE Chroma
call (embedding similarity + metadata `where` filter together) than a
similarity search followed by a second SQL round-trip, so momentum,
price, engagement, brand/handle and stock status are folded into
metadata at build time by joining back to fashion_intel.db. Chroma
tolerates different metadata keys per record, so fields that don't apply
to a source (e.g. price for instagram) are simply omitted, not set to
None - Chroma metadata values must be str/int/float/bool, None isn't a
valid value.

Full rebuild every run: this collection is dropped and recreated each
time - but only this collection, not the whole chroma_dir, since
fashion_knowledge (see src/agent/knowledge_enrichment/) lives in the same
persistent store and must survive a data rebuild.
Depends on output/fashion_intel.db already existing (built by
build_storage.py) for the momentum join - fails clearly if it's missing.
"""
import json
import logging
import sqlite3
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output"
CHROMA_DIR = OUTPUT_DIR / "chroma_db"
DB_PATH = OUTPUT_DIR / "fashion_intel.db"
COLLECTION_NAME = "fashion_intel"
MODEL_NAME = "all-MiniLM-L6-v2"


def load_embedding_model() -> SentenceTransformer:
    """
    Loads the sentence-transformers model. Fails loudly and immediately if
    it can't be - no silent fallback to any other embedding method.
    """
    try:
        return SentenceTransformer(MODEL_NAME)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load embedding model '{MODEL_NAME}'. This pipeline requires it - "
            f"there is no fallback embedding path. Check that sentence-transformers is "
            f"installed and that the model is reachable (huggingface.co) or already cached "
            f"locally. Original error: {e}"
        ) from e


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def load_momentum_lookup(db_path: Path = DB_PATH) -> dict:
    """
    Reads trend_signals and returns two lookups used to stamp momentum onto
    vector metadata at build time:
      - by_source_id[(source, source_id)] -> momentum   (competitor/self, exact SKU)
      - by_tag[(category, style_tag)]     -> momentum   (instagram, cluster-level -
        individual IG posts inherit their (category, style_tag) cluster's momentum,
        since trend_signals has no per-post row for instagram)
    """
    if not db_path.exists():
        raise FileNotFoundError(
            f"{db_path} does not exist. build_vector_store.py joins momentum from "
            f"trend_signals, so build_storage.py must run first."
        )

    conn = sqlite3.connect(db_path)
    try:
        by_source_id, by_tag = {}, {}
        for source, source_id, category, style_tag, momentum in conn.execute(
            "SELECT source, source_id, category, style_tag, momentum FROM trend_signals"
        ):
            if source == "instagram":
                by_tag[(category, style_tag)] = momentum
            else:
                by_source_id[(source, source_id)] = momentum
        return {"by_source_id": by_source_id, "by_tag": by_tag}
    finally:
        conn.close()


def competitor_text(r: dict) -> str:
    bullets = " ".join(r.get("bullet_description") or [])
    return (
        f"{r.get('brand', '')} {r.get('title', '')}. {bullets}. "
        f"{r.get('color_palette', '')}. {r.get('category', '')} {r.get('style_tag', '')}"
    ).strip()


def self_text(r: dict) -> str:
    return (
        f"{r.get('style_name', '')} {r.get('title', '')}. {r.get('color_palette', '')}. "
        f"{r.get('category', '')} {r.get('style_tag', '')}"
    ).strip()


def ig_text(r: dict) -> str:
    hashtags = " ".join(r.get("hashtags") or [])
    return f"{r.get('caption_text', '')} {hashtags}. {r.get('category', '')} {r.get('style_tag', '')}".strip()


def build_corpus(competitor_matched: list, self_matched: list, ig_matched: list, momentum_lookup: dict
                  ) -> tuple[list, list, list]:
    """Returns (ids, texts, metadatas) for every embeddable raw record."""
    ids, texts, metadatas = [], [], []
    by_source_id = momentum_lookup["by_source_id"]
    by_tag = momentum_lookup["by_tag"]

    for r in competitor_matched:
        ids.append(f"competitor::{r['competitor_sku']}")
        texts.append(competitor_text(r))
        meta = {
            "source": "competitor", "source_id": r["competitor_sku"],
            "category": r["category"], "style_tag": r["style_tag"],
            "brand_or_handle": r["brand"], "price": r["price"],
            "oos_status_current": bool(r["oos_status_current"]),
        }
        momentum = by_source_id.get(("competitor", r["competitor_sku"]))
        if momentum is not None:
            meta["momentum"] = momentum
        metadatas.append(meta)

    for r in self_matched:
        ids.append(f"self::{r['sku']}")
        texts.append(self_text(r))
        meta = {
            "source": "self", "source_id": r["sku"],
            "category": r["category"], "style_tag": r["style_tag"],
            "brand_or_handle": "self", "price": r["price"],
            "oos_status_current": bool(r["oos_status_current"]),
        }
        momentum = by_source_id.get(("self", r["sku"]))
        if momentum is not None:
            meta["momentum"] = momentum
        metadatas.append(meta)

    for r in ig_matched:
        if r.get("tag_source") == "unmatched":
            logger.warning(f"build_corpus: skipping unmatched IG post {r.get('post_id')} (no category/style_tag)")
            continue
        ids.append(f"instagram::{r['post_id']}")
        texts.append(ig_text(r))
        meta = {
            "source": "instagram", "source_id": r["post_id"],
            "category": r["category"], "style_tag": r["style_tag"],
            "brand_or_handle": r["influencer_handle"],
            "engagement": (r.get("likes") or 0) + (r.get("comments_count") or 0),
        }
        momentum = by_tag.get((r["category"], r["style_tag"]))
        if momentum is not None:
            meta["momentum"] = momentum
        metadatas.append(meta)

    return ids, texts, metadatas


def embed_texts(texts: list, model: SentenceTransformer) -> list:
    return model.encode(texts, show_progress_bar=False, normalize_embeddings=True).tolist()


def build_vector_store(ids: list, texts: list, metadatas: list, model: SentenceTransformer,
                        chroma_dir: Path = CHROMA_DIR) -> None:
    """
    Full rebuild of just this collection, not the whole chroma_dir - other
    collections in the same persistent client (e.g. fashion_knowledge, see
    src/agent/knowledge_enrichment/) must survive a rebuild of this one.
    """
    chroma_dir.mkdir(parents=True, exist_ok=True)
    embeddings = embed_texts(texts, model=model)

    client = chromadb.PersistentClient(path=str(chroma_dir))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=texts)


def run_build_vector_store(output_dir: Path = OUTPUT_DIR, chroma_dir: Path = CHROMA_DIR,
                            db_path: Path = DB_PATH) -> dict:
    model = load_embedding_model()
    momentum_lookup = load_momentum_lookup(db_path)

    competitor_matched = load_json(output_dir / "competitor_pdp_matched.json")
    self_matched = load_json(output_dir / "self_catalog_matched.json")
    ig_matched = load_json(output_dir / "ig_posts_matched.json")

    ids, texts, metadatas = build_corpus(competitor_matched, self_matched, ig_matched, momentum_lookup)
    build_vector_store(ids, texts, metadatas, model, chroma_dir=chroma_dir)

    by_source = {}
    for m in metadatas:
        by_source[m["source"]] = by_source.get(m["source"], 0) + 1

    return {
        "vectors": len(ids),
        "by_source": by_source,
        "model": MODEL_NAME,
        "chroma_dir": str(chroma_dir),
    }


def build_where(**conditions) -> dict | None:
    """
    Builds a Chroma `where` filter from named conditions, skipping any that
    are None. Chroma 1.x rejects a flat multi-key dict like
    {"category": "x", "momentum": "y"} outright ("Expected where to have
    exactly one operator") - it must be wrapped in "$and" once there's more
    than one condition. Returns None (no filter) if every condition is None.
    """
    clauses = [{k: v} for k, v in conditions.items() if v is not None]
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def semantic_search(query: str, top_k: int = 5, where: dict = None, chroma_dir: Path = CHROMA_DIR,
                     model: SentenceTransformer = None):
    """
    Embed a query with the same model and search the collection.
    `where` is a Chroma metadata filter - build it with build_where(...) if
    you have more than one condition, e.g. build_where(category="streetwear_tops",
    momentum="rising"), so "trending AND aesthetically like X" works in one call.
    """
    model = model or load_embedding_model()
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_collection(name=COLLECTION_NAME)
    query_embedding = embed_texts([query], model=model)
    return collection.query(query_embeddings=query_embedding, n_results=top_k, where=where)


def find_similar(anchor_style_tag: str, top_k: int = 5, exclude_same_tag: bool = True,
                  extra_where: dict = None, chroma_dir: Path = CHROMA_DIR, model: SentenceTransformer = None):
    """
    "Similar to [anchor_style_tag] but not tagged that way" (question bank F25):
    builds a centroid from every vector already carrying anchor_style_tag,
    then searches near that centroid while excluding that exact tag - so
    what comes back is semantically close but sits outside the known
    taxonomy bucket, which is the whole point of embedding raw records
    individually instead of one vector per tag-cluster.

    extra_where, if given, must already be a single valid Chroma clause
    (e.g. one condition, or something already built with build_where()) -
    it gets combined with the tag-exclusion clause via $and, and Chroma
    rejects a flat multi-key dict nested inside $and just as much as at
    the top level.
    """
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_collection(name=COLLECTION_NAME)

    anchor = collection.get(where={"style_tag": anchor_style_tag}, include=["embeddings"])
    if not len(anchor["embeddings"]):
        raise ValueError(f"No vectors found with style_tag={anchor_style_tag!r}")

    import numpy as np
    centroid = np.mean(anchor["embeddings"], axis=0).tolist()

    clauses = [extra_where] if extra_where else []
    if exclude_same_tag:
        clauses.append({"style_tag": {"$ne": anchor_style_tag}})
    where = clauses[0] if len(clauses) == 1 else ({"$and": clauses} if clauses else None)

    return collection.query(query_embeddings=[centroid], n_results=top_k, where=where)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    summary = run_build_vector_store()
    print(json.dumps(summary, indent=2))
