CLAUDE.md — Fashion Intelligence Platform
Status: Fully built and verified end-to-end - offline pipeline, online query
agent, observability/eval, and a chat UI. This file documents what actually
exists, not a design proposal. Structured so each numbered section maps
to one slide.

===============================================================
1. What this is
===============================================================
A fashion trend intelligence system with two halves:

  OFFLINE PIPELINE (runs once per data refresh, before any question is asked)
    Competitor/Self/IG raw data -> matching -> metric normalization
      -> SQLite (structured facts) + Chroma (semantic search)

  ONLINE QUERY AGENT (runs per user question, in a browser chat UI)
    Question -> guardrails -> RAG enrichment -> SQL-gen and/or vector
      retrieval -> narrative answer, with every step traced in Langfuse

Built against real generated mock data (final_data_v3/: 90 tagged images,
32 competitor SKUs, 12 self SKUs, 46 IG posts, all in the streetwear_tops
category - the other 3 taxonomy categories are defined but have no data yet).

===============================================================
2. Offline pipeline - data flow
===============================================================
Competitor/Self/IG raw  ->  Matching rules  ->  Metric normalization
    ->  Unified trend_signals  ->  Structured DB (SQLite) + Vector store (Chroma)

Matching (Step 1) and metric normalization (Step 2) run ONCE, producing one
in-memory trend_signals list. That single list is written to BOTH stores in
the same build_storage.py run, so they can never drift out of sync:

Unified trend_signals (in memory)
        |
        +--> Structured DB  (output/fashion_intel.db)
        |      - trend_signals table (48 rows)
        |      - competitor_pdp / self_catalog / ig_posts pass-through tables
        |
        +--> Vector store (output/chroma_db/, collection "fashion_intel")
               - 90 embeddings, sentence-transformers (all-MiniLM-L6-v2, 384-dim)

Database engine note: SQLite, not a client-server DB - right call at this
data volume (~50-100 rows) and single-machine deployment. Swapping to a
server DB later is a straightforward schema-preserving change, not needed now.

===============================================================
3. Step 1 - Matching (src/pipeline/matching.py)
===============================================================
Mock data comes with a ground-truth manifest (image_manifest.json) recording
the correct style_tag for every image, assigned at generation time. So
matching is a direct join, not inference:

  lookup = {e["used_by_record"]: e for e in manifest if e["used_by_dataset"] == "ig_posts.json"}
  # join ig_posts on post_id == manifest entry's used_by_record
  # style_tag from the manifest; category derived via TAG_TO_CATEGORY

Result: 46/46 IG posts matched, 0 unmatched, 0 vision-API calls needed.
Competitor/self records pass through unchanged (already carry category/
style_tag natively).

Production path, not currently exercised: keyword_match()/vision_match()
are implemented and documented in matching.py as the correct approach for
real (non-mock) IG data - keyword matching on caption_text/hashtags, with
a vision-model fallback - ready to swap in without touching anything else
once real, undecorated data exists.

===============================================================
4. Step 2/3 - Metric normalization + trend_signals (src/pipeline/build_storage.py)
===============================================================
def week_over_week_trend(series):
    start, end = series[0], series[-1]
    pct_change = 0.0 if start == 0 and end == 0 else (1.0 if start == 0 else (end - start) / start)
    momentum = "rising" if pct_change > 0.10 else "declining" if pct_change < -0.10 else "flat"
    return round(pct_change, 3), momentum

Applied to competitor rating_velocity, self sell_through_rate, and IG
likes+comments aggregated per (category, style_tag, week) BEFORE trending -
IG rows represent a tag-week cluster, never one post's raw numbers.

trend_signals: 48 rows - 20 columns, PRIMARY KEY (source, source_id)
  competitor: 32   self: 12   instagram: 4
  momentum:   rising 24, flat 12, declining 12

Plus 3 pass-through tables (competitor_pdp, self_catalog, ig_posts), each
storing the full original record as a JSON blob keyed by its natural ID -
this backs evidence lookups; trend_signals doesn't carry every display
field (bullets, captions, full price/rating detail).

===============================================================
5. Step 4 - Vector store (src/pipeline/build_vector_store.py)
===============================================================
Embedding: sentence-transformers all-MiniLM-L6-v2 (384-dim). Fail-fast
policy - if the model can't load, the build raises, no silent fallback
(this replaced an earlier TF-IDF version entirely).

90 vectors (32 competitor + 12 self + 46 individual IG posts - one per raw
record, not per tag-cluster, so semantic search can surface specific
evidence). Metadata is enriched beyond the join key: momentum, brand_or_
handle, price, oos_status_current for competitor/self; momentum, influencer_
handle, engagement for instagram - fields omitted (not null) where inapplicable.

collection.add(embedding_function=None, ...) - Chroma's own embedding
function is bypassed; vectors are computed explicitly so the embedding
model is fully controlled and swappable.

===============================================================
6. Online query agent - architecture
===============================================================
                 START
                   |
              reset_node  (clears per-turn state; see #9)
                   |
             guardrails_node  --[blocked]--> END (refusal)
                   |  [passed]
                   v
            orchestrator_node  <---------------------------+
         (supervisor hub - re-entered after each step,      |
          decides next action from what's already resolved) |
                   |                                        |
     dispatches: enrichment / sql / vector -------------------+
                   |
     finalizes:    structured -> formatter
                    else       -> synthesizer
                   |
                  END

Built with LangGraph (StateGraph + MemorySaver checkpointer, one thread_id
per conversation). Every node is a plain Python function; the orchestrator
is not a separate control-flow object - it decides one action at a time by
inspecting which state keys are already populated.

===============================================================
7. Guardrails (src/agent/nodes/guardrails.py)
===============================================================
Real NeMo Guardrails (not a custom keyword filter), "self check input" rail
only. Blocks jailbreak/prompt-injection attempts, requests to reveal the
system prompt, and off-policy content, before the question ever reaches
the orchestrator. Uses AWS Bedrock (Claude Haiku) as the checking LLM via
a LangChain adapter. Fail-fast - if guardrails can't initialize, there is
no "skip guardrails" fallback.

===============================================================
8. RAG enrichment (src/agent/knowledge/, src/agent/knowledge_enrichment/)
===============================================================
Replaces hardcoded keyword lists with embedding-similarity lookup against
a curated glossary: 113 entries across 9 types (style_tag, source, intent,
momentum, aesthetic_concept, metric, schema_table, schema_column,
schema_join), collection "fashion_knowledge" in the same Chroma directory
as the trend vectors (collection-scoped rebuilds, so rebuilding one never
wipes the other).

Resolution is top-k inverse-distance-weighted voting per type, not plain
nearest-neighbor - proven necessary because 1-NN over one document per
category is mathematically identical to voting when the pool is that thin;
categories that needed disambiguating (intent, momentum) were expanded
with several natural-language example phrasings each after real testing
showed 1-doc-per-category wasn't discriminative enough (e.g. "trending"
initially matched momentum=flat over momentum=rising).

Enrichment does two jobs, not one:
  - structured extraction: resolves intent (structured/explanatory/
    semantic/strategic) and entities (style_tag, category, source, momentum)
  - query rewriting: appends resolved types' canonical glossary text to
    the raw question, so downstream embedding search works against richer
    text than a user's terse phrasing

===============================================================
9. Schema-RAG SQL generation (src/agent/nodes/sql_node.py)
===============================================================
Not fixed parameterized query functions - the LLM writes real SQL, narrowed
and checked at every step:

  schema_link() -> generate_sql() -> validate_sql() -> execute_sql()

  - schema_link: retrieves the tables/columns/joins relevant to the
    question from fashion_knowledge (all 4 tables/32 columns/3 joins are
    always shown, not distance-filtered - tested and found more reliable
    than narrowing at this schema's small scale)
  - generate_sql: Bedrock Haiku, temperature 0, constrained by the
    retrieved schema slice
  - validate_sql: single SELECT/WITH statement only, forbidden-keyword
    blacklist (INSERT/UPDATE/DELETE/DROP/...), every FROM/JOIN table must
    be one of the 4 known tables, LIMIT auto-appended if missing
  - execute_sql: read-only SQLite connection (mode=ro) as defense in depth
    beneath the validation layer

Self-correcting: on a SQLite execution error, the exact error is fed back
to the model for one bounded repair attempt before giving up - added after
a real, reproducible slip (the model wrote `p.thumbnail_url` directly
instead of `json_extract(p.data, '$.thumbnail_url')` for a pass-through
table) that validate_sql couldn't catch statically but SQLite's own error
caught safely.

===============================================================
10. Vector retrieval + synthesis
===============================================================
vector_node.py: semantic_search over the fashion_intel collection for
semantic/strategic questions, using the enriched (not raw) question text;
a separate find_similar() mode handles "similar to X but not tagged that
way" queries via centroid distance, excluding the exact tag.

synthesizer.py: the one LLM call that turns retrieved evidence into a
narrative answer - Key Insights (grounded bullet points, explicitly
allowed to say "the data doesn't explain this" rather than guess) and
Suggested Actions (or "no specific action is supported by this data
alone"). Explicitly forbidden from adding demographic/audience language
not present in the evidence - added after a real case where a question
phrased "...among youths" got answered as if the data confirmed a youth
trend, when no demographic column exists anywhere in this schema.

formatter.py: only single-aggregate structured answers (e.g. "AVG(price):
51.5625") stay fully deterministic, no LLM. Any multi-row structured
result delegates to the same synthesizer bullet treatment - a table alone
doesn't say what a viewer should notice, and this schema's rows regularly
span wildly different scales across sources (Instagram engagement in the
thousands vs. a competitor rating around 12 vs. a self sell-through rate
around 0.5) that deserve real narrative framing, not just a formatted table.

===============================================================
11. Observability - Langfuse (MELT: Metrics, Eval, Log, Traces)
===============================================================
Every run_agent() call is one root trace, tagged by conversation thread_id,
with a nested, fully typed span per pipeline step - not just the 7 graph
nodes, but the retrieval/generation calls INSIDE them:

  agent_turn (root)
   |- reset, guardrails
   |- orchestrator (re-entered per dispatched step)
   |- enrichment
   |   |- knowledge_retrieval   (RETRIEVER - Chroma query against fashion_knowledge)
   |   `- query_rewrite
   |- sql
   |   |- schema_retrieval      (RETRIEVER - tables/columns/joins retrieved)
   |   |- sql_generation        (GENERATION - model, tokens, prompt/completion)
   |   `- sql_execution
   |- vector
   |   `- vector_retrieval      (RETRIEVER - hits + distances)
   `- formatter / synthesizer
       `- synthesis / structured_summary  (GENERATION)

Deliberately graceful, unlike the fail-fast policy for the embedding model
and guardrails: those are core intelligence dependencies, tracing is not -
a Langfuse outage degrades to "no tracing," never a crash.

LLM-as-judge eval (src/agent/eval/): 10 questions spanning structured/
explanatory/semantic/strategic intents, each run through the real agent,
then scored by a second Bedrock call on groundedness and relevance (1-5),
with scores attached back to the originating trace via Langfuse's scoring
API. Caught two real bugs this way (a SQL column slip, a demographic-
language hallucination) - both fixed and re-verified against the same
eval set.

===============================================================
12. Chat UI (src/agent/ui/app.py, Streamlit)
===============================================================
Left: a standard chat thread (st.chat_input/st.chat_message), one
thread_id per browser session so MemorySaver's conversation memory
(last 3 turns, then truncated) actually accumulates. Each answer renders
as three stacked boxes, in order:
  1. Enriched Query  - what enrichment actually resolved the question to
  2. Data (optional) - the raw SQL rows as a real table; a thumbnail_url
     column is rendered as an actual image gallery instead of a raw path
  3. Answer           - the final bullet-point or single-value answer

Right: a live trace panel for the latest turn - guardrail outcome,
resolved intent/entities, enriched question, generated SQL + row count,
vector matches - plus a link out to the full Langfuse trace for deep-dives
(token usage, latency, judge scores).

Run locally (`streamlit run src/agent/ui/app.py`), reached via SSH local
port forwarding (`ssh -L 8501:localhost:8501 ...`) rather than public
hosting - see #13.

===============================================================
13. Deployment decision
===============================================================
Evaluated Streamlit Community Cloud's free tier and ruled it out:
measured memory footprint at idle is already ~1GB (torch + sentence-
transformers + chromadb + langgraph + nemoguardrails loaded), at or above
the free tier's 1GB cap. Free tier also requires a public GitHub repo
(would expose the SQLite DB / Chroma data) and has no built-in auth (a
public URL would let anyone burn this AWS account's real Bedrock spend
with no rate limiting). Currently run locally, reached via SSH tunnel only.

===============================================================
14. Tech stack
===============================================================
  Data:          SQLite (fashion_intel.db), ChromaDB (persistent, 2 collections)
  Embeddings:    sentence-transformers (all-MiniLM-L6-v2, 384-dim)
  LLM:           AWS Bedrock, Claude Haiku (us.anthropic.claude-haiku-4-5),
                 via boto3 with auto-refreshing STS-assumed-role credentials
  Orchestration: LangGraph (StateGraph, MemorySaver checkpointer)
  Guardrails:    NeMo Guardrails (self-check-input rail)
  Observability: Langfuse Cloud (tracing, generation/retriever spans, scoring)
  Eval:          custom LLM-as-judge harness (10-case set, groundedness + relevance)
  UI:            Streamlit
  Language:      Python 3.10, stdlib unittest for tests (no pytest dependency)

===============================================================
15. Key numbers (for a results/summary slide)
===============================================================
  IG posts matched:        46 / 46 (100%, manifest ground truth)
  trend_signals rows:      48  (competitor 32, self 12, instagram 4)
  Momentum distribution:   rising 24, flat 12, declining 12
  Vector store entries:    90  (fashion_intel collection)
  Knowledge base entries:  113 (fashion_knowledge collection, 9 types)
  Eval set:                10 questions, all 4 intent categories
  Eval scores (last run):  groundedness ~4.6-4.9/5, relevance ~3.6-4.2/5
                            (run-to-run variance tracked in Langfuse -
                             see #11; two real bugs found and fixed this way)

===============================================================
16. Known limitations (worth a slide of its own)
===============================================================
  - Intent classification isn't perfect (~90% on held-out questions) -
    embedding-based routing occasionally misclassifies a terse/ambiguous
    question (e.g. "which influencer posts most about X" as semantic
    instead of structured).
  - Only streetwear_tops has real data; 3 of 4 taxonomy categories are
    defined but empty - the pipeline doesn't hardcode around that, but
    there's nothing to demo there yet.
  - Conversation memory is intentionally simple: exact-repeat detection
    over the last 3 turns, not coreference resolution for follow-ups like
    "what about X" (though history is still passed to synthesis as context).
  - Column existence in generated SQL isn't statically validated (would
    need a real SQL parser) - relies on SQLite's own error at execution
    time, caught safely since the connection is read-only, with one bounded
    self-correction retry before giving up.
