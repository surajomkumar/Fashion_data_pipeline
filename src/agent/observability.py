"""
Langfuse wiring - the MELT (Metrics, Eval, Log, Traces) observability layer
over lang_graph.py. Kept as a separate module (rather than importing
Langfuse directly in each node) so every file that needs it does so
through one place: get_langfuse() for the client singleton, and
traced_bedrock_converse() for the one Bedrock call shape sql_node.py and
synthesizer.py both use.

Deliberately graceful, unlike the embedding-model/guardrails fail-fast
policy elsewhere in this codebase: those are core intelligence
dependencies (no answer is possible without them), tracing is not - a
Langfuse outage should never take the agent down. get_langfuse() logs a
warning and returns None on init failure; every call site checks for
None before using the client, so a bad LANGFUSE_* config degrades to "no
tracing" rather than a crash.

LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL are read
directly from the environment by the Langfuse() constructor itself - see
langfuse/_client/client.py - so nothing here needs to touch os.environ,
same .env file the AWS_* credentials already live in (loaded via
python-dotenv in src/utils/aws_cred.py, already imported transitively by
every node that calls call_cred()).
"""
import logging
from contextlib import contextmanager

from langfuse import Langfuse

logger = logging.getLogger(__name__)

_langfuse = None
_init_failed = False


def get_langfuse():
    """Returns the singleton Langfuse client, or None if it couldn't be built (logs once)."""
    global _langfuse, _init_failed
    if _langfuse is not None or _init_failed:
        return _langfuse
    try:
        _langfuse = Langfuse()
        return _langfuse
    except Exception as e:
        logger.warning(f"Langfuse init failed - tracing disabled for this run: {e}")
        _init_failed = True
        return None


def traced_bedrock_converse(client, model_id: str, messages: list, name: str, inference_config: dict = None):
    """
    Shared wrapper around bedrock_client.converse() for the two raw-boto3
    Bedrock call sites (sql_node.generate_sql, synthesizer.synthesize_narrative).
    Logs a Langfuse generation observation (model, prompt/completion text,
    token usage from the Converse API's own `usage` field) around the call.
    Falls back to a plain untraced call if Langfuse isn't available.

    Returns the raw Converse API response dict, same as calling
    client.converse(...) directly - callers still pull the text out of
    response["output"]["message"]["content"][0]["text"] themselves.
    """
    langfuse = get_langfuse()
    kwargs = {"modelId": model_id, "messages": messages}
    if inference_config:
        kwargs["inferenceConfig"] = inference_config

    if langfuse is None:
        return client.converse(**kwargs)

    with langfuse.start_as_current_observation(name=name, as_type="generation", model=model_id) as gen:
        response = client.converse(**kwargs)
        usage = response.get("usage", {})
        output_text = response["output"]["message"]["content"][0]["text"]
        gen.update(
            input=messages,
            output=output_text,
            usage_details={
                "input": usage.get("inputTokens"),
                "output": usage.get("outputTokens"),
                "total": usage.get("totalTokens"),
            },
        )
    return response


class _NoOpSpan:
    """Returned by traced_span() when Langfuse is unavailable - lets call
    sites unconditionally call span.update(output=...) without an `if
    langfuse:` check at every call site."""

    def update(self, **kwargs):
        pass


@contextmanager
def traced_span(name: str, as_type: str = "span", input=None):
    """
    Wraps a block of retrieval/transform code (a Chroma query, a schema
    lookup, a query rewrite - anything that isn't a raw Bedrock call, which
    traced_bedrock_converse already covers) as its own named Langfuse
    observation, nested under whatever span is currently active.

    Added because the RAG retrieval steps inside enrichment_node
    (enrich_query's Chroma queries) and sql_node (schema_link's Chroma
    queries) were previously invisible in Langfuse - only the outer
    "enrichment"/"sql" node span showed up, with the retrieval collapsed
    into its opaque input/output. as_type="retriever" is the Langfuse
    observation type built for exactly this: showing what was retrieved
    (and from where) as a distinct step before the generation step that
    consumes it.

    Usage:
        with traced_span("schema_retrieval", as_type="retriever", input=question) as span:
            linked = schema_link(question)
            span.update(output=linked)
    """
    langfuse = get_langfuse()
    if langfuse is None:
        yield _NoOpSpan()
        return
    with langfuse.start_as_current_observation(name=name, as_type=as_type, input=input) as span:
        yield span
