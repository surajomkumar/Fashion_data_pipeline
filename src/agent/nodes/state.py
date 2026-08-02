"""
Shared state schema for the online query agent's LangGraph.

Two kinds of fields:
  - per-turn working state (question, intent, entities, results, answer) -
    meant to be fully overwritten each invocation. With the checkpointer
    reusing the same thread_id across turns, LangGraph's default channel
    otherwise just keeps whatever was last written - so lang_graph.py's
    reset_node explicitly clears every one of these at the start of each
    invoke (memory_hit/plan to the UNSET sentinel orchestrator.py checks
    for, the rest to None/{}) rather than relying on "not in state".
  - conversation_history - the one field that persists ACROSS invocations
    via the graph's checkpointer (MemorySaver, keyed by thread_id in
    lang_graph.py). keep_last_3() is its reducer: every merge appends the
    new turn(s) and truncates to the last 3, so history never grows past
    3 turns even within a single long session - older turns are dropped,
    not archived anywhere. reset_node does NOT touch this field.
"""
from typing import Annotated, Any, Dict, List, Optional, TypedDict


def keep_last_3(existing: list, new: list) -> list:
    return ((existing or []) + (new or []))[-3:]


class AgentState(TypedDict, total=False):
    # per-turn input/working state
    question: str
    guardrail_passed: bool
    guardrail_message: Optional[str]        # set if guardrails blocked the question

    knowledge_matches: Dict[str, dict]      # raw output of knowledge_enrichment (per type: style_tag/source/intent/momentum/aesthetic_concept)
    intent: str                             # "structured" | "explanatory" | "semantic" | "strategic"
    entities: Dict[str, Optional[str]]      # {"category", "style_tag", "source", "momentum"}
    enriched_question: str                  # raw question + resolved types' glossary descriptions appended - used by vector_node's embedding query, not sql/synthesizer

    memory_hit: Optional[str]               # short-circuit answer if resolved from conversation_history
    next_step: str                          # orchestrator's routing decision, read by the conditional edge
    plan: List[str]                         # remaining steps to execute, e.g. ["sql", "vector"]

    sql_result: Any
    vector_result: Any
    evidence: Dict[str, List[Dict]]

    answer: str

    # persisted across turns via the checkpointer
    conversation_history: Annotated[List[Dict], keep_last_3]
