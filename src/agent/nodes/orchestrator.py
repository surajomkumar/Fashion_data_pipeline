"""
Orchestrator node - the supervisor hub. Re-entered after every other node
(see lang_graph.py: enrichment/sql/vector all edge straight back here),
deciding one action at a time by inspecting what's already in state
rather than holding its own separate control flow. Assumes guardrails
already passed - it's only ever reached via the graph's post-guardrail
edge, so it doesn't re-check that here.

Decision sequence, driven entirely by which keys are already present in
state (each one only gets set once per turn):

  1. "memory_hit" not in state   -> first call this turn. Check
     conversation_history for an exact repeat of this question; if found,
     short-circuit straight to the cached answer (next_step="end") without
     touching enrichment/sql/vector/synthesizer at all. Otherwise fall
     through to enrichment.
  2. "plan" not in state         -> enrichment has now run (intent/
     entities are set). Re-check memory once more against the resolved
     question in case an equivalent-but-differently-phrased repeat exists
     now that intent/entities are known [not yet implemented - see
     check_memory's docstring], then build the plan from intent
     (PLAN_BY_INTENT) and dispatch its first step.
  3. otherwise                   -> a plan step just returned. Pop it and
     dispatch the next one, or finalize if the plan is exhausted.

Memory is intentionally simple (per the "simple conversation memory,
keep last 3, nothing more" requirement): an exact, case/whitespace-
normalized repeat of a question already in conversation_history. It does
NOT do coreference resolution for follow-ups like "what about X" - that
history is still handed to enrichment/synthesizer as context for such
follow-ups, this check only ever short-circuits a literal repeat.

Step 1/2 check "memory_hit"/"plan" against the UNSET sentinels below
rather than Python's "key not in state" - with the same thread_id reused
across turns (MemorySaver), every field the checkpointer restores is
already "in" state from the previous turn, so presence alone can't tell
"fresh turn" from "mid-turn". lang_graph.py's reset_node writes these
exact sentinels at the start of every invoke for this reason - see its
docstring for the full explanation of the leak this fixes.
"""
from src.agent.nodes.state import AgentState

UNSET = "__unset__"

PLAN_BY_INTENT = {
    "structured": ["sql"],
    "explanatory": ["sql"],
    "semantic": ["vector"],
    "strategic": ["sql", "vector"],
}


def _normalize(question: str) -> str:
    return " ".join(question.strip().lower().split())


def check_memory(question: str, conversation_history: list) -> str | None:
    """Returns a cached answer if this exact question was already asked in the last 3 turns."""
    q = _normalize(question)
    for turn in conversation_history or []:
        if _normalize(turn.get("question", "")) == q:
            return turn.get("answer")
    return None


def build_plan(intent: str) -> list:
    return list(PLAN_BY_INTENT.get(intent, ["sql"]))


def _finalize_step(intent: str) -> str:
    return "formatter" if intent == "structured" else "synthesizer"


def orchestrator_node(state: AgentState) -> dict:
    conversation_history = state.get("conversation_history", [])

    # 1. First call this turn: memory gate before spending anything on enrichment/tools.
    if state.get("memory_hit", UNSET) == UNSET:
        hit = check_memory(state["question"], conversation_history)
        if hit is not None:
            return {"memory_hit": hit, "answer": hit, "next_step": "end"}
        return {"memory_hit": None, "next_step": "enrichment"}

    # 2. Enrichment has just run (intent/entities now set) - build the plan.
    if state.get("plan") is None:
        plan = build_plan(state["intent"])
        if not plan:
            return {"plan": [], "next_step": _finalize_step(state["intent"])}
        return {"plan": plan, "next_step": plan[0]}

    # 3. A plan step just completed - advance or finalize.
    remaining = state["plan"][1:]
    if remaining:
        return {"plan": remaining, "next_step": remaining[0]}
    return {"plan": [], "next_step": _finalize_step(state["intent"])}
