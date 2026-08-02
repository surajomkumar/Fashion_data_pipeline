"""
Synthesizer node - the one LLM call that produces a narrative answer,
grounded ONLY in what sql_node/vector_node actually retrieved this turn.
Finalizes the "explanatory", "semantic", and "strategic" paths
(orchestrator._finalize_step returns "synthesizer" for anything that
isn't "structured").

Also appends this turn to conversation_history - see formatter.py's
docstring, same reasoning applies here.

This is now also the narrative path formatter.py delegates to for
multi-row structured results (see its docstring), so the "don't invent
claims the evidence doesn't support" rule below has to cover more ground
than it originally did. The demographic/audience callout was added after
a real case: a question phrased "...among youths" got answered with
"dominant category for youth fashion" - the question's own wording
(not the data) leaking into the answer as if it were a finding. No
demographic column exists anywhere in this schema, so any audience claim
is always unsupported here.
"""
import json

from src.agent.nodes.state import AgentState
from src.agent.observability import traced_bedrock_converse
from src.utils.aws_cred import call_cred

MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

SYSTEM_PROMPT = """You are a fashion trend intelligence assistant. Answer the user's question using ONLY the
structured data and/or semantic search results provided below - do not use outside knowledge about
fashion trends. Cite specific style_tags, SKUs, brands, or post_ids from the data where relevant.

Format the response as markdown with exactly these two sections, in this order:

**Key Insights**
- 2-4 bullet points, each stating one thing the data shows, grounded in a specific figure or fact from
  the evidence. If the evidence doesn't explain something the question asks about (e.g. a causal "why"),
  say so plainly in a bullet here rather than guessing.

**Suggested Actions**
- 1-3 bullet points of concrete next steps that follow from the insights above. If the evidence doesn't
  support a specific recommendation, say so in one bullet ("No specific action is supported by this
  data alone") rather than inventing one.

Do not speculate beyond what the evidence supports. Do not add audience, demographic, or marketing
language (e.g. "among Gen Z", "popular with young shoppers", "a must-have") unless a demographic or
audience column is literally present in the evidence - a style_tag or category name is not evidence of
who buys it, and neither is a word the question itself used ("...among youths" in the question is not
proof the data says anything about youths)."""


def _build_context(state: AgentState) -> dict:
    context = {"question": state["question"], "entities": state.get("entities")}

    sql_result = state.get("sql_result")
    if sql_result:
        context["sql_data"] = {"row_count": sql_result.get("row_count"), "rows": sql_result.get("rows", [])[:30]}

    vector_result = state.get("vector_result")
    if vector_result:
        context["semantic_matches"] = [
            {"style_tag": h["metadata"].get("style_tag"), "source": h["metadata"].get("source"),
             "distance": h["distance"], "text": h["text"]}
            for h in vector_result
        ]

    return context


def synthesize_narrative(state: AgentState) -> dict:
    context = _build_context(state)
    prompt = f"{SYSTEM_PROMPT}\n\nData:\n{json.dumps(context, indent=2, default=str)}\n\nQuestion: {state['question']}"

    client = call_cred()
    response = traced_bedrock_converse(
        client, MODEL_ID, [{"role": "user", "content": [{"text": prompt}]}], name="synthesis",
    )
    answer = response["output"]["message"]["content"][0]["text"]

    turn = {"question": state["question"], "answer": answer, "intent": state.get("intent")}
    return {"answer": answer, "conversation_history": [turn]}
