"""
Formatter node - finalizes the "structured" intent path only
(orchestrator._finalize_step returns "formatter" solely when
intent == "structured").

Only the single-aggregate case stays fully deterministic, no LLM
involved ("AVG(price): 51.5625" is already a complete answer - a
one-cell, one-row result needs no narrative). Same for the error/empty
cases - there's nothing to synthesize from a query that failed or
returned nothing.

Multi-row results delegate entirely to synthesizer.synthesize_narrative -
the same Key Insights/Suggested Actions bullet treatment explanatory/
semantic/strategic answers get. This used to be a separate deterministic
table-dump (see git history), then a table + one summary sentence; both
turned out to undersell genuinely rich multi-row results (e.g. a
cross-source comparison spanning Instagram engagement, competitor
ratings, and self sell-through, each on a wildly different scale) that
deserve the same insight/action framing as an explanatory question, not
just a formatted table. The row-level detail isn't lost - it's still in
sql_result.rows for the UI's Data table (app.py); this only changes what
prose goes in the Answer box.

Also appends this turn to conversation_history - the formatter/
synthesizer nodes are the last stop before END (orchestrator does not
get re-entered after them), so whichever one runs is responsible for
writing the turn that keep_last_3 will fold in for future turns. (For
the multi-row case this happens inside synthesize_narrative itself.)
"""
from src.agent.nodes.state import AgentState
from src.agent.nodes.synthesizer import synthesize_narrative


def format_structured(state: AgentState) -> dict:
    sql_result = state.get("sql_result") or {}
    error = sql_result.get("error")
    rows = sql_result.get("rows", [])

    if error:
        answer = f"I couldn't answer that - the generated query failed: {error}"
    elif not rows:
        answer = "No matching data found for that question."
    elif len(rows) == 1 and len(rows[0]) == 1:
        # a single aggregate value, e.g. {"AVG(price)": 51.5625}
        key, value = next(iter(rows[0].items()))
        answer = f"{key}: {value}"
    else:
        return synthesize_narrative(state)

    turn = {"question": state["question"], "answer": answer, "intent": state.get("intent")}
    return {"answer": answer, "conversation_history": [turn]}
