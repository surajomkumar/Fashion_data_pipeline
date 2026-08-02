"""
LLM-as-judge evaluation over eval_set.EVAL_CASES. For each case: run the
real agent (src.agent.lang_graph.run_agent), then have a second Bedrock
call - the judge - score the answer against the evidence the agent
actually retrieved (sql_result/vector_result), NOT against outside
knowledge of fashion trends. Two axes, per the "groundedness + relevance"
scope decided for this eval:

  groundedness (1-5): is every factual claim in the answer traceable to
  the evidence, with nothing invented? This is the axis that matters most
  for a schema-RAG pipeline where the LLM writes its own SQL - a fluent
  but ungrounded answer is a silent failure mode, not a loud one.

  relevance (1-5): does the answer actually address what was asked,
  independent of whether it's grounded? (An answer can be perfectly
  grounded in the evidence and still dodge the question.)

Scores are attached back to the ORIGINAL agent trace via
langfuse.create_score(trace_id=...) - run_agent() (lang_graph.py) already
returns that trace_id from inside its @observe-decorated call. The judge
call itself is also traced (as_type="evaluator") but as a separate trace,
since it's a distinct process evaluating the first, not a child of it.
"""
import json
import logging
import re

from src.agent.eval.eval_set import EVAL_CASES
from src.agent.lang_graph import run_agent
from src.agent.observability import get_langfuse, traced_bedrock_converse
from src.utils.aws_cred import call_cred

logger = logging.getLogger(__name__)

MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

JUDGE_PROMPT = """You are grading a fashion trend intelligence assistant's answer. Score it on two axes,
using ONLY the evidence provided below - do not use outside knowledge of fashion trends to judge factual
correctness, only whether the answer is consistent with (or contradicts/invents beyond) this evidence.

groundedness (1-5): 5 = every claim is directly supported by the evidence; 3 = mostly supported with a
minor unsupported detail; 1 = largely invented or contradicts the evidence.

relevance (1-5): 5 = directly and completely addresses the question asked; 3 = partially addresses it or
includes significant tangential content; 1 = does not address the question.

Question: {question}

Evidence (what the agent actually retrieved this turn):
{evidence}

Answer to grade:
{answer}

Respond with ONLY a JSON object, no markdown fences, no other text:
{{"groundedness": <1-5 int>, "groundedness_rationale": "<one sentence>",
  "relevance": <1-5 int>, "relevance_rationale": "<one sentence>"}}
"""


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in judge response: {text!r}")
    return json.loads(match.group(0))


def judge_answer(question: str, answer: str, sql_result, vector_result) -> dict:
    evidence = json.dumps({"sql_result": sql_result, "vector_result": vector_result}, default=str)
    prompt = JUDGE_PROMPT.format(question=question, evidence=evidence, answer=answer)

    langfuse = get_langfuse()
    client = call_cred()

    def _call():
        response = traced_bedrock_converse(
            client, MODEL_ID, [{"role": "user", "content": [{"text": prompt}]}], name="llm_judge",
        )
        return response["output"]["message"]["content"][0]["text"]

    if langfuse is None:
        raw = _call()
    else:
        with langfuse.start_as_current_observation(name="llm_judge_case", as_type="evaluator", input=question):
            raw = _call()

    return _extract_json(raw)


def run_eval(cases=EVAL_CASES) -> list:
    langfuse = get_langfuse()
    results = []

    for case in cases:
        agent_result = run_agent(case["question"], thread_id=f"eval-{case['id']}")
        try:
            scores = judge_answer(
                case["question"], agent_result["answer"],
                agent_result.get("sql_result"), agent_result.get("vector_result"),
            )
        except Exception as e:
            logger.warning(f"eval case {case['id']!r}: judge failed: {e}")
            scores = {"groundedness": None, "groundedness_rationale": str(e), "relevance": None, "relevance_rationale": str(e)}

        trace_id = agent_result.get("trace_id")
        if langfuse and trace_id:
            if scores["groundedness"] is not None:
                langfuse.create_score(
                    trace_id=trace_id, name="groundedness", value=scores["groundedness"],
                    data_type="NUMERIC", comment=scores["groundedness_rationale"],
                )
            if scores["relevance"] is not None:
                langfuse.create_score(
                    trace_id=trace_id, name="relevance", value=scores["relevance"],
                    data_type="NUMERIC", comment=scores["relevance_rationale"],
                )

        results.append({
            "id": case["id"], "question": case["question"], "intent_hint": case["intent_hint"],
            "actual_intent": agent_result.get("intent"), "answer": agent_result["answer"],
            "trace_id": trace_id, **scores,
        })

    if langfuse:
        langfuse.flush()

    return results


def print_report(results: list) -> None:
    grounded = [r["groundedness"] for r in results if r["groundedness"] is not None]
    relevant = [r["relevance"] for r in results if r["relevance"] is not None]

    print(f"\n{'ID':<24}{'Intent':<12}{'Grounded':<10}{'Relevant':<10}Trace")
    print("-" * 100)
    for r in results:
        trace_short = (r["trace_id"] or "")[:16]
        print(f"{r['id']:<24}{r['actual_intent'] or '?':<12}{str(r['groundedness']):<10}{str(r['relevance']):<10}{trace_short}")

    if grounded:
        print(f"\nAvg groundedness: {sum(grounded) / len(grounded):.2f} / 5  ({len(grounded)}/{len(results)} scored)")
    if relevant:
        print(f"Avg relevance:    {sum(relevant) / len(relevant):.2f} / 5  ({len(relevant)}/{len(results)} scored)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    results = run_eval()
    print_report(results)
