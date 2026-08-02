"""
10-question eval set for llm_judge.py, spanning the same 4 intent
categories orchestrator.PLAN_BY_INTENT routes on (structured/
explanatory/semantic/strategic), in the spirit of the 27-question bank
from the design discussion (social-only/competitor-only/self-only/
cross-source/explanatory/semantic/strategic categories) - not a verbatim
reproduction of it (that list lived in chat, not in a file, so this
redraws representative cases against the real data actually in
fashion_intel.db/chroma_db rather than guessing at exact original
wording). Style tags/categories used below are pulled from
knowledge/style_tags.json and verified present in the data pipeline
output, so every case is answerable, not hypothetical.
"""

EVAL_CASES = [
    {
        "id": "structured-self",
        "question": "What's the average sell-through rate for oversized-hoodie style?",
        "intent_hint": "structured",
    },
    {
        "id": "structured-competitor",
        "question": "What's the current price range for cargo-pant across competitors?",
        "intent_hint": "structured",
    },
    {
        "id": "structured-cross-source",
        "question": "Compare momentum for oversized-hoodie between our own catalog and competitors.",
        "intent_hint": "structured",
    },
    {
        "id": "explanatory-decline",
        "question": "Why is the cropped-zip-up style declining?",
        "intent_hint": "explanatory",
    },
    {
        "id": "explanatory-rise",
        "question": "Why is engagement rising for graphic-print on Instagram?",
        "intent_hint": "explanatory",
    },
    {
        "id": "semantic-aesthetic",
        "question": "What styles feel Y2K-adjacent right now?",
        "intent_hint": "semantic",
    },
    {
        "id": "semantic-similar",
        "question": "Show me posts similar to oversized-hoodie that haven't been tagged that way.",
        "intent_hint": "semantic",
    },
    {
        "id": "strategic-launch",
        "question": "Should we launch a new style similar to oversized-hoodie given competitor activity?",
        "intent_hint": "strategic",
    },
    {
        "id": "strategic-whitespace",
        "question": "Is there whitespace in techwear-shell that competitors haven't filled yet?",
        "intent_hint": "strategic",
    },
    {
        "id": "structured-oos",
        "question": "Which competitor products for baggy are currently out of stock?",
        "intent_hint": "structured",
    },
]
