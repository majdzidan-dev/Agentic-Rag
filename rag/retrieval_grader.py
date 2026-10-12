"""
The "thinking" half of the agentic retrieval loop.

Two small, focused LLM calls:
1. grade_sufficiency() - judges whether retrieved excerpts actually answer
   the original question, or fall short.
2. rewrite_query() - if they fall short, generates a better search query
   (e.g. more precise scientific terminology) to retry with.

Both use temperature=0 - this is a judgment/formatting task, not creative
writing, so we want consistent, predictable output rather than variety.
"""

import json

from langchain_google_genai import ChatGoogleGenerativeAI

from config import GOOGLE_API_KEY
from llm_utils import extract_text

_grader_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=GOOGLE_API_KEY,
)


def _clean_json_response(text: str) -> str:
    """Models often wrap JSON in markdown code fences even when told not
    to - strip those off before parsing."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text.removeprefix("json").strip()
    return text.strip()


def grade_sufficiency(query: str, excerpts: str) -> dict:
    """Returns {"sufficient": bool, "reason": str}."""
    prompt = f"""You are grading whether retrieved research excerpts are
enough to answer a question. Respond with ONLY a JSON object - no markdown
fences, no other text.

Question: {query}

Retrieved excerpts:
{excerpts}

Respond with exactly this JSON shape:
{{"sufficient": true or false, "reason": "one short sentence explaining why"}}"""

    response = _grader_llm.invoke(prompt)
    text = _clean_json_response(extract_text(response.content))

    try:
        result = json.loads(text)
        return {
            "sufficient": bool(result.get("sufficient", True)),
            "reason": str(result.get("reason", "")),
        }
    except (json.JSONDecodeError, AttributeError):
        # If grading itself fails to parse, default to "sufficient" so a
        # formatting hiccup can't trap the loop in endless retries.
        return {"sufficient": True, "reason": "Grading response could not be parsed."}


def rewrite_query(original_query: str, reason: str) -> str:
    """Returns a rewritten search query string."""
    prompt = f"""The following PubMed search query did not retrieve good
enough results. Rewrite it as a better, more specific PubMed search query -
consider precise scientific/medical terminology instead of lay language.
Respond with ONLY the rewritten query text - no quotes, no explanation.

Original query: {original_query}
Why it fell short: {reason}"""

    response = _grader_llm.invoke(prompt)
    return extract_text(response.content).strip().strip('"')