"""Shared helper for extracting clean text from LLM responses.

Gemini sometimes returns its response as a plain string, and sometimes as a
list of structured content blocks (each a dict with 'type', 'text', and
internal metadata like a tool-call verification signature). This normalizes
either case into plain text.
"""


def extract_text(content) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)

    return str(content)