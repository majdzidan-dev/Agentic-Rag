"""Calculator tool - uses numexpr instead of eval() for safety."""

from langchain.tools import tool
import numexpr


@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression, e.g. '12 * 30' or '(5 + 3) / 2'."""
    try:
        result = numexpr.evaluate(expression).item()
        return str(result)
    except Exception as e:
        return f"Could not calculate expression: {e}"