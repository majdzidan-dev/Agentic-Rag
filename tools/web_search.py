"""General web search tool - for current events, news, weather, prices, etc."""

from langchain.tools import tool
from langchain_community.utilities import SerpAPIWrapper
from config import SERPAPI_API_KEY

# Created on first use, not at import - so a missing SERPAPI_API_KEY only
# disables this one tool instead of stopping the whole agent from starting.
_serpapi = None


@tool
def search(query: str) -> str:
    """Search the internet for current events, news, or general real-time
    information. For scientific or medical research questions, pubmed_search
    is the primary source - use this only alongside it, for recent context or
    plain-language framing that peer-reviewed abstracts don't cover."""
    global _serpapi
    try:
        if _serpapi is None:
            _serpapi = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)
        return _serpapi.run(query)
    except Exception as e:
        # Only the error TYPE is reported - full messages can include the
        # request URL, which contains the SerpAPI key.
        print(f"[search] failed: {type(e).__name__}")
        return f"Web search failed ({type(e).__name__}), so no web results are available right now."
