"""General web search tool - for current events, news, weather, prices, etc."""

from langchain.tools import tool
from langchain_community.utilities import SerpAPIWrapper
from config import SERPAPI_API_KEY

_serpapi = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)


@tool
def search(query: str) -> str:
    """Search the internet for current events, news, or general real-time
    information. Do NOT use this for scientific or medical research questions -
    use pubmed_search for those instead."""
    return _serpapi.run(query)