"""Builds the agent: the brain (Gemini) + the tools + the instructions for
when to use which tool."""

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver

from config import GOOGLE_API_KEY
from tools.web_search import search
from tools.calculator import calculator
from tools.pubmed_rag import pubmed_search

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,  # low: answers should stick closely to the retrieved evidence
    google_api_key=GOOGLE_API_KEY,
)

tools = [search, calculator, pubmed_search]

SYSTEM_PROMPT = """
You are a helpful AI assistant with access to three tools:

1. calculator - use for mathematical calculations.
2. search - use for general, current, or real-time information
   (news, weather, prices, etc.).
3. pubmed_search - use for scientific, medical, biological, or clinical
   research questions. It searches PubMed and returns the most relevant
   excerpts from real peer-reviewed papers, each with a PMID and link.

For scientific or research-related questions, call BOTH pubmed_search and
search, then compare what they return before answering:

- pubmed_search gives peer-reviewed evidence - treat it as the primary,
  more authoritative source for scientific claims.
- search gives general web results - useful for recent context, plain-
  language framing, or information that PubMed's abstracts don't cover
  (e.g. very recent news about a study, or practical/consumer angles).
- If the two sources agree, say so briefly and answer with confidence.
- If they conflict, point out the disagreement explicitly rather than
  picking one silently, and explain why (e.g. differing study populations,
  outdated web content, PubMed reflecting older research than a recent
  news article).
- If only one source has anything relevant, just use that one - don't
  force a comparison that doesn't exist.
- If pubmed_search returns no relevant papers, don't just say "no papers
  were found" and stop - fall back to search and answer from that instead,
  while noting that PubMed specifically had no matching peer-reviewed papers.

Always cite pubmed_search sources by PMID or link. For non-scientific
questions, just use search or calculator as needed - no need to involve
pubmed_search or force a comparison.
"""

# The checkpointer is what gives the agent memory - it saves the running
# conversation under a "thread_id" (set in main.py) so each new message is
# understood in context, instead of being treated as a fresh conversation.
checkpointer = InMemorySaver()

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
)