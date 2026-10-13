"""
Central place for loading environment variables / API keys.
Every other file imports from here instead of calling load_dotenv() itself.
"""

import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# NCBI (PubMed) doesn't require a key for light use, but asks you to identify
# yourself with an email, and gives you a higher rate limit if you get a free key.
# Set your own in .env - no default here, so no personal address is committed.
NCBI_EMAIL = os.getenv("NCBI_EMAIL")
NCBI_API_KEY = os.getenv("NCBI_API_KEY")  # optional