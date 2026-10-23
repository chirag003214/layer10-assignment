import os
import json
from dotenv import load_dotenv

load_dotenv()

LLM_API_KEY = os.environ["LLM_API_KEY"]
LLM_API_BASE = os.environ["LLM_API_BASE"]
LLM_MODEL = os.environ.get("LLM_MODEL", "anthropic/claude-haiku-4-5-20251001")
LLM_LOGGING_ENABLED = os.environ.get("LLM_LOGGING_ENABLED", "true").lower() == "true"

EXTRA_HEADERS = {
    "X-LLM-METADATA": "{}",
    "X-LLM-LOGGING-CONFIG": json.dumps({"enabled": LLM_LOGGING_ENABLED}),
}

LLM_KWARGS = {
    "api_key": LLM_API_KEY,
    "api_base": LLM_API_BASE,
    "model": LLM_MODEL,
    "custom_llm_provider": "openai",
    "extra_headers": EXTRA_HEADERS,
}

# Embedding configuration for freetext deduplication
USE_LOCAL_EMBEDDINGS = os.environ.get("USE_LOCAL_EMBEDDINGS", "true").lower() == "true"
# Local embedding model (used by default)
EMBEDDING_LOCAL_MODEL = os.environ.get("EMBEDDING_LOCAL_MODEL", "all-MiniLM-L6-v2")
# Remote embedding model (LiteLLM/OpenAI, only used if USE_LOCAL_EMBEDDINGS=false)
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
