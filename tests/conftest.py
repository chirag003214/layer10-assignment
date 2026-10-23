import os

# Provide dummy env vars so config.py can be imported without a real .env file
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("LLM_API_BASE", "http://test.example.com")
