from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv


def pytest_configure(config):
    """Load real .env and override dummy values from parent conftest."""
    load_dotenv(override=True)


@pytest.fixture(autouse=True)
def skip_without_api_key(request):
    """Skip LLM integration tests when running with dummy credentials.

    Tests marked @pytest.mark.no_llm are fully deterministic and always run.
    """
    if request.node.get_closest_marker("no_llm"):
        return
    key = os.environ.get("LLM_API_KEY", "")
    if not key or key == "test-key":
        pytest.skip("LLM_API_KEY not set or is dummy value — skipping integration test")


@pytest.fixture()
def extractor():
    """Real Extractor backed by the LLM gateway."""
    # Re-import config after .env is loaded so it picks up real values
    import importlib
    import config
    importlib.reload(config)

    from extraction.extractor import Extractor
    return Extractor()


@pytest.fixture()
def structured_extractor():
    """StructuredExtractor instance — no API key required."""
    from extraction.extractor import SCHEMA_VERSION, PROMPT_VERSION
    from extraction.structured_extractor import StructuredExtractor
    return StructuredExtractor(schema_version=SCHEMA_VERSION, prompt_version=PROMPT_VERSION)


@pytest.fixture()
def entity_resolver():
    from resolution.entity_resolver import EntityResolver
    return EntityResolver()
