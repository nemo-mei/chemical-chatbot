from __future__ import annotations

import os
from pathlib import Path

from langchain_bot import LangChainChemicalSupportBot


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def clear_keys() -> None:
    for key in ["GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_GENAI_MODEL"]:
        os.environ.pop(key, None)


def test_fallback_mode_without_api_key() -> None:
    clear_keys()
    bot = LangChainChemicalSupportBot(PROJECT_ROOT)
    assert bot.has_live_llm is False
    result = bot.respond("What is the storage condition for acetone?", history=[])
    assert result.mode == "fallback"
    assert result.intent == "CHEMICAL_LOOKUP"
    assert "Acetone" in result.response


def test_fallback_quote_request() -> None:
    clear_keys()
    bot = LangChainChemicalSupportBot(PROJECT_ROOT)
    result = bot.respond("Give me a quote for 250 kg ethanol shipped to California.", history=[])
    assert result.intent in {"QUOTE_REQUEST", "ESCALATION"}
    assert "Quote status:" in result.response
