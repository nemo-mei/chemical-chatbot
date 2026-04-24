from __future__ import annotations

from pathlib import Path

from tools.faq_rag import FAQKnowledgeBase


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_faq_fallback_answers_moq() -> None:
    kb = FAQKnowledgeBase(
        PROJECT_ROOT / "docs",
        persist_dir=PROJECT_ROOT / "vectorstore" / "test_faq_kb",
        prefer_rag=False,
    )
    result = kb.answer("What is the minimum order quantity?")
    assert "minimum order quantity" in result.lower() or "moq" in result.lower()
    assert kb.mode == "keyword_fallback"


def test_faq_diagnostics_shape() -> None:
    kb = FAQKnowledgeBase(
        PROJECT_ROOT / "docs",
        persist_dir=PROJECT_ROOT / "vectorstore" / "test_faq_kb",
        prefer_rag=False,
    )
    diagnostics = kb.diagnostics()
    assert diagnostics["mode"] == "keyword_fallback"
    assert diagnostics["doc_count"] > 0
