from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
import uuid

import gradio as gr

from langchain_bot import LangChainChemicalSupportBot


PROJECT_ROOT = Path(__file__).resolve().parent
BOT = LangChainChemicalSupportBot(PROJECT_ROOT)
CONVERSATION_LOG = PROJECT_ROOT / "logs" / "conversations.csv"
EXPECTED_COLUMNS = [
    "timestamp_utc",
    "session_id",
    "mode",
    "user_message",
    "detected_intent",
    "tool_used",
    "routing_reason",
    "bot_response",
]


def ensure_conversation_log() -> None:
    CONVERSATION_LOG.parent.mkdir(parents=True, exist_ok=True)
    needs_rewrite = True
    if CONVERSATION_LOG.exists():
        with CONVERSATION_LOG.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        if rows and rows[0] == EXPECTED_COLUMNS:
            needs_rewrite = False
        else:
            preserved = rows[1:] if rows else []
            with CONVERSATION_LOG.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(EXPECTED_COLUMNS)
                for row in preserved:
                    padded = list(row[: len(EXPECTED_COLUMNS)]) + [""] * max(0, len(EXPECTED_COLUMNS) - len(row))
                    writer.writerow(padded[: len(EXPECTED_COLUMNS)])
            needs_rewrite = False
    if needs_rewrite:
        with CONVERSATION_LOG.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(EXPECTED_COLUMNS)


def log_turn(session_id: str, mode: str, user_message: str, detected_intent: str, tool_used: str, routing_reason: str, bot_response: str) -> None:
    ensure_conversation_log()
    with CONVERSATION_LOG.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            session_id,
            mode,
            user_message,
            detected_intent,
            tool_used,
            routing_reason,
            bot_response,
        ])


def respond(message: str, history: list[dict], session_id: str):
    result = BOT.respond(message, history=history)
    log_turn(
        session_id=session_id,
        mode=result.mode,
        user_message=message,
        detected_intent=result.intent,
        tool_used=result.tool_used,
        routing_reason=result.reason,
        bot_response=result.response,
    )
    return result.response


def build_demo() -> gr.Blocks:
    ensure_conversation_log()
    mode_badge = "LangChain + Gemini" if BOT.has_live_llm else "Fallback deterministic router"
    kb_badge = getattr(BOT, "kb_mode", "keyword_fallback")
    with gr.Blocks(title="Chemical Customer Support Bot (LangChain)") as demo:
        session_state = gr.State(str(uuid.uuid4()))
        gr.Markdown(
            "# Chemical Customer Support Bot
"
            f"**Run mode:** {mode_badge}
"
            f"**FAQ engine:** {kb_badge}

"
            "This version wraps the support modules as LangChain tools and uses a Gemini model when an API key is configured. "
            "The FAQ module now supports real RAG with markdown chunking, Gemini embeddings, and Chroma persistence when dependencies and API keys are available. "
            "If the optional stack is not available, it automatically falls back to deterministic local retrieval so the demo still runs."
        )
        gr.ChatInterface(
            fn=respond,
            additional_inputs=[session_state],
            examples=[
                "What is the storage condition for acetone?",
                "Give me a quote for 250 kg ethanol shipped to California.",
                "What is your minimum order quantity?",
                "I need a human representative for a damaged shipment complaint.",
            ],
        )
    return demo


demo = build_demo()


if __name__ == "__main__":
    demo.launch()
