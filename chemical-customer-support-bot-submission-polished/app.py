from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import csv
import uuid

import gradio as gr

from router import SupportRouter


PROJECT_ROOT = Path(__file__).resolve().parent
ROUTER = SupportRouter(PROJECT_ROOT)
CONVERSATION_LOG = PROJECT_ROOT / "logs" / "conversations.csv"
EXPECTED_COLUMNS = [
    "timestamp_utc",
    "session_id",
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


def log_turn(session_id: str, user_message: str, detected_intent: str, tool_used: str, routing_reason: str, bot_response: str) -> None:
    ensure_conversation_log()
    with CONVERSATION_LOG.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            session_id,
            user_message,
            detected_intent,
            tool_used,
            routing_reason,
            bot_response,
        ])


def respond(message: str, history: list[dict], session_id: str):
    decision = ROUTER.route(message)
    log_turn(
        session_id=session_id,
        user_message=message,
        detected_intent=decision.intent,
        tool_used=decision.tool_used,
        routing_reason=decision.reason,
        bot_response=decision.response,
    )
    return decision.response


def build_demo() -> gr.Blocks:
    ensure_conversation_log()
    with gr.Blocks(title="Chemical Customer Support Bot") as demo:
        session_state = gr.State(str(uuid.uuid4()))
        gr.Markdown(
            "# Chemical Customer Support Bot
"
            "This demo routes customer requests to catalog lookup, quote generation, FAQ retrieval, or escalation."
        )
        gr.ChatInterface(
            fn=respond,
            additional_inputs=[session_state],
        )
    return demo


demo = build_demo()


if __name__ == "__main__":
    demo.launch()
