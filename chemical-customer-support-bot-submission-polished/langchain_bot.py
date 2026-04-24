from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from router import RouteDecision, SupportRouter
from tools.chemical_lookup import ChemicalLookupTool
from tools.escalation import EscalationTool
from tools.faq_rag import FAQKnowledgeBase
from tools.quote_generator import QuoteGenerator

try:
    from langchain.agents import create_agent
    from langchain.tools import tool
    from langchain_google_genai import ChatGoogleGenerativeAI
    LANGCHAIN_AVAILABLE = True
except Exception:  # pragma: no cover - graceful fallback for environments without LangChain
    create_agent = None
    ChatGoogleGenerativeAI = None
    LANGCHAIN_AVAILABLE = False

    def tool(*args, **kwargs):  # type: ignore
        def decorator(func):
            return func
        return decorator


SYSTEM_PROMPT = """You are a customer support assistant for a chemical products supplier.

Your job is to help customers by using tools instead of guessing.

Use these rules:
1. For product properties, CAS numbers, purity, hazard class, storage, packaging, or catalog details, use chemical_lookup.
2. For pricing, quotations, shipping destinations, quantities, bulk orders, or freight adjustments, use generate_quote.
3. For company policies such as payment terms, minimum order quantity, returns, hazardous shipping, lead times, or standard business FAQs, use answer_faq.
4. For complaints, damaged shipments, SDS or COA requests, custom formulations, regulatory exceptions, legal threats, or any request the system cannot confidently resolve, use escalate_request.
5. Do not invent product data, pricing, or policy details. If the answer is unclear, escalate.
6. Keep answers concise, professional, and customer-facing.
"""


@dataclass
class BotResponse:
    response: str
    intent: str
    tool_used: str
    reason: str
    mode: str


class ChemicalSupportServices:
    """Shared service container for the bot's internal data tools."""

    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root)
        self.chemical_tool = ChemicalLookupTool(self.project_root / "data" / "chemicals.csv")
        self.quote_tool = QuoteGenerator(
            self.project_root / "data" / "chemicals.csv",
            self.project_root / "data" / "pricing_rules.csv",
        )
        self.faq_tool = FAQKnowledgeBase(
            self.project_root / "docs",
            persist_dir=self.project_root / "vectorstore" / "faq_kb",
            prefer_rag=True,
        )
        self.escalation_tool = EscalationTool(self.project_root / "logs" / "escalations.csv")
        self.rule_router = SupportRouter(self.project_root)


class LangChainChemicalSupportBot:
    """LangChain-based bot with automatic fallback to the deterministic router.

    When GOOGLE_API_KEY or GEMINI_API_KEY is available and LangChain packages are
    installed, the bot uses a Gemini chat model plus LangChain tools. Otherwise,
    it falls back to the existing rule-based router so the app still runs.
    """

    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root)
        self.services = ChemicalSupportServices(self.project_root)
        self.mode = "fallback"
        self.kb_mode = self.services.faq_tool.mode if hasattr(self, 'services') else 'keyword_fallback'
        self.agent = None
        self.tool_names = {
            "chemical_lookup": "CHEMICAL_LOOKUP",
            "generate_quote": "QUOTE_REQUEST",
            "answer_faq": "FAQ",
            "escalate_request": "ESCALATION",
        }
        self._build_agent_if_possible()
        self.kb_mode = self.services.faq_tool.mode

    @property
    def has_live_llm(self) -> bool:
        return self.agent is not None

    def _build_agent_if_possible(self) -> None:
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not (LANGCHAIN_AVAILABLE and api_key):
            return

        services = self.services

        @tool
        def chemical_lookup(query: str) -> str:
            """Look up a chemical product by product name, alias, or CAS number. Use this for properties, storage, purity, formula, hazard class, or catalog questions."""
            return services.chemical_tool.answer(query)

        @tool
        def generate_quote(
            product_query: str,
            quantity: float,
            region: str,
            expedited: bool = False,
            refrigerated: bool = False,
            rural: bool = False,
        ) -> str:
            """Generate a preliminary quote for a chemical order. Use for pricing, quotation, cost, quantity, shipping destination, freight, or bulk-order requests."""
            quote = services.quote_tool.generate_quote(
                product_query=product_query,
                quantity=quantity,
                region=region,
                expedited=expedited,
                refrigerated=refrigerated,
                rural=rural,
            )
            formatted = services.quote_tool.format_quote(quote)
            if quote.status == "ESCALATE":
                record = services.escalation_tool.create(
                    customer_message=f"Quote escalation for {product_query} / {quantity:g} / {region}",
                    detected_reason=quote.note,
                    priority="High",
                )
                return formatted + "\n\n" + services.escalation_tool.format_confirmation(record)
            return formatted

        @tool
        def answer_faq(question: str) -> str:
            """Answer internal policy and FAQ questions using the knowledge base. Use for MOQ, payment terms, lead times, shipping policy, returns, or standard company policy questions."""
            return services.faq_tool.answer(question)

        @tool
        def escalate_request(customer_message: str, reason: str = "Needs human review") -> str:
            """Escalate a request to a human representative. Use for complaints, damaged shipments, SDS/COA requests, custom formulations, regulatory exceptions, or unresolved cases."""
            lowered = customer_message.lower()
            priority = "High" if any(k in lowered for k in ["urgent", "damaged", "complaint", "lawsuit"]) else "Medium"
            record = services.escalation_tool.create(
                customer_message=customer_message,
                detected_reason=reason,
                priority=priority,
            )
            return services.escalation_tool.format_confirmation(record)

        model_name = os.getenv("GOOGLE_GENAI_MODEL", "gemini-2.5-flash")
        model = ChatGoogleGenerativeAI(model=model_name, temperature=0.1)
        self.agent = create_agent(model=model, tools=[chemical_lookup, generate_quote, answer_faq, escalate_request], system_prompt=SYSTEM_PROMPT)
        self.mode = "langchain"

    def _extract_final_text(self, result: dict[str, Any]) -> str:
        messages = result.get("messages", [])
        for message in reversed(messages):
            content = getattr(message, "content", None)
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(content, list):
                text_blocks: list[str] = []
                for block in content:
                    if isinstance(block, dict):
                        text = block.get("text") or block.get("content")
                        if isinstance(text, str) and text.strip():
                            text_blocks.append(text.strip())
                if text_blocks:
                    return "\n".join(text_blocks)
        return "I could not generate a response. Please escalate this request to a human representative."

    def _extract_tool_trace(self, result: dict[str, Any]) -> tuple[str, str]:
        messages = result.get("messages", [])
        used_tools: list[str] = []
        for message in messages:
            if message.__class__.__name__.lower() == "toolmessage":
                name = getattr(message, "name", None)
                if isinstance(name, str) and name:
                    used_tools.append(name)
                continue

            content = getattr(message, "content", None)
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_call":
                        name = block.get("name")
                        if isinstance(name, str) and name:
                            used_tools.append(name)

            additional_kwargs = getattr(message, "additional_kwargs", {}) or {}
            tool_calls = additional_kwargs.get("tool_calls", [])
            for call in tool_calls:
                if isinstance(call, dict):
                    name = call.get("name") or call.get("function", {}).get("name")
                    if isinstance(name, str) and name:
                        used_tools.append(name)

        if not used_tools:
            return "FAQ", "No tool call surfaced by the model; treating reply as FAQ/general response."

        tool_chain = " -> ".join(used_tools)
        final_intent = self.tool_names.get(used_tools[-1], "FAQ")
        return final_intent, f"LangChain agent selected tool path: {tool_chain}"

    def _respond_with_agent(self, message: str, history: list[dict[str, str]] | None = None) -> BotResponse:
        assert self.agent is not None
        agent_messages: list[dict[str, str]] = []
        for turn in history or []:
            role = turn.get("role")
            content = turn.get("content")
            if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
                agent_messages.append({"role": role, "content": content})
        agent_messages.append({"role": "user", "content": message})

        result = self.agent.invoke({"messages": agent_messages})
        final_text = self._extract_final_text(result)
        intent, reason = self._extract_tool_trace(result)
        tool_used = reason.replace("LangChain agent selected tool path: ", "") if "selected tool path" in reason else "ModelOnly"
        return BotResponse(
            response=final_text,
            intent=intent,
            tool_used=tool_used,
            reason=reason,
            mode=self.mode,
        )

    def _respond_with_fallback(self, message: str) -> BotResponse:
        decision: RouteDecision = self.services.rule_router.route(message)
        return BotResponse(
            response=decision.response,
            intent=decision.intent,
            tool_used=decision.tool_used,
            reason=decision.reason,
            mode="fallback",
        )

    def respond(self, message: str, history: list[dict[str, str]] | None = None) -> BotResponse:
        if self.agent is not None:
            try:
                return self._respond_with_agent(message, history=history)
            except Exception as exc:  # pragma: no cover - network/runtime fallback
                fallback = self._respond_with_fallback(message)
                fallback.reason = f"LangChain runtime error; used fallback router instead: {exc}"
                return fallback
        return self._respond_with_fallback(message)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    bot = LangChainChemicalSupportBot(root)
    prompts = [
        "What is the hazard class of methanol?",
        "Give me a quote for 250 kg ethanol shipped to California.",
        "What is your minimum order quantity?",
        "I need a human representative for a damaged shipment complaint.",
    ]
    history: list[dict[str, str]] = []
    print(f"Mode: {bot.mode}")
    for prompt in prompts:
        result = bot.respond(prompt, history=history)
        print("=" * 80)
        print("USER:", prompt)
        print("INTENT:", result.intent)
        print("TOOL:", result.tool_used)
        print("REASON:", result.reason)
        print("BOT:", result.response)
        history.extend([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": result.response},
        ])
