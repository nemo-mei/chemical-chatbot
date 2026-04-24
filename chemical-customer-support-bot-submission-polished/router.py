from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Optional

import pandas as pd

from tools.chemical_lookup import ChemicalLookupTool
from tools.quote_generator import QuoteGenerator
from tools.faq_rag import FAQKnowledgeBase
from tools.escalation import EscalationTool


INTENTS = ["CHEMICAL_LOOKUP", "QUOTE_REQUEST", "FAQ", "ESCALATION"]


@dataclass
class RouteDecision:
    intent: str
    tool_used: str
    response: str
    reason: str


class SupportRouter:
    """Simple deterministic router for the chemical support bot.

    This starter version uses rule-based intent detection. It is easy to debug
    and can later be replaced with an LLM classifier while keeping the same
    external interface.
    """

    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root)
        self.chemical_tool = ChemicalLookupTool(self.project_root / "data" / "chemicals.csv")
        self.quote_tool = QuoteGenerator(
            self.project_root / "data" / "chemicals.csv",
            self.project_root / "data" / "pricing_rules.csv",
        )
        self.faq_tool = FAQKnowledgeBase(self.project_root / "docs")
        self.escalation_tool = EscalationTool(self.project_root / "logs" / "escalations.csv")
        pricing_df = pd.read_csv(self.project_root / "data" / "pricing_rules.csv")
        self.pricing_regions = pricing_df["region"].astype(str).tolist()
        chemical_df = pd.read_csv(self.project_root / "data" / "chemicals.csv")
        self.known_chemical_terms = []
        for _, row in chemical_df.iterrows():
            for field in [row.get("product_name", ""), row.get("synonym", ""), row.get("cas_number", "")]:
                value = str(field).strip()
                if value:
                    self.known_chemical_terms.append(value)

    @staticmethod
    def _clean(text: str) -> str:
        return " ".join(str(text).lower().strip().split())

    def detect_intent(self, message: str) -> tuple[str, str]:
        text = self._clean(message)

        escalation_keywords = [
            "human", "agent", "representative", "sales rep", "manager", "complaint",
            "lawsuit", "damaged", "urgent issue", "urgent", "special approval", "custom formulation",
            "coa", "certificate of analysis", "sds", "msds", "regulatory exception",
        ]
        quote_keywords = [
            "quote", "quotation", "price", "cost", "pricing", "how much", "shipped to",
            "ship to", "delivery to", "buy", "purchase",
        ]
        chemical_keywords = [
            "cas", "formula", "purity", "hazard", "storage", "property", "properties",
            "tell me about", "what is", "lookup", "catalog", "chemical",
        ]

        if any(k in text for k in escalation_keywords):
            return "ESCALATION", "Matched escalation keyword pattern."

        if "minimum order quantity" in text or text.startswith("what is your moq") or "payment terms" in text:
            return "FAQ", "Detected policy or commercial FAQ wording."

        if any(k in text for k in quote_keywords) and self._extract_quantity(text) is not None:
            return "QUOTE_REQUEST", "Detected pricing language plus a quantity."

        if any(k in text for k in quote_keywords):
            return "QUOTE_REQUEST", "Detected pricing or quotation language."

        best_match = self.chemical_tool.best_match(message, min_score=0.10)
        if any(k in text for k in chemical_keywords) or (best_match is not None and best_match.score >= 0.16):
            return "CHEMICAL_LOOKUP", "Detected chemical lookup language or a plausible product match."

        return "FAQ", "Defaulted to knowledge-base FAQ flow."

    @staticmethod
    def _extract_quantity(text: str) -> Optional[float]:
        match = re.search(r"(\d+(?:\.\d+)?)\s*(kg|kilograms|g|grams|l|liters|litres|tons|tonnes)?\b", text.lower())
        if not match:
            return None
        value = float(match.group(1))
        unit = (match.group(2) or "kg").lower()
        if unit in {"g", "grams"}:
            return value / 1000.0
        if unit in {"tons", "tonnes"}:
            return value * 1000.0
        return value

    def _extract_region(self, text: str) -> str:
        clean = self._clean(text)
        for region in self.pricing_regions:
            region_clean = self._clean(region)
            if region_clean in clean:
                return region
        patterns = [
            r"shipped to ([a-zA-Z\s]+)",
            r"ship to ([a-zA-Z\s]+)",
            r"delivery to ([a-zA-Z\s]+)",
            r"to ([a-zA-Z\s]+)$",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip(" .,!?")
        return "California"

    def _extract_product_query(self, text: str) -> str:
        lowered = text.lower()

        term_matches = []
        for term in self.known_chemical_terms:
            term_clean = term.lower().strip()
            if term_clean and term_clean in lowered:
                term_matches.append(term)
        if term_matches:
            return max(term_matches, key=len)

        patterns = [
            r"quote for (.+?) shipped to",
            r"quote for (.+?) to",
            r"price for (.+?) shipped to",
            r"price for (.+?) to",
            r"cost of (.+?) shipped to",
            r"cost of (.+?) to",
            r"buy (.+?) shipped to",
        ]
        for pattern in patterns:
            match = re.search(pattern, lowered)
            if match:
                candidate = match.group(1)
                candidate = re.sub(r"\b\d+(?:\.\d+)?\s*(kg|kilograms|g|grams|l|liters|litres|tons|tonnes)?\b", "", candidate)
                return candidate.strip(" ,.")

        strong_match = self.chemical_tool.best_match(text)
        if strong_match:
            return strong_match.product_name
        return text

    def route(self, message: str) -> RouteDecision:
        intent, reason = self.detect_intent(message)

        if intent == "CHEMICAL_LOOKUP":
            product_query = self._extract_product_query(message)
            best_match = self.chemical_tool.best_match(product_query, min_score=0.10)
            if best_match is not None:
                response = self.chemical_tool.format_match(best_match)
            else:
                response = self.chemical_tool.answer(product_query)
            return RouteDecision(intent=intent, tool_used="ChemicalLookupTool", response=response, reason=reason)

        if intent == "QUOTE_REQUEST":
            quantity = self._extract_quantity(message) or 25.0
            region = self._extract_region(message)
            product_query = self._extract_product_query(message)
            expedited = "expedite" in message.lower() or "urgent" in message.lower() or "rush" in message.lower()
            refrigerated = "refrigerated" in message.lower() or "cold chain" in message.lower()
            rural = "rural" in message.lower() or "remote" in message.lower()
            quote = self.quote_tool.generate_quote(
                product_query=product_query,
                quantity=quantity,
                region=region,
                expedited=expedited,
                refrigerated=refrigerated,
                rural=rural,
            )
            response = self.quote_tool.format_quote(quote)
            if quote.status == "ESCALATE":
                record = self.escalation_tool.create(message, detected_reason=quote.note, priority="High")
                response = response + "\n\n" + self.escalation_tool.format_confirmation(record)
                return RouteDecision(intent="ESCALATION", tool_used="QuoteGenerator -> EscalationTool", response=response, reason=quote.note)
            return RouteDecision(intent=intent, tool_used="QuoteGenerator", response=response, reason=reason)

        if intent == "ESCALATION":
            priority = "High" if any(k in message.lower() for k in ["urgent", "damaged", "complaint"]) else "Medium"
            record = self.escalation_tool.create(message, detected_reason=reason, priority=priority)
            response = self.escalation_tool.format_confirmation(record)
            return RouteDecision(intent=intent, tool_used="EscalationTool", response=response, reason=reason)

        response = self.faq_tool.answer(message)
        return RouteDecision(intent="FAQ", tool_used="FAQKnowledgeBase", response=response, reason=reason)


if __name__ == "__main__":
    router = SupportRouter(Path(__file__).resolve().parent)
    demo_messages = [
        "What is the CAS number and storage condition for acetone?",
        "Give me a quote for 250 kg ethanol shipped to California.",
        "Do you offer hazardous materials shipping to Canada?",
        "I need a human agent for a damaged shipment complaint.",
    ]
    for msg in demo_messages:
        decision = router.route(msg)
        print("=" * 80)
        print(f"USER: {msg}")
        print(f"INTENT: {decision.intent} | TOOL: {decision.tool_used}")
        print(decision.response)
