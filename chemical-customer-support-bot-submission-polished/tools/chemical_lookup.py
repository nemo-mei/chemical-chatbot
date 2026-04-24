from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import difflib
from typing import List, Optional

import pandas as pd


@dataclass
class ChemicalMatch:
    product_id: str
    product_name: str
    synonym: str
    cas_number: str
    purity: str
    hazard_class: str
    storage_condition: str
    application: str
    packaging_options: str
    unit: str
    base_price_usd_per_unit: float
    moq_value: float
    moq_unit: str
    lead_time_days: int
    hazmat_shipping: str
    notes: str
    score: float


class ChemicalLookupTool:
    """Search chemical master data and format customer-friendly answers.

    This tool is intentionally simple and deterministic so it is easy to debug.
    Later, you can wrap it as a LangChain Tool with very little extra work.
    """

    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        self._normalized_names = self.df.apply(self._row_search_text, axis=1).tolist()

    @staticmethod
    def _clean(text: str) -> str:
        return " ".join(str(text).lower().strip().split())

    def _row_search_text(self, row: pd.Series) -> str:
        fields = [
            row.get("product_name", ""),
            row.get("synonym", ""),
            row.get("cas_number", ""),
            row.get("formula", ""),
            row.get("application", ""),
            row.get("notes", ""),
        ]
        return self._clean(" ".join(str(x) for x in fields))

    def search(self, query: str, top_k: int = 3) -> List[ChemicalMatch]:
        query_clean = self._clean(query)
        scored = []

        for _, row in self.df.iterrows():
            haystack = self._row_search_text(row)

            # Hybrid score: exact containment + fuzzy ratio.
            contains_bonus = 0.25 if query_clean and query_clean in haystack else 0.0
            fuzzy = difflib.SequenceMatcher(None, query_clean, haystack).ratio()

            # Token overlap helps with partial names like "iso propyl" or "ethyl alcohol".
            query_tokens = set(query_clean.split())
            hay_tokens = set(haystack.split())
            overlap = len(query_tokens & hay_tokens) / max(len(query_tokens), 1)

            score = 0.45 * fuzzy + 0.30 * overlap + contains_bonus
            scored.append((score, row))

        scored.sort(key=lambda item: item[0], reverse=True)
        results: List[ChemicalMatch] = []
        for score, row in scored[:top_k]:
            results.append(
                ChemicalMatch(
                    product_id=str(row["product_id"]),
                    product_name=str(row["product_name"]),
                    synonym=str(row.get("synonym", "")),
                    cas_number=str(row.get("cas_number", "")),
                    purity=str(row.get("purity", "")),
                    hazard_class=str(row.get("hazard_class", "")),
                    storage_condition=str(row.get("storage_condition", "")),
                    application=str(row.get("application", "")),
                    packaging_options=str(row.get("packaging_options", "")),
                    unit=str(row.get("unit", "kg")),
                    base_price_usd_per_unit=float(row.get("base_price_usd_per_unit", 0.0)),
                    moq_value=float(row.get("moq_value", 0)),
                    moq_unit=str(row.get("moq_unit", "kg")),
                    lead_time_days=int(row.get("lead_time_days", 0)),
                    hazmat_shipping=str(row.get("hazmat_shipping", "No")),
                    notes=str(row.get("notes", "")),
                    score=round(float(score), 3),
                )
            )
        return results

    def best_match(self, query: str, min_score: float = 0.18) -> Optional[ChemicalMatch]:
        matches = self.search(query, top_k=1)
        if not matches:
            return None
        return matches[0] if matches[0].score >= min_score else None

    def format_match(self, match: ChemicalMatch) -> str:
        return (
            f"Product: {match.product_name}\n"
            f"Synonym: {match.synonym}\n"
            f"CAS: {match.cas_number}\n"
            f"Purity: {match.purity}\n"
            f"Hazard class: {match.hazard_class}\n"
            f"Storage: {match.storage_condition}\n"
            f"Applications: {match.application}\n"
            f"Packaging: {match.packaging_options}\n"
            f"Base price: ${match.base_price_usd_per_unit:.2f}/{match.unit}\n"
            f"MOQ: {match.moq_value:g} {match.moq_unit}\n"
            f"Lead time: {match.lead_time_days} days\n"
            f"Hazmat shipping: {match.hazmat_shipping}\n"
            f"Notes: {match.notes}"
        )

    def answer(self, query: str) -> str:
        match = self.best_match(query)
        if not match:
            return (
                "I could not find a confident match in the current catalog. "
                "Please check the product name, CAS number, or escalate to sales."
            )
        return self.format_match(match)


if __name__ == "__main__":
    tool = ChemicalLookupTool(Path(__file__).resolve().parents[1] / "data" / "chemicals.csv")
    print(tool.answer("ethyl alcohol"))
