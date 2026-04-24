from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from tools.chemical_lookup import ChemicalLookupTool, ChemicalMatch


@dataclass
class QuoteResult:
    product_name: str
    region: str
    quantity: float
    unit: str
    unit_price: float
    product_subtotal: float
    shipping_fee: float
    hazmat_surcharge: float
    refrigerated_surcharge: float
    rural_surcharge: float
    expedite_fee: float
    discount_rate: float
    discount_amount: float
    total_price: float
    estimated_lead_time_days: int
    status: str
    note: str


class QuoteGenerator:
    """Generate preliminary quotes from catalog and pricing rules.

    This is intended for demo / portfolio use only, not final commercial pricing.
    """

    def __init__(self, chemicals_csv: str | Path, pricing_csv: str | Path):
        self.lookup_tool = ChemicalLookupTool(chemicals_csv)
        self.pricing_df = pd.read_csv(pricing_csv)

    @staticmethod
    def _clean(text: str) -> str:
        return " ".join(str(text).lower().strip().split())

    def _find_region_row(self, region: str) -> Optional[pd.Series]:
        region_clean = self._clean(region)
        df = self.pricing_df.copy()
        df["_region_clean"] = df["region"].astype(str).map(self._clean)

        exact = df[df["_region_clean"] == region_clean]
        if not exact.empty:
            return exact.iloc[0]

        contains = df[df["_region_clean"].str.contains(region_clean, na=False)]
        if not contains.empty:
            return contains.iloc[0]

        return None

    def generate_quote(
        self,
        product_query: str,
        quantity: float,
        region: str,
        expedited: bool = False,
        refrigerated: bool = False,
        rural: bool = False,
    ) -> QuoteResult:
        chem: Optional[ChemicalMatch] = self.lookup_tool.best_match(product_query)
        if chem is None:
            return QuoteResult(
                product_name=product_query,
                region=region,
                quantity=quantity,
                unit="kg",
                unit_price=0.0,
                product_subtotal=0.0,
                shipping_fee=0.0,
                hazmat_surcharge=0.0,
                refrigerated_surcharge=0.0,
                rural_surcharge=0.0,
                expedite_fee=0.0,
                discount_rate=0.0,
                discount_amount=0.0,
                total_price=0.0,
                estimated_lead_time_days=0,
                status="ESCALATE",
                note="Product not found in catalog. Route to a human sales rep.",
            )

        region_row = self._find_region_row(region)
        if region_row is None:
            return QuoteResult(
                product_name=chem.product_name,
                region=region,
                quantity=quantity,
                unit=chem.unit,
                unit_price=chem.base_price_usd_per_unit,
                product_subtotal=0.0,
                shipping_fee=0.0,
                hazmat_surcharge=0.0,
                refrigerated_surcharge=0.0,
                rural_surcharge=0.0,
                expedite_fee=0.0,
                discount_rate=0.0,
                discount_amount=0.0,
                total_price=0.0,
                estimated_lead_time_days=chem.lead_time_days,
                status="ESCALATE",
                note="Region not found in pricing rules. Route to sales for a manual quote.",
            )

        status = "OK"
        note_parts = []
        if quantity < chem.moq_value:
            status = "BELOW_MOQ"
            note_parts.append(
                f"Requested quantity is below MOQ ({chem.moq_value:g} {chem.moq_unit}). Pricing is preliminary only."
            )

        unit_price = chem.base_price_usd_per_unit
        product_subtotal = unit_price * quantity
        shipping_fee = float(region_row["standard_shipping_fee_usd"])
        hazmat_surcharge = float(region_row["hazmat_surcharge_usd"]) if str(chem.hazmat_shipping).lower() == "yes" else 0.0
        refrigerated_surcharge = float(region_row["refrigerated_surcharge_usd"]) if refrigerated else 0.0
        rural_surcharge = float(region_row["rural_surcharge_usd"]) if rural else 0.0
        expedite_fee = float(region_row["expedite_fee_usd"]) if expedited else 0.0

        threshold = float(region_row["bulk_discount_threshold_qty"])
        discount_rate = float(region_row["bulk_discount_rate"]) if quantity >= threshold else 0.0
        discount_amount = product_subtotal * discount_rate

        total = (
            product_subtotal
            - discount_amount
            + shipping_fee
            + hazmat_surcharge
            + refrigerated_surcharge
            + rural_surcharge
            + expedite_fee
        )

        estimated_lead_time_days = chem.lead_time_days + (1 if refrigerated else 0)
        if expedited:
            estimated_lead_time_days = max(1, estimated_lead_time_days - 1)

        if str(chem.hazard_class).strip():
            note_parts.append(f"Hazard class: {chem.hazard_class}.")
        note_parts.append("This is a demo quote and not a binding commercial offer.")

        return QuoteResult(
            product_name=chem.product_name,
            region=str(region_row["region"]),
            quantity=quantity,
            unit=chem.unit,
            unit_price=unit_price,
            product_subtotal=round(product_subtotal, 2),
            shipping_fee=round(shipping_fee, 2),
            hazmat_surcharge=round(hazmat_surcharge, 2),
            refrigerated_surcharge=round(refrigerated_surcharge, 2),
            rural_surcharge=round(rural_surcharge, 2),
            expedite_fee=round(expedite_fee, 2),
            discount_rate=discount_rate,
            discount_amount=round(discount_amount, 2),
            total_price=round(total, 2),
            estimated_lead_time_days=int(estimated_lead_time_days),
            status=status,
            note=" ".join(note_parts),
        )

    @staticmethod
    def format_quote(result: QuoteResult) -> str:
        if result.status == "ESCALATE":
            return f"Quote status: {result.status}\nReason: {result.note}"

        return (
            f"Quote status: {result.status}\n"
            f"Product: {result.product_name}\n"
            f"Destination: {result.region}\n"
            f"Quantity: {result.quantity:g} {result.unit}\n"
            f"Unit price: ${result.unit_price:.2f}/{result.unit}\n"
            f"Product subtotal: ${result.product_subtotal:.2f}\n"
            f"Shipping fee: ${result.shipping_fee:.2f}\n"
            f"Hazmat surcharge: ${result.hazmat_surcharge:.2f}\n"
            f"Refrigerated surcharge: ${result.refrigerated_surcharge:.2f}\n"
            f"Rural surcharge: ${result.rural_surcharge:.2f}\n"
            f"Expedite fee: ${result.expedite_fee:.2f}\n"
            f"Discount: {result.discount_rate:.0%} (-${result.discount_amount:.2f})\n"
            f"Estimated total: ${result.total_price:.2f}\n"
            f"Estimated lead time: {result.estimated_lead_time_days} days\n"
            f"Note: {result.note}"
        )


if __name__ == "__main__":
    generator = QuoteGenerator(
        Path(__file__).resolve().parents[1] / "data" / "chemicals.csv",
        Path(__file__).resolve().parents[1] / "data" / "pricing_rules.csv",
    )
    quote = generator.generate_quote("ethanol", 250, "California", expedited=True)
    print(generator.format_quote(quote))
