from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.chemical_lookup import ChemicalLookupTool
from tools.quote_generator import QuoteGenerator
from tools.faq_rag import FAQKnowledgeBase


def main() -> None:
    chemical_tool = ChemicalLookupTool(PROJECT_ROOT / "data" / "chemicals.csv")
    print("=" * 80)
    print("CHEMICAL LOOKUP TEST")
    print(chemical_tool.answer("ethyl alcohol"))

    quote_tool = QuoteGenerator(
        PROJECT_ROOT / "data" / "chemicals.csv",
        PROJECT_ROOT / "data" / "pricing_rules.csv",
    )
    print("\n" + "=" * 80)
    print("QUOTE GENERATOR TEST")
    quote = quote_tool.generate_quote(
        product_query="acetone",
        quantity=250,
        region="California",
        expedited=True,
        rural=False,
        refrigerated=False,
    )
    print(quote_tool.format_quote(quote))

    faq_tool = FAQKnowledgeBase(PROJECT_ROOT / "docs")
    print("\n" + "=" * 80)
    print("FAQ / RAG TEST")
    print(faq_tool.answer("What is the minimum order quantity and do you ship hazardous materials?"))


if __name__ == "__main__":
    main()
