from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from router import SupportRouter


def main() -> None:
    router = SupportRouter(PROJECT_ROOT)
    messages = [
        "What is the storage condition for acetone?",
        "Give me a quote for 250 kg ethanol shipped to California.",
        "What is your minimum order quantity?",
        "I need a human agent for a damaged shipment complaint.",
    ]

    for msg in messages:
        decision = router.route(msg)
        print("=" * 80)
        print(f"USER: {msg}")
        print(f"INTENT: {decision.intent}")
        print(f"TOOL: {decision.tool_used}")
        print(f"WHY: {decision.reason}")
        print(decision.response)


if __name__ == "__main__":
    main()
