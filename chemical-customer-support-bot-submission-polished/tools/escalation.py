from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import csv


@dataclass
class EscalationRecord:
    timestamp_utc: str
    customer_message: str
    detected_reason: str
    priority: str
    status: str = "OPEN"


class EscalationTool:
    """Append unresolved requests to the escalation log."""

    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp_utc",
                    "customer_message",
                    "detected_reason",
                    "priority",
                    "status",
                ])

    def create(self, customer_message: str, detected_reason: str, priority: str = "Medium") -> EscalationRecord:
        record = EscalationRecord(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            customer_message=customer_message.strip(),
            detected_reason=detected_reason.strip(),
            priority=priority,
        )
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                record.timestamp_utc,
                record.customer_message,
                record.detected_reason,
                record.priority,
                record.status,
            ])
        return record

    @staticmethod
    def format_confirmation(record: EscalationRecord) -> str:
        return (
            "I have routed this request to a human representative.\n"
            f"Priority: {record.priority}\n"
            f"Reason: {record.detected_reason}\n"
            "A sales or support agent should review the request in the escalation queue."
        )
