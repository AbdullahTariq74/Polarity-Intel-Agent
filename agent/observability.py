import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


class AgentObserver:
    """
    Structured observability layer.
    Every agent decision, tool call, failure, and ceiling event is logged
    as a structured JSON record - making the system fully diagnosable
    without manual inspection of running state.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.decisions: List[dict] = []

        self._logger = logging.getLogger(f"polarity.{session_id}")
        if not self._logger.handlers:
            handler = logging.FileHandler(LOG_DIR / f"{session_id}.jsonl")
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.DEBUG)

    def _emit(self, event_type: str, payload: dict) -> dict:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "event": event_type,
            **payload
        }
        self.decisions.append(entry)
        self._logger.info(json.dumps(entry))
        return entry

    def log_decision(self, node: str, decision: str, reasoning: str,
                      metadata: Optional[dict] = None) -> None:
        self._emit("decision", {
            "node": node,
            "decision": decision,
            "reasoning": reasoning,
            "metadata": metadata or {}
        })

    def log_tool_call(self, tool_name: str, inputs: dict,
                       result_summary: str, success: bool) -> None:
        self._emit("tool_call", {
            "tool": tool_name,
            "inputs": inputs,
            "result_summary": result_summary,
            "success": success
        })

    def log_failure(self, node: str, error: str, recovery_action: str) -> None:
        self._emit("failure", {
            "node": node,
            "error": error,
            "recovery_action": recovery_action
        })

    def log_ceiling_hit(self, ceiling_type: str, current: Any, limit: Any) -> None:
        self._emit("ceiling_hit", {
            "ceiling_type": ceiling_type,
            "current_value": current,
            "limit": limit
        })

    def get_trace(self) -> List[dict]:
        return list(self.decisions)
