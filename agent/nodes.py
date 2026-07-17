import json
from typing import Any, Dict, Optional

from anthropic import Anthropic

from agent.memory import log_decision
from agent.observability import AgentObserver
from agent.state import AgentState
from agent.tools import web_search
from prompts.decompose import DECOMPOSE_PROMPT, DECOMPOSE_SYSTEM

CLAUDE_MODEL = "claude-sonnet-4-6"

_client: Optional[Anthropic] = None


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic()
    return _client


def _extract_json(text: str) -> Dict[str, Any]:
    """Extract a JSON object from a model response, tolerating code fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    return json.loads(text)


def decompose_mandate(state: AgentState) -> AgentState:
    """
    Node 1: Decompose the investor mandate into targeted search queries.
    The model decides how to break the mandate down, bounded by max_search_iterations.
    On failure, falls back to two generic queries so the graph keeps moving.
    """
    observer = AgentObserver(state["session_id"])
    mandate = state["mandate"]
    max_queries = min(mandate.max_search_iterations, 5)

    try:
        client = _get_client()
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1000,
            system=DECOMPOSE_SYSTEM,
            messages=[{
                "role": "user",
                "content": DECOMPOSE_PROMPT.format(mandate=mandate.raw_input, max_queries=max_queries)
            }]
        )
        data = _extract_json(response.content[0].text)
        queries = data.get("queries", [])
        reasoning = data.get("reasoning", "")

        observer.log_decision(
            node="decompose_mandate",
            decision=f"Generated {len(queries)} search queries",
            reasoning=reasoning,
            metadata={"queries": queries}
        )
        log_decision(state["session_id"], "decompose_mandate", f"Generated {len(queries)} queries", reasoning)

        return {
            **state,
            "search_queries": queries,
            "decomposition_reasoning": reasoning,
            "decision_trace": state["decision_trace"] + observer.get_trace()
        }
    except Exception as exc:
        fallback_queries = [
            f"{mandate.raw_input} family office investor",
            f"{mandate.raw_input} institutional LP investment mandate"
        ]
        observer.log_failure("decompose_mandate", str(exc), "Using fallback generic queries")
        return {
            **state,
            "search_queries": fallback_queries,
            "decomposition_reasoning": f"Fallback due to error: {exc}",
            "errors": state["errors"] + [str(exc)],
            "decision_trace": state["decision_trace"] + observer.get_trace()
        }


def execute_searches(state: AgentState) -> AgentState:
    """
    Node 2: Execute a web search for each decomposed query.
    Enforces the mandate's cost ceiling - once max_search_iterations is
    reached, remaining queries are skipped and cost_ceiling_hit is set.
    """
    observer = AgentObserver(state["session_id"])
    mandate = state["mandate"]
    all_results = []
    iterations = 0
    ceiling_hit = False

    for query in state["search_queries"]:
        if iterations >= mandate.max_search_iterations:
            observer.log_ceiling_hit("search_iterations", iterations, mandate.max_search_iterations)
            ceiling_hit = True
            break

        results = web_search(query, max_results=mandate.max_results_per_query)
        for result in results:
            result["source_query"] = query
        all_results.extend(results)
        iterations += 1

        observer.log_tool_call(
            tool_name="web_search",
            inputs={"query": query, "max_results": mandate.max_results_per_query},
            result_summary=f"Returned {len(results)} results",
            success=len(results) > 0
        )

    return {
        **state,
        "raw_results": all_results,
        "iterations_used": iterations,
        "cost_ceiling_hit": ceiling_hit,
        "decision_trace": state["decision_trace"] + observer.get_trace()
    }
