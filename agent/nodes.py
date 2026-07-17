import json
from typing import Any, Dict, Optional

from anthropic import Anthropic

from agent.memory import log_decision, save_result
from agent.observability import AgentObserver
from agent.state import AgentState
from agent.tools import web_search
from models.result import ClassificationLabel, ClassifiedResult
from prompts.classify import CLASSIFY_PROMPT
from prompts.decompose import DECOMPOSE_PROMPT, DECOMPOSE_SYSTEM

CONTENT_CHAR_LIMIT = 2000

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


def classify_results(state: AgentState) -> AgentState:
    """
    Node 3: Classify each raw search result against the mandate.
    The model chooses exactly one of four bounded labels per result.
    AMBIGUOUS results are routed to ambiguous_items_pending for the
    human validation checkpoint rather than being guessed at.
    """
    observer = AgentObserver(state["session_id"])
    mandate = state["mandate"]
    client = _get_client()

    classified = []
    ambiguous = []

    for result in state["raw_results"]:
        try:
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=500,
                messages=[{
                    "role": "user",
                    "content": CLASSIFY_PROMPT.format(
                        mandate=mandate.raw_input,
                        title=result.get("title", ""),
                        url=result.get("url", ""),
                        content=result.get("content", "")[:CONTENT_CHAR_LIMIT]
                    )
                }]
            )
            data = _extract_json(response.content[0].text)

            classified_result = ClassifiedResult(
                url=result.get("url", ""),
                title=result.get("title", ""),
                raw_content=result.get("content", ""),
                classification=ClassificationLabel(data["classification"]),
                confidence_score=data["confidence_score"],
                reasoning=data["reasoning"],
                entity_name=data.get("entity_name"),
                mandate_signals=data.get("mandate_signals", []),
                contact_hints=data.get("contact_hints", []),
                requires_human_review=data.get("requires_human_review", False),
                source_query=result.get("source_query")
            )

            observer.log_decision(
                node="classify_results",
                decision=f"Classified as {classified_result.classification.value}",
                reasoning=classified_result.reasoning,
                metadata={"url": classified_result.url, "confidence": classified_result.confidence_score}
            )
            save_result(
                state["session_id"],
                classified_result.source_query or "",
                classified_result.url,
                classified_result.title,
                classified_result.classification.value,
                classified_result.confidence_score,
                classified_result.reasoning
            )

            if classified_result.classification == ClassificationLabel.AMBIGUOUS:
                ambiguous.append(classified_result)
            else:
                classified.append(classified_result)

        except Exception as exc:
            observer.log_failure("classify_results", str(exc), "Skipping unparsable result")
            continue

    return {
        **state,
        "classified_results": classified,
        "ambiguous_items_pending": ambiguous,
        "human_validation_required": len(ambiguous) > 0,
        "decision_trace": state["decision_trace"] + observer.get_trace()
    }
