import argparse
import uuid

from dotenv import load_dotenv
from rich.console import Console

from agent.graph import agent_graph
from agent.memory import init_db, save_session, update_session_status
from models.mandate import InvestorMandate

load_dotenv()
console = Console()


def run_agent(mandate_text: str, max_iterations: int = 5, max_results: int = 8) -> dict:
    """Run the Private Markets Intelligence Agent end-to-end for a single mandate."""
    init_db()
    session_id = str(uuid.uuid4())[:8]

    mandate = InvestorMandate(
        raw_input=mandate_text,
        max_search_iterations=max_iterations,
        max_results_per_query=max_results
    )
    save_session(session_id, mandate_text)

    initial_state = {
        "mandate": mandate,
        "session_id": session_id,
        "search_queries": [],
        "decomposition_reasoning": "",
        "raw_results": [],
        "classified_results": [],
        "iterations_used": 0,
        "cost_ceiling_hit": False,
        "ambiguous_items_pending": [],
        "human_validation_required": False,
        "report": None,
        "decision_trace": [],
        "errors": [],
        "messages": []
    }

    console.print(f"\n[bold]Starting session:[/bold] {session_id}")
    console.print(f"[bold]Mandate:[/bold] {mandate_text}\n")

    final_state = agent_graph.invoke(initial_state)
    update_session_status(session_id, "completed")

    report = final_state.get("report")
    if report:
        console.print("\n[bold underline]INTELLIGENCE REPORT[/bold underline]")
        console.print(f"Strong Matches: {len(report.strong_matches)}")
        console.print(f"Weak Matches: {len(report.weak_matches)}")
        console.print(f"Flagged for Review: {len(report.flagged_for_review)}")
        console.print(f"\n[bold]Synthesis:[/bold]\n{report.synthesis}")

        if report.recommended_actions:
            console.print("\n[bold]Recommended Actions:[/bold]")
            for action in report.recommended_actions:
                console.print(f"  - {action}")

        if report.cost_ceiling_hit:
            console.print("\n[yellow]Cost ceiling was hit - results may be partial.[/yellow]")

    if final_state.get("errors"):
        console.print(f"\n[red]Errors encountered:[/red] {final_state['errors']}")

    return final_state


def main() -> None:
    parser = argparse.ArgumentParser(description="PolarityIQ Private Markets Intelligence Agent")
    parser.add_argument("mandate", type=str, help="Investor mandate in natural language")
    parser.add_argument("--max-iterations", type=int, default=5, help="Cost ceiling: max search iterations")
    parser.add_argument("--max-results", type=int, default=8, help="Action ceiling: max results per query")
    args = parser.parse_args()

    run_agent(args.mandate, max_iterations=args.max_iterations, max_results=args.max_results)


if __name__ == "__main__":
    main()
