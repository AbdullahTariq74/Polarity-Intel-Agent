# PolarityIQ Private Markets Intelligence Agent

An autonomous research agent for private markets intelligence. Give it an investor
mandate in plain English and it decomposes the mandate into targeted research
queries, searches the web for real matching investors, classifies every result
it finds against the mandate with explicit reasoning, escalates genuinely
ambiguous cases to a human reviewer, and produces a structured intelligence
report an analyst can act on directly.

## Architecture

The agent is built as a directed graph of single-responsibility nodes, wired
together with LangGraph. State flows through the graph explicitly - no node
mutates hidden global state, and every transition is either a fixed edge or an
explicit routing decision.

```
                 ┌──────────────┐
  mandate  ───▶  │  decompose   │  Claude breaks the mandate into
   (text)        │  (Claude)    │  targeted search queries
                 └──────┬───────┘
                        ▼
                 ┌──────────────┐
                 │    search    │  Tavily web search per query,
                 │   (Tavily)   │  bounded by the cost ceiling
                 └──────┬───────┘
                        ▼
                 ┌──────────────┐
                 │   classify   │  Claude labels every result:
                 │  (Claude)    │  strong / weak / irrelevant / ambiguous
                 └──────┬───────┘
                        │
             ambiguous? │
              ┌─────────┴─────────┐
              ▼                   ▼
       ┌─────────────┐      ┌──────────────┐
       │  validate   │      │  synthesize  │
       │  (human)    │─────▶│  (Claude)    │
       └─────────────┘      └──────┬───────┘
                                    ▼
                          IntelligenceReport
```

Every node receives the full `AgentState` and returns an updated copy of it.
Routing after classification is an explicit function
(`_route_after_classification` in `agent/graph.py`), not an implicit side
effect - if any result was classified `ambiguous`, the graph routes to the
human checkpoint before synthesis; otherwise it goes straight to synthesis.

## Why this is an agentic system, not a prompt chain

- **Bounded autonomy.** The model chooses what to search for and how to
  classify each result, but only within limits the mandate defines
  (`max_search_iterations`, `max_results_per_query`) and within a fixed label
  set (`strong_match` / `weak_match` / `irrelevant` / `ambiguous`). It never
  has unconstrained freedom to act.
- **Ambiguity is surfaced, not guessed away.** Any result the model can't
  classify with confidence is routed to a human checkpoint rather than forced
  into one of the other three buckets.
- **Failure handling is designed in, not bolted on.** Every LLM call and tool
  call is wrapped in error handling with a defined fallback: decomposition
  falls back to generic queries, search failures return an empty result set,
  unparsable classifications are skipped and logged, and synthesis falls back
  to a partial report rather than crashing the run.
- **Full observability.** Every decision, tool call, failure, and ceiling
  event is logged as a structured JSON record via `AgentObserver`, so any run
  can be reconstructed and audited after the fact without re-running it.
- **State persists across steps.** Session metadata, individual search
  results, and per-node decisions are written to SQLite as the graph runs, so
  intermediate state survives process restarts.

## Project layout

```
polarity-intel-agent/
├── agent/
│   ├── graph.py          # LangGraph wiring and routing
│   ├── nodes.py          # The five agent nodes
│   ├── tools.py          # Tavily web search with retry/backoff
│   ├── state.py          # AgentState TypedDict
│   ├── memory.py         # SQLite persistence layer
│   └── observability.py  # Structured JSON decision logging
├── models/
│   ├── mandate.py        # InvestorMandate (input)
│   ├── result.py         # ClassifiedResult
│   └── report.py         # IntelligenceReport (output)
├── prompts/               # Prompt templates per node
├── app.py                 # Streamlit UI
├── main.py                # CLI entry point
└── tests/
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env          # then fill in your API keys
```

You'll need:

1. **Anthropic API key** - https://console.anthropic.com
2. **Tavily API key** - https://tavily.com (free tier available)

## Usage

**CLI:**

```bash
python main.py "family office investing in B2B SaaS Series B, North America"
python main.py "PE fund targeting mid-market industrials in the DACH region" --max-iterations 3 --max-results 5
```

**Streamlit UI:**

```bash
streamlit run app.py
```

The UI exposes the same cost/action ceilings as sliders, shows live progress
while the agent runs, and renders the full decision trace as expandable JSON
alongside the report.

## Observability

Every session gets its own JSON-lines log file at `logs/<session_id>.jsonl`.
Each line is a structured event: a `decision` (what the agent chose and why),
a `tool_call` (what was called, with what inputs, and whether it succeeded),
a `failure` (what broke and how the system recovered), or a `ceiling_hit`
(which limit was reached). The same trace is also attached to the final
`IntelligenceReport` under `agent_decisions_log`, so a report is self-contained
evidence of how it was produced.

## Cost and action ceilings

Two ceilings bound the agent's autonomy, both set on the `InvestorMandate`:

- **`max_search_iterations`** (cost ceiling) - the maximum number of search
  queries the agent will execute in one run. Once reached, the search node
  stops issuing new queries, logs a `ceiling_hit` event, and marks
  `cost_ceiling_hit=True` on the resulting report so it's clear the results
  may be partial.
- **`max_results_per_query`** (action ceiling) - the maximum number of results
  requested per individual search, keeping the classification workload (and
  token spend) proportional to what was actually asked for.

## Design decisions and tradeoffs

**SQLite over a hosted database.** This agent runs single-session, local-first
workloads - a full database server would add operational weight with no real
benefit at this scale. SQLite gives durable state across steps with zero
infrastructure.

**Bounded label set for classification.** Rather than letting the model
produce free-form categories, classification is constrained to four fixed
labels. This makes downstream logic (routing, reporting, ceilings) simple to
reason about and keeps the model's autonomy scoped to *which* label applies,
not *what* labels exist.

**Graceful degradation over exceptions.** Every external call (Claude, Tavily)
can fail, and none of those failures should take down a multi-minute agent
run. Each node is written to catch its own failures, log them, and continue
with a sane fallback rather than letting an exception propagate to the graph
level.

**Human checkpoint as a graph node, not a side channel.** Ambiguous items are
handled by routing to a dedicated `validate` node rather than special-casing
them outside the graph. That keeps the control flow explicit and visible in
`agent/graph.py` rather than hidden in application code.
