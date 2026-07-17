DECOMPOSE_SYSTEM = (
    "You are a specialized private markets research analyst. "
    "You decompose investor mandates into precise, targeted search queries "
    "that surface real investor entities - not generic market information."
)

DECOMPOSE_PROMPT = """\
Investor mandate: {mandate}

Generate exactly {max_queries} targeted web search queries to find real investors, \
family offices, or institutional allocators that match this mandate.

Rules:
- Each query must target real named entities, not general information
- Include fund names, AUM ranges, geography, and sector keywords where relevant
- Queries must be specific enough to produce actionable intelligence

Respond with valid JSON only:
{{
  "queries": ["query1", "query2", ...],
  "reasoning": "Why these queries will surface the right entities"
}}"""
