SYNTHESIZE_PROMPT = """\
Mandate: {mandate}

Strong matches ({strong_count}):
{strong_matches}

Weak matches ({weak_count}):
{weak_matches}

Produce a private markets intelligence synthesis covering:
1. What was found vs. what was sought
2. Highest-value leads and the specific signals that qualify them
3. Gaps in the intelligence gathered
4. Concrete recommended next actions for an analyst

Respond with valid JSON only:
{{
  "synthesis": "Your narrative intelligence synthesis",
  "recommended_actions": ["action1", "action2", "action3"]
}}"""
