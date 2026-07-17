CLASSIFY_PROMPT = """\
Mandate: {mandate}

Search result:
Title: {title}
URL: {url}
Content: {content}

Classify this result against the mandate using exactly one label:
- strong_match: Clear entity match with explicit mandate signals
- weak_match: Partial match - some signals but incomplete or indirect
- irrelevant: Does not match - wrong type, sector, geography, or stage
- ambiguous: Insufficient information to determine - requires human review

Respond with valid JSON only:
{{
  "classification": "strong_match|weak_match|irrelevant|ambiguous",
  "confidence_score": 0.0-1.0,
  "reasoning": "Explicit reasoning for this classification",
  "entity_name": "Investor or entity name if identifiable, else null",
  "mandate_signals": ["signal1", "signal2"],
  "contact_hints": ["any contact info found"],
  "requires_human_review": true/false
}}"""
