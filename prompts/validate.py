VALIDATE_PROMPT = """\
This result could not be classified automatically with sufficient confidence.

Entity: {entity_name}
URL: {url}
Content summary: {content_summary}
Agent reasoning: {reasoning}
Confidence: {confidence}

Please classify this result:
(1) Strong Match
(2) Weak Match
(3) Irrelevant
(4) Skip (exclude from report)

Enter 1, 2, 3, or 4:"""
