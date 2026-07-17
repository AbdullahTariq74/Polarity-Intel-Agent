from agent.graph import _route_after_classification
from models.result import ClassificationLabel, ClassifiedResult


def _dummy_result(classification: ClassificationLabel) -> ClassifiedResult:
    return ClassifiedResult(
        url="https://example.com",
        title="Dummy",
        raw_content="content",
        classification=classification,
        confidence_score=0.5,
        reasoning="test fixture"
    )


def test_routes_to_validate_when_ambiguous_items_pending():
    state = {
        "human_validation_required": True,
        "ambiguous_items_pending": [_dummy_result(ClassificationLabel.AMBIGUOUS)]
    }
    assert _route_after_classification(state) == "validate"


def test_routes_to_synthesize_when_no_ambiguous_items():
    state = {
        "human_validation_required": False,
        "ambiguous_items_pending": []
    }
    assert _route_after_classification(state) == "synthesize"


def test_routes_to_synthesize_when_flag_set_but_list_empty():
    """Guards against a stale True flag with no items actually pending."""
    state = {
        "human_validation_required": True,
        "ambiguous_items_pending": []
    }
    assert _route_after_classification(state) == "synthesize"
