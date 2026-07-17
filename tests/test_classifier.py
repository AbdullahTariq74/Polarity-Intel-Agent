import pytest
from pydantic import ValidationError

from models.result import ClassificationLabel, ClassifiedResult


def test_classification_label_values():
    assert ClassificationLabel.STRONG_MATCH.value == "strong_match"
    assert ClassificationLabel.WEAK_MATCH.value == "weak_match"
    assert ClassificationLabel.IRRELEVANT.value == "irrelevant"
    assert ClassificationLabel.AMBIGUOUS.value == "ambiguous"


def test_classification_label_from_raw_string():
    assert ClassificationLabel("strong_match") == ClassificationLabel.STRONG_MATCH


def test_classified_result_accepts_valid_payload():
    result = ClassifiedResult(
        url="https://example.com/fund",
        title="Example Family Office",
        raw_content="Example Family Office invests in Series B SaaS companies.",
        classification=ClassificationLabel.STRONG_MATCH,
        confidence_score=0.92,
        reasoning="Explicit sector and stage match found in content.",
        entity_name="Example Family Office",
        mandate_signals=["Series B", "SaaS"],
        contact_hints=[],
        requires_human_review=False
    )
    assert result.classification == ClassificationLabel.STRONG_MATCH
    assert result.confidence_score == pytest.approx(0.92)
    assert result.requires_human_review is False


def test_classified_result_rejects_out_of_range_confidence():
    with pytest.raises(ValidationError):
        ClassifiedResult(
            url="https://example.com",
            title="Bad Confidence",
            raw_content="content",
            classification=ClassificationLabel.WEAK_MATCH,
            confidence_score=1.5,
            reasoning="Confidence above 1.0 should be rejected."
        )


def test_classified_result_defaults():
    result = ClassifiedResult(
        url="https://example.com",
        title="Minimal Result",
        raw_content="content",
        classification=ClassificationLabel.IRRELEVANT,
        confidence_score=0.1,
        reasoning="Not a match."
    )
    assert result.mandate_signals == []
    assert result.contact_hints == []
    assert result.requires_human_review is False
