import pytest

from app.grounding.citations import CitationValidator
from app.grounding.models import Evidence, GeneratedAnswer, REFUSAL_ANSWER
from app.retrieval.models import RetrievalCandidate


def _evidence(citation_id: str, *, content: str | None = None) -> Evidence:
    candidate = RetrievalCandidate(
        content=f"Canonical content for {citation_id}" if content is None else content,
        document_id=f"trusted-document-{citation_id}",
        chunk_id=f"trusted-chunk-{citation_id}",
        source_url=f"https://trusted.example/{citation_id}",
        metadata={"nested": {"id": citation_id}},
    )
    return Evidence(
        citation_id=citation_id,
        candidate=candidate,
        prompt_content="model-visible bounded content",
    )


def test_validator_maps_known_ids_only_from_trusted_evidence():
    result = CitationValidator(excerpt_max_chars=200).validate(
        GeneratedAnswer(answer="Supported answer [E1].", citation_ids=["E1"]),
        [_evidence("E1")],
    )

    assert result.answer == "Supported answer [E1]."
    assert result.status == "supported"
    assert [citation.model_dump() for citation in result.citations] == [{
        "citation_id": "E1",
        "document_id": "trusted-document-E1",
        "chunk_id": "trusted-chunk-E1",
        "source_url": "https://trusted.example/E1",
        "excerpt": "Canonical content for E1",
    }]


def test_validator_rejects_repeated_citation_ids():
    evidence = [_evidence("E1"), _evidence("E2")]
    result = CitationValidator().validate(
        GeneratedAnswer(answer="answer [E2] [E1].", citation_ids=["E2", "E1", "E2", "E1"]),
        evidence,
    )

    assert result.answer == REFUSAL_ANSWER
    assert result.citations == []
    assert result.status == "validation_failed"


@pytest.mark.parametrize(
    "citation_ids",
    [
        ["E999"],
        ["https://evil.example/fake"],
        ["E1", "E999"],
    ],
)
def test_validator_removes_invented_ids_and_marks_validation_failed(citation_ids):
    result = CitationValidator().validate(
        GeneratedAnswer(answer="answer [E1].", citation_ids=citation_ids),
        [_evidence("E1")],
    )

    assert result.answer == REFUSAL_ANSWER
    assert result.citations == []
    assert result.status == "validation_failed"
    assert all(citation.source_url.startswith("https://trusted.example/") for citation in result.citations)


def test_validator_rejects_url_like_answer_text_and_fails_closed():
    result = CitationValidator().validate(
        GeneratedAnswer(
            answer="Read https://model-invented.example for details.",
            citation_ids=[],
        ),
        [_evidence("E1")],
    )

    assert result.answer == REFUSAL_ANSWER
    assert result.citations == []
    assert result.status == "validation_failed"


def test_validator_rejects_detached_or_partially_cited_claims_and_fails_closed():
    evidence = [_evidence("E1")]

    detached = CitationValidator().validate(
        GeneratedAnswer(answer="Supported answer.", citation_ids=["E1"]),
        evidence,
    )
    partial = CitationValidator().validate(
        GeneratedAnswer(answer="First claim [E1]. Second claim.", citation_ids=["E1"]),
        evidence,
    )

    assert detached.answer == REFUSAL_ANSWER
    assert detached.status == "validation_failed"
    assert partial.answer == REFUSAL_ANSWER
    assert partial.status == "validation_failed"


@pytest.mark.parametrize(
    ("content", "limit", "expected"),
    [
        ("12345", 5, "12345"),
        ("123456", 5, "12345"),
        ("", 5, ""),
        ("😀😀😀", 2, "😀😀"),
    ],
)
def test_validator_bounds_excerpt_by_configured_character_count(content, limit, expected):
    result = CitationValidator(excerpt_max_chars=limit).validate(
            GeneratedAnswer(answer="answer [E1].", citation_ids=["E1"]),
        [_evidence("E1", content=content)],
    )

    assert result.citations[0].excerpt == expected


def test_validator_exact_refusal_with_zero_citations_is_insufficient_evidence():
    result = CitationValidator().validate(
        GeneratedAnswer(answer=REFUSAL_ANSWER, citation_ids=[]),
        [_evidence("E1")],
    )

    assert result.answer == REFUSAL_ANSWER
    assert result.citations == []
    assert result.status == "insufficient_evidence"


def test_validator_non_refusal_without_citations_fails_closed():
    result = CitationValidator().validate(
        GeneratedAnswer(answer="An uncited answer.", citation_ids=[]),
        [_evidence("E1")],
    )

    assert result.answer == REFUSAL_ANSWER
    assert result.status == "validation_failed"
    assert result.citations == []


@pytest.mark.parametrize(
    "generated",
    [
        GeneratedAnswer(answer=None, citation_ids=None),
        GeneratedAnswer(answer="answer", citation_ids=None),
        GeneratedAnswer(answer=None, citation_ids=[]),
        GeneratedAnswer(answer="", citation_ids=[]),
    ],
)
def test_validator_malformed_generation_fails_closed(generated):
    result = CitationValidator().validate(generated, [_evidence("E1")])

    assert result.answer == REFUSAL_ANSWER
    assert result.citations == []
    assert result.status == "validation_failed"


def test_validator_empty_selected_evidence_is_insufficient_and_has_no_citations():
    result = CitationValidator().validate(
        GeneratedAnswer(answer=REFUSAL_ANSWER, citation_ids=[]),
        [],
    )

    assert result.answer == REFUSAL_ANSWER
    assert result.citations == []
    assert result.status == "insufficient_evidence"


def test_validator_refusal_with_citations_is_structurally_invalid():
    result = CitationValidator().validate(
        GeneratedAnswer(answer=REFUSAL_ANSWER, citation_ids=["E1"]),
        [_evidence("E1")],
    )

    assert result.answer == REFUSAL_ANSWER
    assert result.citations == []
    assert result.status == "validation_failed"


def test_validator_rejects_duplicate_or_malformed_supplied_evidence_ids():
    generated = GeneratedAnswer(answer="answer [E1].", citation_ids=["E1"])

    duplicate = CitationValidator().validate(generated, [_evidence("E1"), _evidence("E1")])
    malformed = CitationValidator().validate(generated, [_evidence("https://evil.example")])

    assert duplicate.status == "validation_failed"
    assert duplicate.citations == []
    assert malformed.status == "validation_failed"
    assert malformed.citations == []


def test_validator_does_not_mutate_generation_or_evidence():
    generated = GeneratedAnswer(answer="answer [E1].", citation_ids=["E1"])
    evidence = [_evidence("E1")]
    generated_before = generated.model_copy(deep=True)
    evidence_before = [item.model_copy(deep=True) for item in evidence]

    result = CitationValidator().validate(generated, evidence)

    assert generated == generated_before
    assert evidence == evidence_before
    result.citations[0].excerpt = "changed"
    assert evidence[0].candidate.content == "Canonical content for E1"


def test_validator_rejects_non_positive_excerpt_bound():
    with pytest.raises(ValueError, match="excerpt_max_chars"):
        CitationValidator(excerpt_max_chars=0)
