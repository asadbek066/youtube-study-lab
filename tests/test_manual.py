import re

import pytest

from youtube_study_tool.fallback import generate_fallback_bundle
from youtube_study_tool.manual import build_manual_transcript


def test_build_manual_transcript_normalizes_user_text() -> None:
    raw_text = "  Neural networks learn from examples.\n\nThey improve by comparing predictions with targets.  "

    bundle = build_manual_transcript(raw_text, title="  My lecture notes  ")

    assert bundle.video_id == "pasted-transcript"
    assert bundle.source_url == ""
    assert bundle.video_title == "My lecture notes"
    assert bundle.source_label == "Pasted transcript"
    assert bundle.transcript_text == (
        "Neural networks learn from examples. They improve by comparing predictions with targets."
    )
    assert bundle.duration_seconds == 0
    assert bundle.word_count == 12
    assert len(bundle.segments) == 1


def test_build_manual_transcript_rejects_empty_text() -> None:
    with pytest.raises(ValueError, match="Paste transcript text"):
        build_manual_transcript("   ")


def test_manual_study_pack_contains_no_fabricated_timestamps() -> None:
    bundle = build_manual_transcript(
        "A model predicts an answer. The loss measures its error. Gradient descent updates the weights."
    )
    analysis = generate_fallback_bundle(bundle)
    complete_pack = "\n".join((analysis.summary, analysis.study_notes, analysis.quiz))

    assert not re.search(r"\b\d{2}:\d{2}(?::\d{2})?\b", complete_pack)
