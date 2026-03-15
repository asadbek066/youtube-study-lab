from youtube_study_tool.classification import heuristic_classification, parse_classification_json
from youtube_study_tool.models import TranscriptBundle, TranscriptSegment


def test_parse_classification_json_accepts_required_shape() -> None:
    raw = """
    {
      "video_type": "tutorial",
      "confidence": 0.88,
      "reason": "The transcript uses step-based instructional language.",
      "best_summary_style": "condensed step-by-step summary",
      "best_note_style": "procedural checklist"
    }
    """

    result = parse_classification_json(raw)

    assert result.video_type == "tutorial"
    assert result.confidence == 0.88


def test_heuristic_classification_detects_coding_walkthrough() -> None:
    bundle = TranscriptBundle(
        video_id="code123abcd",
        source_url="https://www.youtube.com/watch?v=code123abcd",
        transcript_text=(
            "In this coding walkthrough we build a React component, create a function, "
            "debug an API call, and deploy the app from the terminal."
        ),
        segments=(
            TranscriptSegment(
                text="In this coding walkthrough we build a React component, create a function,",
                start=0,
                duration=4,
            ),
            TranscriptSegment(
                text="debug an API call, and deploy the app from the terminal.",
                start=5,
                duration=4,
            ),
        ),
        language_code="en",
        language_name="English",
        is_generated=False,
        duration_seconds=9,
        word_count=23,
        video_title="Build a React app",
    )

    result = heuristic_classification(bundle)

    assert result.video_type == "coding walkthrough"
    assert result.confidence >= 0.35
