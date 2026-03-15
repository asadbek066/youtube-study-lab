from youtube_study_tool.fallback import generate_fallback_bundle
from youtube_study_tool.models import TranscriptBundle, TranscriptSegment


def test_generate_fallback_bundle_creates_all_sections() -> None:
    bundle = TranscriptBundle(
        video_id="abc123xyz00",
        source_url="https://www.youtube.com/watch?v=abc123xyz00",
        transcript_text=(
            "Python lists let you store items in order. "
            "Dictionaries store key value pairs. "
            "Functions help you reuse logic and keep code organized."
        ),
        segments=(
            TranscriptSegment(text="Python lists let you store items in order.", start=0, duration=4),
            TranscriptSegment(text="Dictionaries store key value pairs.", start=5, duration=4),
            TranscriptSegment(text="Functions help you reuse logic and keep code organized.", start=10, duration=5),
        ),
        language_code="en",
        language_name="English",
        is_generated=False,
        duration_seconds=15,
        word_count=20,
        video_title="Python basics",
    )

    analysis = generate_fallback_bundle(bundle)

    assert "## Summary" in analysis.summary
    assert "### 1. Overview" in analysis.summary
    assert "### 6. One-paragraph compressed version" in analysis.summary
    assert "## Study Notes" in analysis.study_notes
    assert "### 1. Topic" in analysis.study_notes
    assert "### 6. What to remember" in analysis.study_notes
    assert "## Quiz" in analysis.quiz
    assert "### 1. Multiple-choice questions" in analysis.quiz
    assert "### 2. Short-answer questions" in analysis.quiz
    assert "### 3. Application-based questions" in analysis.quiz
    assert analysis.quiz.count("Answer:") == 18
    assert analysis.quiz.count("Explanation:") == 18
    assert analysis.classification.video_type in {
        "tutorial",
        "lecture",
        "motivational",
        "interview",
        "commentary",
        "storytelling",
        "coding walkthrough",
    }


def test_generate_fallback_bundle_uses_tutorial_shape() -> None:
    bundle = TranscriptBundle(
        video_id="abc123xyz00",
        source_url="https://www.youtube.com/watch?v=abc123xyz00",
        transcript_text=(
            "In this tutorial we build a portfolio website step by step. "
            "First we set up the project, then create the layout, then deploy it."
        ),
        segments=(
            TranscriptSegment(text="In this tutorial we build a portfolio website step by step.", start=0, duration=4),
            TranscriptSegment(text="First we set up the project, then create the layout, then deploy it.", start=5, duration=4),
        ),
        language_code="en",
        language_name="English",
        is_generated=False,
        duration_seconds=9,
        word_count=22,
        video_title="Build a portfolio website",
    )

    analysis = generate_fallback_bundle(bundle)

    assert "### 3. Step-by-step breakdown" in analysis.summary
    assert analysis.classification.video_type in {"tutorial", "coding walkthrough"}
