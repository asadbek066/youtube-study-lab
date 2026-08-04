from youtube_study_tool.demo import build_demo_transcript
from youtube_study_tool.fallback import generate_fallback_bundle


def test_build_demo_transcript_is_complete_and_deterministic() -> None:
    first = build_demo_transcript()
    second = build_demo_transcript()

    assert first == second
    assert first.video_id == "study-demo"
    assert first.video_title == "How neural networks learn from examples"
    assert first.source_url == ""
    assert first.language_code == "en"
    assert first.language_name == "English"
    assert first.is_generated is False
    assert len(first.segments) >= 6
    assert first.duration_seconds == first.segments[-1].end
    assert first.transcript_text == " ".join(segment.text for segment in first.segments)
    assert first.word_count == len(first.transcript_text.split())
    assert all(segment.text and segment.duration > 0 for segment in first.segments)


def test_demo_study_pack_uses_plain_timestamps_without_dead_youtube_links() -> None:
    analysis = generate_fallback_bundle(build_demo_transcript())

    assert "00:00" in analysis.summary
    assert "youtube.com/watch?v=study-demo" not in analysis.summary
    assert "[00:00](" not in analysis.summary
