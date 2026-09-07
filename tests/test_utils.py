from youtube_study_tool.models import TranscriptSegment
from youtube_study_tool.utils import (
    Passage,
    build_chunked_text,
    escape_html_text,
    select_key_passages,
)


def test_build_chunked_text_can_omit_timestamps_for_text_only_sources() -> None:
    segments = (
        TranscriptSegment(text="First idea.", start=0.0, duration=0.0),
        TranscriptSegment(text="Second idea.", start=0.0, duration=0.0),
    )

    assert build_chunked_text(segments, include_timestamps=False) == [
        "First idea.\nSecond idea."
    ]


def test_escape_html_text_neutralizes_untrusted_metadata() -> None:
    assert escape_html_text('<img src=x onerror="alert(1)">&') == (
        "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;&amp;"
    )


def test_sanitize_untrusted_markdown_disables_links_images_and_html() -> None:
    from youtube_study_tool.utils import sanitize_untrusted_markdown

    safe = sanitize_untrusted_markdown(
        "[phish](https://attacker.example) ![pixel](https://attacker.example/x) "
        "<img src='https://attacker.example/y'>"
    )

    assert "attacker.example" not in safe
    assert "phish" in safe and "pixel" in safe
    assert "&lt;img" in safe


def test_sanitize_untrusted_markdown_removes_reference_links() -> None:
    from youtube_study_tool.utils import sanitize_untrusted_markdown

    safe = sanitize_untrusted_markdown(
        "[phish][target]\n\n[target]: https://attacker.example/secret"
    )

    assert "attacker.example" not in safe
    assert "phish" in safe


def test_sanitize_untrusted_markdown_removes_multiline_javascript_links() -> None:
    from youtube_study_tool.utils import sanitize_untrusted_markdown

    safe = sanitize_untrusted_markdown("[click\nhere](javascript:alert(1))")

    assert "javascript:" not in safe
    assert "alert" not in safe
    assert "click" in safe and "here" in safe


def test_select_key_passages_covers_short_lesson_instead_of_fixed_gap() -> None:
    passages = [
        Passage(text=f"Lesson point {index}", start=index * 10.0, end=index * 10.0 + 8)
        for index in range(8)
    ]

    selected = select_key_passages(passages, limit=5)

    assert len(selected) == 5
    assert [passage.start for passage in selected] == sorted(
        passage.start for passage in selected
    )
    assert selected[0].start == 0.0
    assert selected[-1].start >= 50.0
    assert any(20.0 <= passage.start <= 40.0 for passage in selected)


def test_select_key_passages_handles_non_positive_limit() -> None:
    assert select_key_passages([Passage("point", 0, 1)], limit=0) == []


def test_build_chunked_text_bounds_a_single_oversized_segment() -> None:
    chunks = build_chunked_text([TranscriptSegment("word " * 100, 0, 1)], max_chars=32)

    assert chunks
    assert max(len(chunk) for chunk in chunks) <= 32


def test_build_passages_bounds_a_single_oversized_segment() -> None:
    from youtube_study_tool.utils import build_passages

    passages = build_passages(
        [TranscriptSegment("word " * 1000, 0, 1)], target_chars=32
    )

    assert passages
    assert max(len(passage.text) for passage in passages) <= 32


def test_tokenize_keeps_unicode_words() -> None:
    from youtube_study_tool.utils import tokenize

    assert "künstliche" in tokenize("Künstliche Intelligenz")
    assert tokenize("神经网络") == ["神经网络"]
