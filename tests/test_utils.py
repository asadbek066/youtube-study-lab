from youtube_study_tool.models import TranscriptSegment
from youtube_study_tool.utils import build_chunked_text, escape_html_text


def test_build_chunked_text_can_omit_timestamps_for_text_only_sources() -> None:
    segments = (
        TranscriptSegment(text="First idea.", start=0.0, duration=0.0),
        TranscriptSegment(text="Second idea.", start=0.0, duration=0.0),
    )

    assert build_chunked_text(segments, include_timestamps=False) == ["First idea.\nSecond idea."]


def test_escape_html_text_neutralizes_untrusted_metadata() -> None:
    assert escape_html_text('<img src=x onerror="alert(1)">&') == (
        "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;&amp;"
    )
