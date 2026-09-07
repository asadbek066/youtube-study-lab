import json
from typing import ClassVar

import pytest
import requests

from youtube_study_tool.models import MAX_TRANSCRIPT_CHARS, TranscriptSegment
from youtube_study_tool.transcripts import (
    TranscriptRetrievalError,
    TranscriptService,
    extract_video_id,
    normalize_languages,
)


def test_extract_video_id_from_watch_url() -> None:
    assert (
        extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    )


def test_extract_video_id_from_short_link() -> None:
    assert extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_extract_video_id_rejects_non_https_schemes_and_unicode_ids() -> None:
    import pytest

    with pytest.raises(ValueError):
        extract_video_id("javascript://youtube.com/watch?v=dQw4w9WgXcQ")
    with pytest.raises(ValueError):
        extract_video_id("ééééééééééé")


def test_normalize_languages_defaults_to_english() -> None:
    assert normalize_languages("") == ("en", "en-US", "en-GB")


def test_normalize_languages_filters_invalid_and_dedupes() -> None:
    assert normalize_languages("en,en,  ,xx_yy,uz,EN") == ("en", "uz")


def test_fetch_rejects_oversized_primary_transcript(monkeypatch) -> None:
    class DummyTranscript:
        language_code = "en"
        language = "English"
        is_generated = False

        def fetch(self):
            return [
                type(
                    "Caption",
                    (),
                    {
                        "text": "x" * (MAX_TRANSCRIPT_CHARS + 1),
                        "start": 0.0,
                        "duration": 1.0,
                    },
                )()
            ]

    class DummyTranscriptList:
        def find_transcript(self, _languages):
            return DummyTranscript()

        def __iter__(self):
            return iter((DummyTranscript(),))

    service = TranscriptService()
    service.api = type(
        "DummyApi",
        (),
        {"list": lambda _self, _video_id: DummyTranscriptList()},
    )()
    monkeypatch.setattr(service, "_fetch_video_title", lambda _source_url: None)

    try:
        service._fetch_with_youtube_transcript_api(
            "dQw4w9WgXcQ", "https://www.youtube.com/watch?v=dQw4w9WgXcQ", ("en",)
        )
    except TranscriptRetrievalError as error:
        assert "Transcript is too long" in str(error)
    else:
        raise AssertionError("oversized transcript was accepted")


def test_youtube_api_client_enforces_caption_byte_limit(monkeypatch) -> None:
    from youtube_study_tool.transcripts import _BoundedYouTubeSession

    class DummyResponse:
        headers: ClassVar[dict[str, str]] = {}
        _content_consumed = False

        def iter_content(self, chunk_size: int):
            assert chunk_size > 0
            yield b"x" * 1_500_000
            yield b"x" * 1_000_001

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        "youtube_study_tool.transcripts.requests.Session.get",
        lambda _self, _url, **_kwargs: DummyResponse(),
    )
    with pytest.raises(TranscriptRetrievalError, match="too large"):
        _BoundedYouTubeSession().get("https://youtube.example/captions")


def test_youtube_api_post_client_enforces_caption_byte_limit(monkeypatch) -> None:
    from youtube_study_tool.transcripts import _BoundedYouTubeSession

    class DummyResponse:
        headers: ClassVar[dict[str, str]] = {}
        _content_consumed = False

        def iter_content(self, chunk_size: int):
            assert chunk_size > 0
            yield b"x" * 2_000_001

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        "youtube_study_tool.transcripts.requests.Session.post",
        lambda _self, _url, **_kwargs: DummyResponse(),
    )
    with pytest.raises(TranscriptRetrievalError, match="too large"):
        _BoundedYouTubeSession().post("https://youtube.example/transcript", json={})


def test_download_caption_segments_parses_json3_payload(monkeypatch) -> None:
    class DummyResponse:
        text = json.dumps(
            {
                "events": [
                    {
                        "tStartMs": 0,
                        "dDurationMs": 1200,
                        "segs": [{"utf8": "Hello"}, {"utf8": " world"}],
                    },
                    {
                        "tStartMs": 1500,
                        "dDurationMs": 800,
                        "segs": [{"utf8": "Next line"}],
                    },
                ]
            }
        )

        def raise_for_status(self) -> None:
            return None

    monkeypatch.setattr(
        "youtube_study_tool.transcripts.requests.get",
        lambda *args, **kwargs: DummyResponse(),
    )

    service = TranscriptService()
    segments = service._download_caption_segments("https://example.com/captions.json3")

    assert len(segments) == 2
    assert segments[0].text == "Hello world"
    assert segments[1].start == 1.5


def test_download_caption_segments_rejects_oversized_payload(monkeypatch) -> None:
    class DummyResponse:
        text = "x" * 2_000_001
        headers: ClassVar[dict[str, str]] = {}

        def raise_for_status(self) -> None:
            return None

    monkeypatch.setattr(
        "youtube_study_tool.transcripts.requests.get",
        lambda *args, **kwargs: DummyResponse(),
    )
    try:
        TranscriptService()._download_caption_segments(
            "https://example.com/captions.vtt"
        )
    except TranscriptRetrievalError as error:
        assert "too large" in str(error)
    else:
        raise AssertionError("oversized caption payload was accepted")


def test_download_caption_segments_stops_streaming_at_the_payload_limit(
    monkeypatch,
) -> None:
    class DummyResponse:
        headers: ClassVar[dict[str, str]] = {}

        def raise_for_status(self) -> None:
            return None

        def iter_content(self, chunk_size: int):
            assert chunk_size > 0
            yield b"x" * 1_500_000
            yield b"x" * 1_000_001

    monkeypatch.setattr(
        "youtube_study_tool.transcripts.requests.get",
        lambda *args, **kwargs: DummyResponse(),
    )
    try:
        TranscriptService()._download_caption_segments(
            "https://example.com/captions.vtt"
        )
    except TranscriptRetrievalError as error:
        assert "too large" in str(error)
    else:
        raise AssertionError("streaming caption payload was accepted")


def test_normalization_rejects_an_infinite_caption_endpoint() -> None:
    segments = TranscriptService()._normalize_segments(
        [TranscriptSegment("bad endpoint", 1e308, 1e308)]
    )
    assert segments == ()


def test_caption_network_errors_do_not_echo_signed_urls(monkeypatch) -> None:
    signed_url = "https://caption.example/track?sig=secret-value"

    def fail(*args, **kwargs):
        raise requests.RequestException(signed_url)

    monkeypatch.setattr("youtube_study_tool.transcripts.requests.get", fail)
    monkeypatch.setattr(
        "youtube_study_tool.transcripts.time.sleep", lambda _delay: None
    )
    with pytest.raises(TranscriptRetrievalError) as raised:
        TranscriptService()._get_response_with_retries(signed_url, timeout=1)
    assert signed_url not in str(raised.value)


def test_download_caption_segments_parses_webvtt(monkeypatch) -> None:
    class DummyResponse:
        text = """WEBVTT

00:00.000 --> 00:01.200
Hello <c.colorE5E5E5>world</c>

00:01.200 --> 00:02.000
Next line
"""

        def raise_for_status(self) -> None:
            return None

    monkeypatch.setattr(
        "youtube_study_tool.transcripts.requests.get",
        lambda *args, **kwargs: DummyResponse(),
    )

    segments = TranscriptService()._download_caption_segments(
        "https://example.com/captions.vtt", track_ext="vtt"
    )

    assert [
        (segment.text, segment.start, segment.duration) for segment in segments
    ] == [
        ("Hello world", 0.0, 1.2),
        ("Next line", 1.2, 0.8),
    ]


def test_download_caption_segments_parses_srt(monkeypatch) -> None:
    class DummyResponse:
        text = """1
00:00:00,000 --> 00:00:01,200
Hello world

2
00:00:01,200 --> 00:00:02,000
Next line
"""

        def raise_for_status(self) -> None:
            return None

    monkeypatch.setattr(
        "youtube_study_tool.transcripts.requests.get",
        lambda *args, **kwargs: DummyResponse(),
    )

    segments = TranscriptService()._download_caption_segments(
        "https://example.com/captions.srt", track_ext="srt"
    )

    assert [segment.text for segment in segments] == ["Hello world", "Next line"]
    assert segments[0].duration == 1.2


def test_xml_caption_entities_are_rejected(monkeypatch) -> None:
    service = TranscriptService()
    try:
        service._segments_from_xml(
            '<!DOCTYPE foo [<!ENTITY x "expanded">]>'
            '<transcript><text start="0">&x;</text></transcript>'
        )
    except TranscriptRetrievalError as error:
        assert "valid XML" in str(error)
    else:
        raise AssertionError("unsafe XML entity was accepted")


def test_caption_segments_are_sorted_and_repeated_speech_is_preserved(
    monkeypatch,
) -> None:
    class DummyResponse:
        text = """1
00:02 --> 00:03
again

2
00:00 --> 00:01
again
"""

        def raise_for_status(self) -> None:
            return None

    monkeypatch.setattr(
        "youtube_study_tool.transcripts.requests.get",
        lambda *args, **kwargs: DummyResponse(),
    )

    segments = TranscriptService()._download_caption_segments(
        "https://example.com/captions.srt", track_ext="srt"
    )

    assert [(segment.text, segment.start) for segment in segments] == [
        ("again", 0.0),
        ("again", 2.0),
    ]


def test_api_caption_invalid_timestamps_are_skipped(monkeypatch) -> None:
    class DummyTranscript:
        language_code = "en"
        language = "English"
        is_generated = False

        def fetch(self):
            return [
                type(
                    "Caption",
                    (),
                    {"text": "bad", "start": float("inf"), "duration": 1.0},
                )(),
                type("Caption", (), {"text": "good", "start": 4.0, "duration": 1.0})(),
            ]

    class DummyTranscriptList:
        def find_transcript(self, _languages):
            return DummyTranscript()

        def __iter__(self):
            return iter((DummyTranscript(),))

    service = TranscriptService()
    service.api = type(
        "DummyApi", (), {"list": lambda _self, _video_id: DummyTranscriptList()}
    )()
    monkeypatch.setattr(service, "_fetch_video_title", lambda _source_url: None)

    bundle = service._fetch_with_youtube_transcript_api(
        "dQw4w9WgXcQ", "https://www.youtube.com/watch?v=dQw4w9WgXcQ", ("en",)
    )

    assert [segment.text for segment in bundle.segments] == ["good"]
    assert bundle.duration_seconds == 5.0


def test_caption_parser_skips_non_finite_timestamps() -> None:
    segments = TranscriptService()._segments_from_json3(
        {
            "events": [
                {
                    "tStartMs": "NaN",
                    "dDurationMs": 1000,
                    "segs": [{"utf8": "bad"}],
                },
                {
                    "tStartMs": 0,
                    "dDurationMs": 1000,
                    "segs": [{"utf8": "good"}],
                },
            ]
        }
    )

    assert [segment.text for segment in segments] == ["good"]


def test_download_caption_segments_parses_srv_xml(monkeypatch) -> None:
    class DummyResponse:
        text = '<transcript><text start="0" dur="1.5">Hello &amp; welcome</text><text start="1.5" dur="0.5">Next</text></transcript>'

        def raise_for_status(self) -> None:
            return None

    monkeypatch.setattr(
        "youtube_study_tool.transcripts.requests.get",
        lambda *args, **kwargs: DummyResponse(),
    )

    segments = TranscriptService()._download_caption_segments(
        "https://example.com/captions.srv3", track_ext="srv3"
    )

    assert [segment.text for segment in segments] == ["Hello & welcome", "Next"]
    assert segments[1].start == 1.5


def test_download_caption_segments_parses_ttml_xml(monkeypatch) -> None:
    class DummyResponse:
        text = """<tt xmlns=\"http://www.w3.org/ns/ttml\"><body><div>
        <p begin=\"00:00:00.000\" end=\"00:00:01.250\">First</p>
        <p begin=\"1.250s\" dur=\"750ms\">Second</p>
        </div></body></tt>"""

        def raise_for_status(self) -> None:
            return None

    monkeypatch.setattr(
        "youtube_study_tool.transcripts.requests.get",
        lambda *args, **kwargs: DummyResponse(),
    )

    segments = TranscriptService()._download_caption_segments(
        "https://example.com/captions.ttml", track_ext="ttml"
    )

    assert [segment.text for segment in segments] == ["First", "Second"]
    assert segments[0].duration == 1.25
    assert segments[1].duration == 0.75
