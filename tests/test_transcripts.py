import json
from typing import ClassVar

import pytest
import requests

from youtube_study_tool.models import MAX_TRANSCRIPT_CHARS, TranscriptSegment
from youtube_study_tool.transcripts import (
    MAX_CAPTION_PAYLOAD_BYTES,
    MAX_WATCH_PAYLOAD_BYTES,
    TranscriptBusyError,
    TranscriptRetrievalError,
    TranscriptService,
    TranscriptTimeoutError,
    _BoundedYouTubeSession,
    _payload_limit,
    extract_video_id,
    fetch_with_deadline,
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


def test_download_caption_segments_skips_reversed_webvtt_cues(monkeypatch) -> None:
    class DummyResponse:
        text = """WEBVTT

00:02 --> 00:01
Reversed cue

00:01 --> 00:02
Valid cue
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

    assert [segment.text for segment in segments] == ["Valid cue"]


def test_caption_retries_close_http_error_responses(monkeypatch) -> None:
    responses = []

    class DummyResponse:
        status_code = 503

        def __init__(self) -> None:
            self.closed = False

        def raise_for_status(self) -> None:
            raise requests.HTTPError(response=self)

        def close(self) -> None:
            self.closed = True

    def fail(*args, **kwargs):
        response = DummyResponse()
        responses.append(response)
        return response

    monkeypatch.setattr("youtube_study_tool.transcripts.requests.get", fail)
    monkeypatch.setattr(
        "youtube_study_tool.transcripts.time.sleep", lambda _delay: None
    )

    with pytest.raises(TranscriptRetrievalError, match="after 2 attempts"):
        TranscriptService()._get_response_with_retries(
            "https://example.com/captions.vtt", timeout=1, retries=2, stream=True
        )

    assert [response.closed for response in responses] == [True, True]


def test_get_json_with_retries_closes_successful_response(monkeypatch) -> None:
    class DummyResponse:
        def __init__(self) -> None:
            self.closed = False

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {"status": "ok"}

        def close(self) -> None:
            self.closed = True

    response = DummyResponse()
    monkeypatch.setattr(
        "youtube_study_tool.transcripts.requests.get",
        lambda *args, **kwargs: response,
    )

    payload = TranscriptService()._get_json_with_retries(
        "https://example.com/captions.json", timeout=1
    )

    assert payload == {"status": "ok"}
    assert response.closed is True


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


def test_fetch_video_title_passes_source_url_to_oembed(monkeypatch) -> None:
    source_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    calls = []

    class DummyResponse:
        headers: ClassVar[dict[str, str]] = {}
        content = b'{"title":"A useful lesson"}'
        encoding = "utf-8"

        def raise_for_status(self) -> None:
            return None

        def close(self) -> None:
            return None

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return DummyResponse()

    monkeypatch.setattr("youtube_study_tool.transcripts.requests.get", fake_get)

    title = TranscriptService()._fetch_video_title(source_url)

    assert title == "A useful lesson"
    assert calls == [
        (
            "https://www.youtube.com/oembed",
            {
                "params": {"url": source_url, "format": "json"},
                "timeout": 5.0,
                "stream": True,
            },
        )
    ]


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


def test_fetch_falls_back_to_ytdlp_when_the_api_backend_fails(monkeypatch) -> None:
    from youtube_study_tool.models import TranscriptSegment
    from youtube_study_tool.transcripts import TranscriptBundle

    service = TranscriptService()
    fallback_bundle = TranscriptBundle(
        video_id="dQw4w9WgXcQ",
        source_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        transcript_text="fallback text",
        segments=(TranscriptSegment("fallback text", 0.0, 1.0),),
        language_code="en",
        language_name="English",
        is_generated=True,
        duration_seconds=1.0,
        word_count=2,
    )

    def failing_primary(*_args):
        raise RuntimeError("api backend exploded")

    monkeypatch.setattr(service, "_fetch_with_youtube_transcript_api", failing_primary)
    monkeypatch.setattr(service, "_fetch_with_ytdlp", lambda *_args: fallback_bundle)

    bundle = service.fetch("https://www.youtube.com/watch?v=dQw4w9WgXcQ", ("en",))

    assert bundle.transcript_text == "fallback text"
    assert bundle.video_id == "dQw4w9WgXcQ"


def test_fetch_reports_both_backend_failures_and_type_names_only(monkeypatch) -> None:
    service = TranscriptService()
    source = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def failing_primary(*_args):
        raise RuntimeError("primary leaked detail")

    def failing_fallback(*_args):
        raise ValueError("fallback leaked detail")

    monkeypatch.setattr(service, "_fetch_with_youtube_transcript_api", failing_primary)
    monkeypatch.setattr(service, "_fetch_with_ytdlp", failing_fallback)

    with pytest.raises(TranscriptRetrievalError) as raised:
        service.fetch(source, ("en",))

    message = str(raised.value)
    assert "RuntimeError" in message
    assert "ValueError" in message
    assert "primary leaked detail" not in message
    assert "fallback leaked detail" not in message


def test_caption_client_errors_are_not_retried(monkeypatch) -> None:
    class DummyResponse:
        status_code = 404
        closed = False

        def raise_for_status(self) -> None:
            raise requests.HTTPError(response=self)

        def close(self) -> None:
            self.closed = True

    calls = []
    sleeps = []

    def fail(*args, **kwargs):
        calls.append(args)
        return DummyResponse()

    monkeypatch.setattr("youtube_study_tool.transcripts.requests.get", fail)
    monkeypatch.setattr("youtube_study_tool.transcripts.time.sleep", sleeps.append)

    with pytest.raises(TranscriptRetrievalError, match="HTTP 404"):
        TranscriptService()._get_response_with_retries(
            "https://example.com/captions.vtt", timeout=1
        )

    assert len(calls) == 1
    assert sleeps == []


def test_caption_rate_limit_responses_are_still_retried(monkeypatch) -> None:
    class DummyResponse:
        status_code = 429
        headers: ClassVar[dict[str, str]] = {"Retry-After": "3"}
        closed = False

        def raise_for_status(self) -> None:
            raise requests.HTTPError(response=self)

        def close(self) -> None:
            self.closed = True

    calls = []
    sleeps = []

    def fail(*args, **kwargs):
        calls.append(args)
        return DummyResponse()

    monkeypatch.setattr("youtube_study_tool.transcripts.requests.get", fail)
    monkeypatch.setattr("youtube_study_tool.transcripts.time.sleep", sleeps.append)

    with pytest.raises(TranscriptRetrievalError, match="HTTP 429"):
        TranscriptService()._get_response_with_retries(
            "https://example.com/captions.vtt", timeout=1
        )

    assert len(calls) == 3
    # Retry-After (3s) wins over the jittered exponential backoff, and every
    # delay stays within the documented ceiling plus jitter.
    assert all(3.0 <= delay <= 3.25 for delay in sleeps)


def test_caption_retry_backoff_is_jittered_and_bounded(monkeypatch) -> None:
    class DummyResponse:
        status_code = 503
        headers: ClassVar[dict[str, str]] = {}
        closed = False

        def raise_for_status(self) -> None:
            raise requests.HTTPError(response=self)

        def close(self) -> None:
            self.closed = True

    sleeps = []

    monkeypatch.setattr(
        "youtube_study_tool.transcripts.requests.get",
        lambda *args, **kwargs: DummyResponse(),
    )
    monkeypatch.setattr("youtube_study_tool.transcripts.time.sleep", sleeps.append)

    with pytest.raises(TranscriptRetrievalError, match="HTTP 503"):
        TranscriptService()._get_response_with_retries(
            "https://example.com/captions.vtt", timeout=1
        )

    assert 0.5 <= sleeps[0] <= 0.75
    assert 1.0 <= sleeps[1] <= 1.25


class _FakeTranscript:
    def __init__(self, code: str, translatable: bool = False) -> None:
        self.language_code = code
        self.language = code
        self.is_translatable = translatable

    def translate(self, code: str) -> "_FakeTranscript":
        return _FakeTranscript(code)


class _FakeTranscriptList:
    def __init__(self, transcripts, findable=None) -> None:
        self._transcripts = list(transcripts)
        self._findable = findable or {}

    def find_transcript(self, languages):
        for language in languages:
            if language in self._findable:
                return self._findable[language]
        from youtube_transcript_api import NoTranscriptFound

        raise NoTranscriptFound(
            video_id="video",
            requested_language_codes=list(languages),
            transcript_data=[],
        )

    def __iter__(self):
        return iter(self._transcripts)


def test_select_transcript_prefers_a_translatable_track_when_english_is_missing() -> (
    None
):
    german = _FakeTranscript("de", translatable=True)
    service = TranscriptService()

    selected = service._select_transcript(_FakeTranscriptList([german]), ("en",))

    assert selected.language_code == "en"
    assert selected is not german


def test_select_transcript_falls_back_to_the_first_available_track() -> None:
    japanese = _FakeTranscript("ja", translatable=False)
    service = TranscriptService()

    selected = service._select_transcript(_FakeTranscriptList([japanese]), ("en",))

    assert selected is japanese


def test_select_transcript_uses_exact_language_when_available() -> None:
    german = _FakeTranscript("de")
    english = _FakeTranscript("en")
    service = TranscriptService()
    transcript_list = _FakeTranscriptList([english, german], {"de": german})

    selected = service._select_transcript(transcript_list, ("de", "en"))

    assert selected is german


def test_caption_track_selection_prefers_manual_tracks_in_language_order() -> None:
    service = TranscriptService()
    manual = {"url": "https://example.test/de.vtt", "ext": "vtt", "name": "Deutsch"}
    automatic = {"url": "https://example.test/en.vtt", "ext": "vtt", "name": "English"}
    subtitles = {"de": [manual]}
    automatic_captions = {"en": [automatic]}

    track, generated, code, name = service._select_caption_track(
        subtitles, automatic_captions, ("de", "en")
    )

    assert track is manual
    assert generated is False
    assert code == "de"
    assert name == "Deutsch"


def test_caption_track_selection_falls_back_to_automatic_captions() -> None:
    service = TranscriptService()
    automatic = {"url": "https://example.test/en.vtt", "ext": "vtt"}

    track, generated, code, _name = service._select_caption_track(
        {}, {"en": [automatic]}, ("en",)
    )

    assert track is automatic
    assert generated is True
    assert code == "en"


def test_caption_track_selection_prefers_json3_over_vtt() -> None:
    service = TranscriptService()
    vtt = {"url": "https://example.test/en.vtt", "ext": "vtt"}
    json3 = {"url": "https://example.test/en.json3", "ext": "json3"}

    track, _generated, _code, _name = service._select_caption_track(
        {"en": [vtt, json3]}, {}, ("en",)
    )

    assert track is json3


def test_payload_limit_separates_watch_pages_from_captions() -> None:
    assert (
        _payload_limit("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        == MAX_WATCH_PAYLOAD_BYTES
    )
    assert (
        _payload_limit("https://www.youtube.com/youtubei/v1/player")
        == MAX_WATCH_PAYLOAD_BYTES
    )
    assert (
        _payload_limit("https://example.test/captions.vtt") == MAX_CAPTION_PAYLOAD_BYTES
    )


def test_cancellable_pool_adapter_preserves_specialized_proxy_pools() -> None:
    from types import SimpleNamespace

    from urllib3 import HTTPConnectionPool

    import youtube_study_tool.transcripts as transcripts_module

    class OptionalSocksPool(HTTPConnectionPool):
        pass

    manager = SimpleNamespace(
        pool_classes_by_scheme={"http": OptionalSocksPool, "https": OptionalSocksPool}
    )

    transcripts_module._CancellableHTTPAdapter._configure_pool_classes(manager)

    assert issubclass(manager.pool_classes_by_scheme["http"], OptionalSocksPool)
    assert issubclass(manager.pool_classes_by_scheme["https"], OptionalSocksPool)


class _CaptionResponse:
    def __init__(self, chunks, headers=None, encoding=None) -> None:
        self._chunks = list(chunks)
        self.headers = headers or {}
        self.encoding = encoding
        self.closed = False
        self.iterated = False
        self._content = None
        self._content_consumed = False

    def iter_content(self, chunk_size=0):
        self.iterated = True
        yield from self._chunks

    def close(self) -> None:
        self.closed = True


def test_caption_response_decodes_utf8_without_a_charset_header() -> None:
    # requests reports ISO-8859-1 for text/* without a charset; the reader must
    # prefer UTF-8 anyway.
    response = _CaptionResponse(["café".encode()], encoding="ISO-8859-1")

    text = TranscriptService()._read_caption_response(response)

    assert text == "café"
    assert response.closed


def test_bounded_session_overrides_the_implicit_latin1_default() -> None:
    payload = "über café".encode()

    class _Response:
        headers: ClassVar[dict[str, str]] = {"Content-Type": "text/xml"}
        encoding = "ISO-8859-1"

        def iter_content(self, chunk_size=0):
            yield payload

        def close(self) -> None:
            pass

    materialized = _BoundedYouTubeSession._materialize(
        _Response(), MAX_CAPTION_PAYLOAD_BYTES
    )

    assert materialized.encoding == "utf-8-sig"
    assert materialized._content.decode(materialized.encoding) == "über café"


def test_caption_response_honors_an_explicit_charset() -> None:
    response = _CaptionResponse(
        ["café".encode("latin-1")],
        headers={"Content-Type": "text/vtt; charset=iso-8859-1"},
        encoding="iso-8859-1",
    )

    text = TranscriptService()._read_caption_response(response)

    assert text == "café"


def test_caption_response_rejects_invalid_utf8() -> None:
    response = _CaptionResponse([b"\xff\xfe\x00bad"])

    with pytest.raises(TranscriptRetrievalError, match="invalid text encoding"):
        TranscriptService()._read_caption_response(response)


def test_caption_response_rejects_oversized_content_length_before_reading() -> None:
    response = _CaptionResponse(
        [b"small"],
        headers={"Content-Length": str(MAX_CAPTION_PAYLOAD_BYTES + 1)},
    )

    with pytest.raises(TranscriptRetrievalError, match="too large"):
        TranscriptService()._read_caption_response(response)

    assert response.iterated is False
    assert response.closed


def test_bounded_session_allows_larger_watch_payloads() -> None:
    payload = b"x" * (MAX_CAPTION_PAYLOAD_BYTES + 1000)
    response = _CaptionResponse([payload])

    materialized = _BoundedYouTubeSession._materialize(
        response, MAX_WATCH_PAYLOAD_BYTES
    )

    assert len(materialized._content) == len(payload)


def test_bounded_session_preserves_an_explicit_charset() -> None:
    payload = "über café".encode("latin-1")

    class _Response:
        headers: ClassVar[dict[str, str]] = {
            "Content-Type": "text/vtt; charset=iso-8859-1"
        }
        encoding = "iso-8859-1"

        def iter_content(self, chunk_size=0):
            yield payload

        def close(self) -> None:
            pass

    materialized = _BoundedYouTubeSession._materialize(
        _Response(), MAX_CAPTION_PAYLOAD_BYTES
    )

    assert materialized.encoding == "iso-8859-1"
    assert materialized._content.decode(materialized.encoding) == "über café"


def test_bounded_session_rejects_caption_payloads_over_the_caption_cap() -> None:
    payload = b"x" * (MAX_CAPTION_PAYLOAD_BYTES + 1)
    response = _CaptionResponse([payload])

    with pytest.raises(TranscriptRetrievalError, match="too large"):
        _BoundedYouTubeSession._materialize(response, MAX_CAPTION_PAYLOAD_BYTES)


def test_fetch_with_deadline_returns_the_service_result() -> None:
    class FakeService:
        def fetch(self, source, languages):
            return ("bundle", source, tuple(languages))

    result = fetch_with_deadline(FakeService(), "video", ("en",), timeout=1.0)

    assert result == ("bundle", "video", ("en",))


def test_fetch_with_deadline_raises_a_timeout_error() -> None:
    import time as time_module

    class SlowService:
        def fetch(self, source, languages):
            time_module.sleep(0.3)
            return "late"

    with pytest.raises(TranscriptTimeoutError, match="timed out"):
        fetch_with_deadline(SlowService(), "video", ("en",), timeout=0.01)


def test_timed_out_fetches_cannot_fill_an_unbounded_worker_queue(monkeypatch) -> None:
    import queue
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    import youtube_study_tool.transcripts as transcripts_module

    worker_count = transcripts_module.MAX_CONCURRENT_TRANSCRIPT_FETCHES
    executor = ThreadPoolExecutor(max_workers=worker_count)
    admission = threading.BoundedSemaphore(worker_count)
    monkeypatch.setattr(transcripts_module, "_FETCH_EXECUTOR", executor)
    monkeypatch.setattr(transcripts_module, "_FETCH_ADMISSION", admission)
    started: queue.Queue[None] = queue.Queue()
    finished: queue.Queue[None] = queue.Queue()
    results: queue.Queue[BaseException | None] = queue.Queue()

    class BlockedService:
        def fetch(self, _source: str, _languages: tuple[str, ...]) -> None:
            started.put(None)
            try:
                while True:
                    transcripts_module._check_fetch_context()
                    time.sleep(0.005)
            except TranscriptTimeoutError:
                finished.put(None)
                raise

    def submit_blocked_request() -> None:
        try:
            fetch_with_deadline(BlockedService(), "video", ("en",), timeout=1.0)
        except TranscriptTimeoutError as error:
            results.put(error)
        else:
            results.put(None)

    try:
        callers = [
            threading.Thread(target=submit_blocked_request) for _ in range(worker_count)
        ]
        for caller in callers:
            caller.start()
        for _ in range(worker_count):
            started.get(timeout=2)

        started_at = time.monotonic()
        with pytest.raises(TranscriptBusyError, match="busy"):
            fetch_with_deadline(BlockedService(), "video", ("en",), timeout=0.01)
        assert time.monotonic() - started_at < 0.5
        assert started.empty()

        for caller in callers:
            caller.join(timeout=2)
            assert not caller.is_alive()
        assert all(
            isinstance(results.get(timeout=1), TranscriptTimeoutError)
            for _ in range(worker_count)
        )
        for _ in range(worker_count):
            finished.get(timeout=1)

        class QuickService:
            def fetch(self, _source: str, _languages: tuple[str, ...]) -> str:
                return "recovered"

        assert fetch_with_deadline(QuickService(), "video", ("en",), timeout=1.0) == (
            "recovered"
        )
    finally:
        executor.shutdown(wait=True)


def test_timed_out_transcript_workers_free_slots_without_external_intervention(
    monkeypatch,
) -> None:
    import queue
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    import youtube_study_tool.transcripts as transcripts_module

    worker_count = transcripts_module.MAX_CONCURRENT_TRANSCRIPT_FETCHES
    executor = ThreadPoolExecutor(max_workers=worker_count)
    admission = threading.BoundedSemaphore(worker_count)
    monkeypatch.setattr(transcripts_module, "_FETCH_EXECUTOR", executor)
    monkeypatch.setattr(transcripts_module, "_FETCH_ADMISSION", admission)
    started: queue.Queue[None] = queue.Queue()
    finished: queue.Queue[None] = queue.Queue()
    results: queue.Queue[BaseException | None] = queue.Queue()

    class CooperativeService:
        def fetch(self, _source: str, _languages: tuple[str, ...]) -> None:
            started.put(None)
            try:
                while True:
                    transcripts_module._check_fetch_context()
                    time.sleep(0.005)
            except TranscriptTimeoutError:
                finished.put(None)
                raise

    def call_fetch() -> None:
        try:
            fetch_with_deadline(CooperativeService(), "video", ("en",), timeout=0.5)
        except TranscriptTimeoutError as error:
            results.put(error)
        else:
            results.put(None)

    callers = [threading.Thread(target=call_fetch) for _ in range(worker_count)]
    try:
        for caller in callers:
            caller.start()
        for _ in range(worker_count):
            started.get(timeout=2)

        with pytest.raises(TranscriptBusyError, match="busy"):
            fetch_with_deadline(CooperativeService(), "video", ("en",), timeout=0.1)

        for caller in callers:
            caller.join(timeout=2)
            assert not caller.is_alive()
        assert all(
            isinstance(results.get(timeout=1), TranscriptTimeoutError)
            for _ in range(worker_count)
        )
        for _ in range(worker_count):
            finished.get(timeout=1)

        class QuickService:
            def fetch(self, _source: str, _languages: tuple[str, ...]) -> str:
                return "recovered"

        assert fetch_with_deadline(QuickService(), "video", ("en",), timeout=1.0) == (
            "recovered"
        )
    finally:
        executor.shutdown(wait=True)


@pytest.mark.parametrize("client_kind", ["direct", "bounded_session"])
def test_timed_out_streaming_response_is_closed_and_releases_worker(
    client_kind: str,
) -> None:
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    import youtube_study_tool.transcripts as transcripts_module

    started = threading.Event()
    server_finished = threading.Event()
    stop_stream = threading.Event()

    class SlowStreamHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Length", "100000")
            self.end_headers()
            started.set()
            try:
                for _ in range(100_000):
                    if stop_stream.wait(0.01):
                        break
                    self.wfile.write(b"x")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                server_finished.set()

        def log_message(self, *_args) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), SlowStreamHandler)
    server.daemon_threads = True
    server.block_on_close = False
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    service = TranscriptService()
    worker_finished = threading.Event()
    url = f"http://127.0.0.1:{server.server_port}/slow"

    class StreamingService:
        def fetch(self, _source, _languages):
            try:
                if client_kind == "bounded_session":
                    with transcripts_module._BoundedYouTubeSession() as session:
                        return session.get(url)
                response = service._get_response_with_retries(
                    url, timeout=5.0, stream=True
                )
                return service._read_caption_response(response)
            finally:
                worker_finished.set()

    try:
        with pytest.raises(TranscriptTimeoutError, match="timed out"):
            fetch_with_deadline(StreamingService(), "video", ("en",), timeout=0.2)

        assert started.wait(1.0)
        assert worker_finished.wait(1.0)
        assert server_finished.wait(1.0)

        class QuickService:
            def fetch(self, _source, _languages):
                return "recovered"

        assert fetch_with_deadline(QuickService(), "video", ("en",), timeout=1.0) == (
            "recovered"
        )
    finally:
        stop_stream.set()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=1.0)
        worker_finished.wait(1.0)


@pytest.mark.parametrize("client_kind", ["direct", "bounded_session", "ytdlp"])
def test_timed_out_slow_response_headers_interrupt_the_active_connection(
    client_kind: str,
) -> None:
    import socketserver
    import threading
    import time

    import youtube_study_tool.transcripts as transcripts_module

    started = threading.Event()
    server_finished = threading.Event()

    class SlowHeaderHandler(socketserver.BaseRequestHandler):
        def handle(self) -> None:
            try:
                self.request.recv(4096)
                self.request.sendall(b"HTTP/1.1 200 OK\r\nX-Slow: ")
                started.set()
                for _ in range(100):
                    self.request.sendall(b"a")
                    time.sleep(0.03)
                self.request.sendall(b"\r\nContent-Length: 2\r\n\r\nok")
                self.request.settimeout(2.0)
                self.request.recv(1)
            except OSError:
                pass
            finally:
                server_finished.set()

    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), SlowHeaderHandler)
    server.daemon_threads = True
    server.block_on_close = False
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    worker_finished = threading.Event()
    service = TranscriptService()
    url = f"http://127.0.0.1:{server.server_address[1]}/slow-headers"

    class HeaderService:
        def fetch(self, _source, _languages):
            try:
                if client_kind == "ytdlp":
                    with transcripts_module._DeadlineYoutubeDL(
                        {"quiet": True, "no_warnings": True, "socket_timeout": 5}
                    ) as downloader:
                        return downloader.urlopen(url)
                if client_kind == "bounded_session":
                    with transcripts_module._BoundedYouTubeSession() as session:
                        return session.get(url)
                return service._get_response_with_retries(url, timeout=5.0, stream=True)
            finally:
                worker_finished.set()

    try:
        with pytest.raises(TranscriptTimeoutError, match="timed out"):
            fetch_with_deadline(HeaderService(), "video", ("en",), timeout=0.5)

        assert started.wait(1.0)
        assert worker_finished.wait(1.0)
        assert server_finished.wait(1.0)

        class QuickService:
            def fetch(self, _source, _languages):
                return "recovered"

        assert fetch_with_deadline(QuickService(), "video", ("en",), timeout=1.0) == (
            "recovered"
        )
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=1.0)


def test_timed_out_redirect_body_is_interrupted_before_following_location() -> None:
    import threading
    import time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    started = threading.Event()
    server_finished = threading.Event()

    class SlowRedirectHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/start":
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"ok")
                return

            self.send_response(302)
            self.send_header("Location", "/final")
            self.send_header("Content-Length", "100")
            self.end_headers()
            started.set()
            try:
                for _ in range(100):
                    self.wfile.write(b"x")
                    self.wfile.flush()
                    time.sleep(0.03)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                server_finished.set()

        def log_message(self, *_args) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), SlowRedirectHandler)
    server.daemon_threads = True
    server.block_on_close = False
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    service = TranscriptService()
    worker_finished = threading.Event()
    url = f"http://127.0.0.1:{server.server_port}/start"

    class RedirectService:
        def fetch(self, _source, _languages):
            try:
                return service._get_response_with_retries(url, timeout=5.0, stream=True)
            finally:
                worker_finished.set()

    try:
        with pytest.raises(TranscriptTimeoutError, match="timed out"):
            fetch_with_deadline(RedirectService(), "video", ("en",), timeout=0.2)

        assert started.wait(1.0)
        assert worker_finished.wait(1.0)
        assert server_finished.wait(1.0)

        class QuickService:
            def fetch(self, _source, _languages):
                return "recovered"

        assert fetch_with_deadline(QuickService(), "video", ("en",), timeout=1.0) == (
            "recovered"
        )
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=1.0)


def test_ytdlp_checks_deadline_between_network_requests(monkeypatch) -> None:
    import threading
    import time

    import yt_dlp
    from yt_dlp.networking.common import Request

    requests_seen: list[Request] = []
    worker_finished = threading.Event()

    def fake_urlopen(_downloader, request):
        requests_seen.append(request)
        return object()

    def fake_extract_info(downloader, _source_url, download=False):
        downloader.urlopen(Request("https://example.test/first"))
        time.sleep(0.3)
        downloader.urlopen(Request("https://example.test/second"))
        return {}

    monkeypatch.setattr(yt_dlp.YoutubeDL, "urlopen", fake_urlopen)
    monkeypatch.setattr(yt_dlp.YoutubeDL, "extract_info", fake_extract_info)
    service = TranscriptService()

    class ExtractingService:
        def fetch(self, _source, _languages):
            try:
                return service._extract_video_info(
                    "https://www.youtube.com/watch?v=abc"
                )
            finally:
                worker_finished.set()

    with pytest.raises(TranscriptTimeoutError, match="timed out"):
        fetch_with_deadline(ExtractingService(), "video", ("en",), timeout=0.2)

    assert worker_finished.wait(1.0)
    assert len(requests_seen) == 1
    assert 0 < requests_seen[0].extensions["timeout"] <= 0.2


def test_timed_out_ytdlp_response_body_is_interrupted() -> None:
    import socketserver
    import threading

    import youtube_study_tool.transcripts as transcripts_module

    started = threading.Event()
    server_finished = threading.Event()
    stop_stream = threading.Event()

    class SlowBodyHandler(socketserver.BaseRequestHandler):
        def handle(self) -> None:
            try:
                self.request.recv(4096)
                self.request.sendall(
                    b"HTTP/1.1 200 OK\r\nContent-Length: 100000\r\n"
                    b"Connection: close\r\n\r\n"
                )
                started.set()
                for _ in range(100_000):
                    if stop_stream.wait(0.01):
                        break
                    self.request.sendall(b"x")
            except OSError:
                pass
            finally:
                server_finished.set()

    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), SlowBodyHandler)
    server.daemon_threads = True
    server.block_on_close = False
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    worker_finished = threading.Event()
    url = f"http://127.0.0.1:{server.server_address[1]}/slow-body"

    class YtDlpStreamingService:
        def fetch(self, _source, _languages):
            try:
                with transcripts_module._DeadlineYoutubeDL(
                    {"quiet": True, "no_warnings": True, "socket_timeout": 5}
                ) as downloader:
                    response = downloader.urlopen(url)
                    started.set()
                    return response.read()
            finally:
                worker_finished.set()

    try:
        with pytest.raises(TranscriptTimeoutError, match="timed out"):
            fetch_with_deadline(YtDlpStreamingService(), "video", ("en",), timeout=0.5)

        assert started.wait(1.0)
        assert worker_finished.wait(1.0)
        assert server_finished.wait(1.0)

        class QuickService:
            def fetch(self, _source, _languages):
                return "recovered"

        assert fetch_with_deadline(QuickService(), "video", ("en",), timeout=1.0) == (
            "recovered"
        )
    finally:
        stop_stream.set()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=1.0)
        worker_finished.wait(1.0)
