from youtube_study_tool.transcripts import (
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


def test_normalize_languages_defaults_to_english() -> None:
    assert normalize_languages("") == ("en", "en-US", "en-GB")


def test_normalize_languages_filters_invalid_and_dedupes() -> None:
    assert normalize_languages("en,en,  ,xx_yy,uz,EN") == ("en", "uz")


def test_download_caption_segments_parses_json3_payload(monkeypatch) -> None:
    class DummyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
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

    monkeypatch.setattr(
        "youtube_study_tool.transcripts.requests.get",
        lambda *args, **kwargs: DummyResponse(),
    )

    service = TranscriptService()
    segments = service._download_caption_segments("https://example.com/captions.json3")

    assert len(segments) == 2
    assert segments[0].text == "Hello world"
    assert segments[1].start == 1.5


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
