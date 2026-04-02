from youtube_study_tool.transcripts import TranscriptService, extract_video_id, normalize_languages


def test_extract_video_id_from_watch_url() -> None:
    assert extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"


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

    monkeypatch.setattr("youtube_study_tool.transcripts.requests.get", lambda *args, **kwargs: DummyResponse())

    service = TranscriptService()
    segments = service._download_caption_segments("https://example.com/captions.json3")

    assert len(segments) == 2
    assert segments[0].text == "Hello world"
    assert segments[1].start == 1.5
