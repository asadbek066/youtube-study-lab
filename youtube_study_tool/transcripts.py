from __future__ import annotations

import logging
import math
import re
import time
from collections.abc import Iterable
from html import unescape
from urllib.parse import parse_qs, urlparse
from xml.etree import ElementTree

import requests
import yt_dlp
from youtube_transcript_api import (
    NoTranscriptFound,
    Transcript,
    TranscriptList,
    YouTubeTranscriptApi,
)

from youtube_study_tool.models import (
    MAX_TRANSCRIPT_CHARS,
    TranscriptBundle,
    TranscriptSegment,
)
from youtube_study_tool.utils import clean_whitespace

VIDEO_ID_LENGTH = 11
LANGUAGE_CODE_RE = re.compile(r"^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$")
logger = logging.getLogger(__name__)


class TranscriptRetrievalError(Exception):
    """Raised when all available transcript backends fail."""


def extract_video_id(raw_value: str) -> str:
    candidate = clean_whitespace(raw_value)
    if not candidate:
        raise ValueError("Enter a YouTube URL or a video ID.")

    if len(candidate) == VIDEO_ID_LENGTH and all(
        char.isalnum() or char in "-_" for char in candidate
    ):
        return candidate

    parsed = urlparse(candidate)
    host = parsed.netloc.lower()
    path_parts = [part for part in parsed.path.split("/") if part]

    if host in {"youtu.be", "www.youtu.be"} and path_parts:
        candidate_id = path_parts[0]
        if len(candidate_id) == VIDEO_ID_LENGTH and all(
            char.isalnum() or char in "-_" for char in candidate_id
        ):
            return candidate_id

    if host in {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "music.youtube.com",
        "www.youtube-nocookie.com",
        "youtube-nocookie.com",
    }:
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [None])[0]
            if (
                video_id
                and len(video_id) == VIDEO_ID_LENGTH
                and all(char.isalnum() or char in "-_" for char in video_id)
            ):
                return video_id
        if (
            path_parts
            and path_parts[0] in {"embed", "shorts", "live", "v"}
            and len(path_parts) > 1
        ):
            candidate_id = path_parts[1]
            if len(candidate_id) == VIDEO_ID_LENGTH and all(
                char.isalnum() or char in "-_" for char in candidate_id
            ):
                return candidate_id

    raise ValueError("That does not look like a valid YouTube URL or video ID.")


def normalize_languages(raw_languages: str) -> tuple[str, ...]:
    if not raw_languages:
        return ("en", "en-US", "en-GB")

    deduped: list[str] = []
    seen: set[str] = set()
    for item in raw_languages.split(","):
        language = item.strip()
        if not language:
            continue
        if not LANGUAGE_CODE_RE.match(language):
            continue
        normalized = language.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(language)
        if len(deduped) >= 10:
            break
    return tuple(deduped or ["en", "en-US", "en-GB"])


class TranscriptService:
    def __init__(self) -> None:
        self.api = YouTubeTranscriptApi()

    def fetch(
        self, source: str, preferred_languages: Iterable[str]
    ) -> TranscriptBundle:
        video_id = extract_video_id(source)
        source_url = f"https://www.youtube.com/watch?v={video_id}"
        languages = tuple(
            dict.fromkeys(
                language.strip() for language in preferred_languages if language.strip()
            )
        )

        primary_error: Exception | None = None
        try:
            return self._fetch_with_youtube_transcript_api(
                video_id, source_url, languages
            )
        except Exception as error:  # noqa: BLE001 - either backend may fail with library-specific errors.
            primary_error = error

        try:
            return self._fetch_with_ytdlp(video_id, source_url, languages)
        except Exception as fallback_error:
            raise TranscriptRetrievalError(
                "Transcript extraction failed with both backends.\n\n"
                f"Primary backend error:\n{primary_error}\n\n"
                f"Fallback backend error:\n{fallback_error}"
            ) from fallback_error

    def _fetch_with_youtube_transcript_api(
        self,
        video_id: str,
        source_url: str,
        preferred_languages: tuple[str, ...],
    ) -> TranscriptBundle:
        transcript_list = self.api.list(video_id)
        transcript = self._select_transcript(transcript_list, preferred_languages)
        fetched = transcript.fetch()
        segments = tuple(
            TranscriptSegment(
                text=clean_whitespace(item.text),
                start=item.start,
                duration=item.duration,
            )
            for item in fetched
            if clean_whitespace(item.text)
        )
        if not segments:
            raise TranscriptRetrievalError(
                "A transcript track was returned, but it contained no usable text."
            )
        transcript_text = clean_whitespace(
            " ".join(segment.text for segment in segments)
        )
        if len(transcript_text) > MAX_TRANSCRIPT_CHARS:
            raise TranscriptRetrievalError(
                f"Transcript is too long; keep it under {MAX_TRANSCRIPT_CHARS:,} characters."
            )
        return TranscriptBundle(
            video_id=video_id,
            source_url=source_url,
            transcript_text=transcript_text,
            segments=segments,
            language_code=getattr(transcript, "language_code", "unknown"),
            language_name=getattr(transcript, "language", "Unknown"),
            is_generated=bool(getattr(transcript, "is_generated", False)),
            duration_seconds=segments[-1].end if segments else 0.0,
            word_count=len(transcript_text.split()),
            video_title=self._fetch_video_title(source_url),
        )

    def _fetch_with_ytdlp(
        self,
        video_id: str,
        source_url: str,
        preferred_languages: tuple[str, ...],
    ) -> TranscriptBundle:
        info = self._extract_video_info(source_url)
        subtitles = info.get("subtitles") or {}
        automatic_captions = info.get("automatic_captions") or {}
        track, is_generated, language_code, language_name = self._select_caption_track(
            subtitles,
            automatic_captions,
            preferred_languages,
        )
        segments = self._download_caption_segments(
            track["url"], track_ext=str(track.get("ext") or "")
        )
        transcript_text = clean_whitespace(
            " ".join(segment.text for segment in segments)
        )
        if len(transcript_text) > MAX_TRANSCRIPT_CHARS:
            raise TranscriptRetrievalError(
                f"Transcript is too long; keep it under {MAX_TRANSCRIPT_CHARS:,} characters."
            )
        return TranscriptBundle(
            video_id=video_id,
            source_url=source_url,
            transcript_text=transcript_text,
            segments=segments,
            language_code=language_code,
            language_name=language_name,
            is_generated=is_generated,
            duration_seconds=segments[-1].end if segments else 0.0,
            word_count=len(transcript_text.split()),
            video_title=clean_whitespace(str(info.get("title") or ""))
            or self._fetch_video_title(source_url),
        )

    def _select_transcript(
        self,
        transcript_list: TranscriptList,
        preferred_languages: Iterable[str],
    ) -> Transcript:
        languages = tuple(
            dict.fromkeys(
                language.strip() for language in preferred_languages if language.strip()
            )
        )

        if languages:
            try:
                return transcript_list.find_transcript(languages)
            except NoTranscriptFound:
                pass

        english_requested = (
            any(language.lower().startswith("en") for language in languages)
            or not languages
        )
        if english_requested:
            for transcript in transcript_list:
                if getattr(transcript, "language_code", "").startswith("en"):
                    return transcript
            for transcript in transcript_list:
                if getattr(transcript, "is_translatable", False):
                    try:
                        return transcript.translate("en")
                    except Exception:
                        logger.debug(
                            "Transcript translation failed; trying the next track.",
                            exc_info=True,
                        )
                        continue

        for transcript in transcript_list:
            return transcript

        raise NoTranscriptFound(
            video_id="unknown",
            requested_language_codes=list(languages),
            transcript_data=[],
        )

    def _extract_video_info(self, source_url: str) -> dict:
        options = {
            "skip_download": True,
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
        }
        with yt_dlp.YoutubeDL(options) as downloader:
            return downloader.extract_info(source_url, download=False)

    def _select_caption_track(
        self,
        subtitles: dict,
        automatic_captions: dict,
        preferred_languages: tuple[str, ...],
    ) -> tuple[dict, bool, str, str]:
        for language_code in self._candidate_language_codes(
            preferred_languages, subtitles, automatic_captions
        ):
            manual_tracks = subtitles.get(language_code) or []
            track = self._pick_track(manual_tracks)
            if track:
                return (
                    track,
                    False,
                    language_code,
                    str(track.get("name") or language_code),
                )

            automatic_tracks = automatic_captions.get(language_code) or []
            track = self._pick_track(automatic_tracks)
            if track:
                return (
                    track,
                    True,
                    language_code,
                    str(track.get("name") or language_code),
                )

        raise TranscriptRetrievalError(
            "No subtitle or automatic caption tracks were available via yt-dlp."
        )

    def _candidate_language_codes(
        self,
        preferred_languages: tuple[str, ...],
        subtitles: dict,
        automatic_captions: dict,
    ) -> list[str]:
        available_codes = list(
            dict.fromkeys([*subtitles.keys(), *automatic_captions.keys()])
        )
        if not available_codes:
            return []

        ordered: list[str] = []
        for language in preferred_languages or ("en", "en-US", "en-GB"):
            normalized = language.lower()
            for code in available_codes:
                lower_code = str(code).lower()
                if (
                    lower_code == normalized
                    or lower_code.startswith(normalized)
                    or normalized.startswith(lower_code)
                ) and code not in ordered:
                    ordered.append(code)

        for fallback_code in available_codes:
            if fallback_code not in ordered:
                ordered.append(fallback_code)
        return ordered

    def _pick_track(self, tracks: list[dict]) -> dict | None:
        if not tracks:
            return None
        preferred_extensions = (
            "json3",
            "srv3",
            "srv2",
            "srv1",
            "vtt",
            "srt",
            "ttml",
        )
        for extension in preferred_extensions:
            for track in tracks:
                if track.get("ext") == extension and track.get("url"):
                    return track
        for track in tracks:
            if track.get("url"):
                return track
        return None

    def _download_caption_segments(
        self, track_url: str, *, track_ext: str = ""
    ) -> tuple[TranscriptSegment, ...]:
        response = self._get_response_with_retries(track_url, timeout=20)
        raw_text = str(getattr(response, "text", "") or "")
        stripped = raw_text.lstrip()
        extension = track_ext.lower().lstrip(".")

        if extension == "json3" or not stripped or stripped.startswith("{"):
            try:
                segments = self._segments_from_json3(response.json())
            except (TypeError, ValueError) as error:
                raise TranscriptRetrievalError(
                    "The JSON caption track was not valid JSON3 data."
                ) from error
        elif extension in {"srt", "vtt"} or stripped.startswith("WEBVTT"):
            segments = self._segments_from_vtt(raw_text)
        else:
            segments = self._segments_from_xml(raw_text)

        deduped_segments: list[TranscriptSegment] = []
        for segment in segments:
            if deduped_segments and deduped_segments[-1].text == segment.text:
                continue
            deduped_segments.append(segment)

        if not deduped_segments:
            raise TranscriptRetrievalError(
                "A caption track was found, but it did not contain usable transcript text."
            )
        return tuple(deduped_segments)

    def _segments_from_json3(self, payload: object) -> list[TranscriptSegment]:
        if not isinstance(payload, dict):
            raise TypeError("JSON3 payload must be an object")
        segments: list[TranscriptSegment] = []
        for event in payload.get("events", []) or []:
            if not isinstance(event, dict):
                continue
            text_parts = []
            for item in event.get("segs", []) or []:
                if not isinstance(item, dict):
                    continue
                text = clean_whitespace(str(item.get("utf8") or ""))
                if text:
                    text_parts.append(text)
            text = clean_whitespace(" ".join(text_parts))
            if not text:
                continue
            try:
                start_ms = float(event.get("tStartMs", 0))
                duration_ms = float(event.get("dDurationMs", 0))
            except (TypeError, ValueError):
                continue
            if (
                not math.isfinite(start_ms)
                or not math.isfinite(duration_ms)
                or start_ms < 0
                or duration_ms < 0
            ):
                continue
            segments.append(
                TranscriptSegment(
                    text=text, start=start_ms / 1000.0, duration=duration_ms / 1000.0
                )
            )
        return segments

    def _segments_from_vtt(self, raw_text: str) -> list[TranscriptSegment]:
        segments: list[TranscriptSegment] = []
        start: float | None = None
        end: float | None = None
        cue_lines: list[str] = []

        def flush() -> None:
            nonlocal start, end, cue_lines
            if start is not None and end is not None:
                text = clean_whitespace(
                    re.sub(r"<[^>]+>", "", unescape(" ".join(cue_lines)))
                )
                if text:
                    segments.append(
                        TranscriptSegment(
                            text=text,
                            start=max(start, 0.0),
                            duration=max(end - start, 0.0),
                        )
                    )
            start = end = None
            cue_lines = []

        for line in [*raw_text.splitlines(), ""]:
            match = re.match(r"^\s*(\S+)\s+-->\s+(\S+)", line)
            if match:
                flush()
                try:
                    start = _parse_caption_time(match.group(1))
                    end = _parse_caption_time(match.group(2))
                except ValueError:
                    start = end = None
                continue
            if not line.strip():
                flush()
            elif start is not None:
                cue_lines.append(line.strip())
        return segments

    def _segments_from_xml(self, raw_text: str) -> list[TranscriptSegment]:
        try:
            root = ElementTree.fromstring(raw_text)
        except ElementTree.ParseError as error:
            raise TranscriptRetrievalError(
                "The caption track was neither JSON3, WebVTT, nor valid XML."
            ) from error

        segments: list[TranscriptSegment] = []
        for element in root.iter():
            tag = element.tag.rsplit("}", 1)[-1]
            if tag not in {"text", "p"}:
                continue
            text = clean_whitespace("".join(element.itertext()))
            if not text:
                continue
            start_raw = (
                element.attrib.get("start") or element.attrib.get("begin") or "0"
            )
            end_raw = element.attrib.get("end")
            duration_raw = element.attrib.get("dur") or element.attrib.get("duration")
            try:
                start = _parse_caption_time(start_raw)
                if end_raw is not None:
                    duration = _parse_caption_time(end_raw) - start
                else:
                    duration = _parse_caption_time(duration_raw or "0")
            except ValueError:
                continue
            if duration < 0:
                continue
            segments.append(
                TranscriptSegment(
                    text=text, start=max(start, 0.0), duration=max(duration, 0.0)
                )
            )
        return segments

    def _fetch_video_title(self, source_url: str) -> str | None:
        for attempt in range(3):
            try:
                response = requests.get(
                    "https://www.youtube.com/oembed",
                    params={"url": source_url, "format": "json"},
                    timeout=10,
                )
                if response.ok:
                    try:
                        payload = response.json()
                    except (TypeError, ValueError):
                        return None
                    title = payload.get("title")
                    if isinstance(title, str) and title.strip():
                        return title.strip()
                    return None
            except requests.RequestException:
                if attempt == 2:
                    return None
                time.sleep(0.2 * (attempt + 1))
        return None

    def _get_json_with_retries(
        self, url: str, timeout: float, retries: int = 3
    ) -> dict:
        response = self._get_response_with_retries(
            url, timeout=timeout, retries=retries
        )
        try:
            payload = response.json()
        except (TypeError, ValueError) as error:
            raise TranscriptRetrievalError(
                "Caption endpoint returned invalid JSON."
            ) from error
        if not isinstance(payload, dict):
            raise TranscriptRetrievalError(
                "Caption endpoint returned a non-object JSON payload."
            )
        return payload

    def _get_response_with_retries(
        self, url: str, *, timeout: float, retries: int = 3
    ) -> requests.Response:
        last_error: Exception | None = None
        for attempt in range(max(1, retries)):
            try:
                response = requests.get(url, timeout=timeout)
                response.raise_for_status()
                return response
            except requests.RequestException as error:
                last_error = error
                if attempt < max(1, retries) - 1:
                    time.sleep(0.3 * (attempt + 1))
        raise TranscriptRetrievalError(
            f"Failed to download caption track after {retries} attempts: {last_error}"
        )


def _parse_caption_time(value: object) -> float:
    text = str(value).strip().replace(",", ".")
    if not text:
        raise ValueError("empty caption timestamp")
    if text.endswith("ms"):
        parsed = float(text[:-2]) / 1000.0
    elif text.endswith("s"):
        parsed = float(text[:-1])
    else:
        parts = text.split(":")
        if len(parts) == 3:
            hours, minutes, seconds = (float(part) for part in parts)
            parsed = hours * 3600 + minutes * 60 + seconds
        elif len(parts) == 2:
            minutes, seconds = (float(part) for part in parts)
            parsed = minutes * 60 + seconds
        else:
            parsed = float(text)
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError("caption timestamp must be finite and non-negative")
    return parsed
