from __future__ import annotations

from collections.abc import Iterable
import re
import time
from urllib.parse import parse_qs, urlparse

import requests
import yt_dlp
from youtube_transcript_api import NoTranscriptFound, Transcript, TranscriptList, YouTubeTranscriptApi

from youtube_study_tool.models import TranscriptBundle, TranscriptSegment
from youtube_study_tool.utils import clean_whitespace

VIDEO_ID_LENGTH = 11
LANGUAGE_CODE_RE = re.compile(r"^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$")


class TranscriptRetrievalError(Exception):
    """Raised when all available transcript backends fail."""


def extract_video_id(raw_value: str) -> str:
    candidate = clean_whitespace(raw_value)
    if not candidate:
        raise ValueError("Enter a YouTube URL or a video ID.")

    if len(candidate) == VIDEO_ID_LENGTH and all(char.isalnum() or char in "-_" for char in candidate):
        return candidate

    parsed = urlparse(candidate)
    host = parsed.netloc.lower()
    path_parts = [part for part in parsed.path.split("/") if part]

    if host in {"youtu.be", "www.youtu.be"} and path_parts:
        candidate_id = path_parts[0]
        if len(candidate_id) == VIDEO_ID_LENGTH and all(char.isalnum() or char in "-_" for char in candidate_id):
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
            if video_id and len(video_id) == VIDEO_ID_LENGTH and all(
                char.isalnum() or char in "-_" for char in video_id
            ):
                return video_id
        if path_parts and path_parts[0] in {"embed", "shorts", "live", "v"} and len(path_parts) > 1:
            candidate_id = path_parts[1]
            if len(candidate_id) == VIDEO_ID_LENGTH and all(char.isalnum() or char in "-_" for char in candidate_id):
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

    def fetch(self, source: str, preferred_languages: Iterable[str]) -> TranscriptBundle:
        video_id = extract_video_id(source)
        source_url = f"https://www.youtube.com/watch?v={video_id}"
        languages = tuple(dict.fromkeys(language.strip() for language in preferred_languages if language.strip()))

        primary_error: Exception | None = None
        try:
            return self._fetch_with_youtube_transcript_api(video_id, source_url, languages)
        except Exception as error:
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
        return TranscriptBundle(
            video_id=video_id,
            source_url=source_url,
            transcript_text=clean_whitespace(" ".join(segment.text for segment in segments)),
            segments=segments,
            language_code=getattr(transcript, "language_code", "unknown"),
            language_name=getattr(transcript, "language", "Unknown"),
            is_generated=bool(getattr(transcript, "is_generated", False)),
            duration_seconds=segments[-1].end if segments else 0.0,
            word_count=len(" ".join(segment.text for segment in segments).split()),
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
        segments = self._download_caption_segments(track["url"])
        transcript_text = clean_whitespace(" ".join(segment.text for segment in segments))
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
            video_title=clean_whitespace(str(info.get("title") or "")) or self._fetch_video_title(source_url),
        )

    def _select_transcript(
        self,
        transcript_list: TranscriptList,
        preferred_languages: Iterable[str],
    ) -> Transcript:
        languages = tuple(dict.fromkeys(language.strip() for language in preferred_languages if language.strip()))

        if languages:
            try:
                return transcript_list.find_transcript(languages)
            except NoTranscriptFound:
                pass

        english_requested = any(language.lower().startswith("en") for language in languages) or not languages
        if english_requested:
            for transcript in transcript_list:
                if getattr(transcript, "language_code", "").startswith("en"):
                    return transcript
            for transcript in transcript_list:
                if getattr(transcript, "is_translatable", False):
                    try:
                        return transcript.translate("en")
                    except Exception:
                        continue

        for transcript in transcript_list:
            return transcript

        raise NoTranscriptFound(video_id="unknown", requested_language_codes=list(languages), transcript_data=[])

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
        for language_code in self._candidate_language_codes(preferred_languages, subtitles, automatic_captions):
            manual_tracks = subtitles.get(language_code) or []
            track = self._pick_track(manual_tracks)
            if track:
                return track, False, language_code, str(track.get("name") or language_code)

            automatic_tracks = automatic_captions.get(language_code) or []
            track = self._pick_track(automatic_tracks)
            if track:
                return track, True, language_code, str(track.get("name") or language_code)

        raise TranscriptRetrievalError("No subtitle or automatic caption tracks were available via yt-dlp.")

    def _candidate_language_codes(
        self,
        preferred_languages: tuple[str, ...],
        subtitles: dict,
        automatic_captions: dict,
    ) -> list[str]:
        available_codes = list(dict.fromkeys([*subtitles.keys(), *automatic_captions.keys()]))
        if not available_codes:
            return []

        ordered: list[str] = []
        for language in preferred_languages or ("en", "en-US", "en-GB"):
            normalized = language.lower()
            for code in available_codes:
                lower_code = str(code).lower()
                if lower_code == normalized or lower_code.startswith(normalized) or normalized.startswith(lower_code):
                    if code not in ordered:
                        ordered.append(code)

        for fallback_code in available_codes:
            if fallback_code not in ordered:
                ordered.append(fallback_code)
        return ordered

    def _pick_track(self, tracks: list[dict]) -> dict | None:
        if not tracks:
            return None
        preferred_extensions = ("json3", "srv3", "srv2", "srv1", "vtt", "ttml")
        for extension in preferred_extensions:
            for track in tracks:
                if track.get("ext") == extension and track.get("url"):
                    return track
        for track in tracks:
            if track.get("url"):
                return track
        return None

    def _download_caption_segments(self, track_url: str) -> tuple[TranscriptSegment, ...]:
        payload = self._get_json_with_retries(track_url, timeout=20)
        segments: list[TranscriptSegment] = []

        for event in payload.get("events", []):
            text_parts = []
            for segment in event.get("segs", []) or []:
                text = clean_whitespace(str(segment.get("utf8") or ""))
                if text:
                    text_parts.append(text)
            if not text_parts:
                continue

            text = clean_whitespace(" ".join(text_parts))
            if not text:
                continue

            start = float(event.get("tStartMs", 0)) / 1000.0
            duration = float(event.get("dDurationMs", 0)) / 1000.0
            segments.append(TranscriptSegment(text=text, start=start, duration=max(duration, 0.0)))

        deduped_segments: list[TranscriptSegment] = []
        for segment in segments:
            if deduped_segments and deduped_segments[-1].text == segment.text:
                continue
            deduped_segments.append(segment)

        if not deduped_segments:
            raise TranscriptRetrievalError("A caption track was found, but it did not contain usable transcript text.")
        return tuple(deduped_segments)

    def _fetch_video_title(self, source_url: str) -> str | None:
        for attempt in range(3):
            try:
                response = requests.get(
                    "https://www.youtube.com/oembed",
                    params={"url": source_url, "format": "json"},
                    timeout=10,
                )
                if response.ok:
                    payload = response.json()
                    title = payload.get("title")
                    if isinstance(title, str) and title.strip():
                        return title.strip()
                    return None
            except requests.RequestException:
                if attempt == 2:
                    return None
                time.sleep(0.2 * (attempt + 1))
        return None

    def _get_json_with_retries(self, url: str, timeout: float, retries: int = 3) -> dict:
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                response = requests.get(url, timeout=timeout)
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise TranscriptRetrievalError("Caption endpoint returned a non-object JSON payload.")
                return payload
            except (requests.RequestException, ValueError, TranscriptRetrievalError) as error:
                last_error = error
                if attempt < retries - 1:
                    time.sleep(0.3 * (attempt + 1))
        raise TranscriptRetrievalError(f"Failed to download caption track after {retries} attempts: {last_error}")
