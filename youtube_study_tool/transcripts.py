from __future__ import annotations

import json
import logging
import math
import random
import re
import socket
import threading
import time
from collections.abc import Iterable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import ContextVar
from html import unescape
from urllib.parse import parse_qs, urlparse

import requests
import yt_dlp
from defusedxml import ElementTree
from defusedxml.common import DefusedXmlException
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from youtube_transcript_api import (
    NoTranscriptFound,
    Transcript,
    TranscriptList,
    YouTubeTranscriptApi,
)
from yt_dlp.networking._requests import (
    RequestsHTTPAdapter as YtDlpHTTPAdapter,
)
from yt_dlp.networking._requests import (
    RequestsRH as YtDlpRequestsRH,
)
from yt_dlp.networking._requests import (
    RequestsSession as YtDlpRequestsSession,
)
from yt_dlp.networking.common import Request as YtDlpRequest
from yt_dlp.networking.exceptions import RequestError as YtDlpRequestError

from youtube_study_tool.models import (
    MAX_TRANSCRIPT_CHARS,
    TranscriptBundle,
    TranscriptSegment,
)
from youtube_study_tool.utils import clean_whitespace

VIDEO_ID_LENGTH = 11
VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
LANGUAGE_CODE_RE = re.compile(r"^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$")
MAX_CAPTION_PAYLOAD_BYTES = 2_000_000
MAX_WATCH_PAYLOAD_BYTES = 8_000_000
TRANSCRIPT_FETCH_DEADLINE_SECONDS = 90.0
MAX_RETRY_DELAY_SECONDS = 5.0
MAX_CONCURRENT_TRANSCRIPT_FETCHES = 4
MAX_NETWORK_IDLE_TIMEOUT_SECONDS = 5.0
logger = logging.getLogger(__name__)

# A long-lived executor lets the caller abandon a slow fetch after the
# wall-clock budget without joining it at the call site (a `with
# ThreadPoolExecutor` would block on shutdown).
_FETCH_EXECUTOR = ThreadPoolExecutor(
    max_workers=MAX_CONCURRENT_TRANSCRIPT_FETCHES,
    thread_name_prefix="transcript-fetch",
)
_FETCH_ADMISSION = threading.BoundedSemaphore(MAX_CONCURRENT_TRANSCRIPT_FETCHES)
_FETCH_DEADLINE: ContextVar[float | None] = ContextVar("fetch_deadline", default=None)


class _FetchCancellation:
    """Share cancellation and active response handles with the waiting caller."""

    def __init__(self) -> None:
        self.event = threading.Event()
        self._lock = threading.Lock()
        self._responses: set[object] = set()
        self._persistent_responses: set[object] = set()
        self._connections: set[object] = set()
        self.session = _CancellableRequestsSession()

    def register(self, response: object, *, persistent: bool = False) -> None:
        with self._lock:
            cancelled = self.event.is_set()
            if not cancelled:
                self._responses.add(response)
                if persistent:
                    self._persistent_responses.add(response)
        if cancelled:
            close = getattr(response, "close", None)
            if callable(close):
                close()
            raise TranscriptTimeoutError(
                "Transcript retrieval was cancelled after timeout."
            )

    def release(self, response: object) -> None:
        with self._lock:
            self._responses.discard(response)
            self._persistent_responses.discard(response)

    def release_persistent_responses(self) -> None:
        with self._lock:
            responses = tuple(self._persistent_responses)
            self._persistent_responses.clear()
            for response in responses:
                self._responses.discard(response)

    def register_connection(self, connection: object) -> None:
        with self._lock:
            cancelled = self.event.is_set()
            if not cancelled:
                self._connections.add(connection)
        if cancelled:
            _interrupt_fetch_connection(connection)
            raise TranscriptTimeoutError(
                "Transcript retrieval was cancelled after timeout."
            )

    def release_connection(self, connection: object) -> None:
        with self._lock:
            self._connections.discard(connection)

    def cancel(self) -> None:
        self.event.set()
        with self._lock:
            responses = tuple(self._responses)
            connections = tuple(self._connections)
        for connection in connections:
            _interrupt_fetch_connection(connection)
        for response in responses:
            _interrupt_fetch_response(response)


def _interrupt_fetch_connection(connection: object) -> None:
    sock = getattr(connection, "sock", None)
    shutdown = getattr(sock, "shutdown", None)
    if callable(shutdown):
        try:
            shutdown(socket.SHUT_RDWR)
        except OSError:
            pass


def _interrupt_fetch_response(response: object) -> None:
    """Interrupt a blocked HTTP response read without waiting on its read lock."""
    raw = getattr(response, "raw", None) or getattr(response, "fp", None)
    raw_shutdown = getattr(raw, "_sock_shutdown", None)
    if callable(raw_shutdown):
        try:
            raw_shutdown(socket.SHUT_RDWR)
            return
        except OSError:
            pass
    connection = getattr(raw, "_connection", None)
    sock = getattr(connection, "sock", None)
    if sock is None:
        file_pointer = getattr(raw, "_fp", None)
        buffered = getattr(file_pointer, "fp", None)
        socket_io = getattr(buffered, "raw", None)
        sock = getattr(socket_io, "_sock", None)
    if sock is None:
        response_file = getattr(response, "fp", None)
        raw_file = getattr(response_file, "fp", None)
        socket_io = getattr(raw_file, "raw", None)
        sock = getattr(socket_io, "_sock", None)
    shutdown = getattr(sock, "shutdown", None)
    if callable(shutdown):
        try:
            shutdown(socket.SHUT_RDWR)
        except OSError:
            pass


_FETCH_CANCELLED: ContextVar[_FetchCancellation | None] = ContextVar(
    "fetch_cancelled", default=None
)


def _check_fetch_context() -> float | None:
    """Cooperatively stop fetch work when its caller abandons the deadline."""
    cancellation = _FETCH_CANCELLED.get()
    if cancellation is not None and cancellation.event.is_set():
        raise TranscriptTimeoutError(
            "Transcript retrieval was cancelled after timeout."
        )
    deadline = _FETCH_DEADLINE.get()
    if deadline is None:
        return None
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TranscriptTimeoutError("Transcript retrieval exceeded its time budget.")
    return remaining


def _register_fetch_response(response: requests.Response) -> None:
    cancellation = _FETCH_CANCELLED.get()
    if cancellation is not None:
        cancellation.register(response)


def _release_fetch_response(response: requests.Response) -> None:
    cancellation = _FETCH_CANCELLED.get()
    if cancellation is not None:
        cancellation.release(response)
    close = getattr(response, "close", None)
    if callable(close):
        close()


def _bounded_request_timeout(timeout: object, remaining: float | None) -> object:
    if timeout is None:
        timeout = MAX_NETWORK_IDLE_TIMEOUT_SECONDS
    if isinstance(timeout, tuple):
        connect_timeout, read_timeout = timeout
    else:
        connect_timeout = read_timeout = timeout

    connect = min(
        float(connect_timeout)
        if connect_timeout is not None
        else MAX_NETWORK_IDLE_TIMEOUT_SECONDS,
        MAX_NETWORK_IDLE_TIMEOUT_SECONDS,
    )
    read = min(
        float(read_timeout)
        if read_timeout is not None
        else MAX_NETWORK_IDLE_TIMEOUT_SECONDS,
        MAX_NETWORK_IDLE_TIMEOUT_SECONDS,
    )
    if remaining is not None:
        return min(connect, remaining), read
    if isinstance(timeout, tuple):
        return connect, read
    return connect


class _CancellablePoolMixin:
    """Expose the connection while Requests waits for response headers."""

    def _make_request(self, connection, *args, **kwargs):
        cancellation = _FETCH_CANCELLED.get()
        if cancellation is not None:
            cancellation.register_connection(connection)
        try:
            return super()._make_request(connection, *args, **kwargs)
        finally:
            if cancellation is not None:
                cancellation.release_connection(connection)


_CANCELLABLE_POOL_CLASSES: dict[type, type] = {}
_CANCELLABLE_POOL_CLASSES_LOCK = threading.Lock()


def _cancellable_pool_class(pool_class: type) -> type:
    if issubclass(pool_class, _CancellablePoolMixin):
        return pool_class
    with _CANCELLABLE_POOL_CLASSES_LOCK:
        wrapped = _CANCELLABLE_POOL_CLASSES.get(pool_class)
        if wrapped is None:
            wrapped = type(
                f"Cancellable{pool_class.__name__}",
                (_CancellablePoolMixin, pool_class),
                {"__module__": __name__},
            )
            _CANCELLABLE_POOL_CLASSES[pool_class] = wrapped
        return wrapped


class _CancellableHTTPAdapter(HTTPAdapter):
    @staticmethod
    def _configure_pool_classes(manager) -> None:
        manager.pool_classes_by_scheme = {
            scheme: _cancellable_pool_class(pool_class)
            for scheme, pool_class in manager.pool_classes_by_scheme.items()
        }

    def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs):
        super().init_poolmanager(connections, maxsize, block, **pool_kwargs)
        self._configure_pool_classes(self.poolmanager)

    def proxy_manager_for(self, proxy, **proxy_kwargs):
        manager = super().proxy_manager_for(proxy, **proxy_kwargs)
        self._configure_pool_classes(manager)
        return manager

    def send(self, request, **kwargs):
        response = super().send(request, **kwargs)
        cancellation = _FETCH_CANCELLED.get()
        if cancellation is not None:
            # Requests can consume redirect bodies before Session.get returns.
            # Keep each adapter response cancellable through that redirect walk.
            cancellation.register(response, persistent=True)
        return response


class _CancellableYtDlpHTTPAdapter(_CancellableHTTPAdapter, YtDlpHTTPAdapter):
    pass


class _CancellableYtDlpRequestsRH(YtDlpRequestsRH):
    RH_KEY = YtDlpRequestsRH.RH_KEY

    def _create_instance(self, cookiejar, legacy_ssl_support=None):
        session = YtDlpRequestsSession()
        adapter = _CancellableYtDlpHTTPAdapter(
            ssl_context=self._make_sslcontext(legacy_ssl_support=legacy_ssl_support),
            source_address=self.source_address,
            max_retries=Retry(False),
        )
        session.adapters.clear()
        session.headers = requests.models.CaseInsensitiveDict()
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.cookies = cookiejar
        session.trust_env = False
        return session

    def _send(self, request):
        try:
            return super()._send(request)
        except Exception:
            try:
                _check_fetch_context()
            except TranscriptTimeoutError as timeout_error:
                raise YtDlpRequestError(cause=timeout_error) from timeout_error
            raise


class _CancellableRequestsSession(requests.Session):
    def __init__(self) -> None:
        super().__init__()
        self.mount("http://", _CancellableHTTPAdapter())
        self.mount("https://", _CancellableHTTPAdapter())


class TranscriptRetrievalError(Exception):
    """Raised when all available transcript backends fail."""


class TranscriptTimeoutError(TranscriptRetrievalError):
    """Raised when transcript retrieval exceeds its wall-clock budget."""


class TranscriptBusyError(TranscriptRetrievalError):
    """Raised when all bounded transcript-fetch slots are occupied."""


def _payload_limit(url: str) -> int:
    # Watch pages and player metadata are HTML/JSON that legitimately exceed the
    # caption ceiling; caption payloads keep the smaller bound.
    if "/watch" in url or "youtubei/v1/player" in url:
        return MAX_WATCH_PAYLOAD_BYTES
    return MAX_CAPTION_PAYLOAD_BYTES


class _BoundedYouTubeSession(_CancellableRequestsSession):
    """Materialize YouTube API responses only after enforcing a byte ceiling."""

    @staticmethod
    def _materialize(response: requests.Response, limit: int) -> requests.Response:
        try:
            _check_fetch_context()
            content_length = response.headers.get(
                "Content-Length"
            ) or response.headers.get("content-length")
            if content_length is not None and int(content_length) > limit:
                raise TranscriptRetrievalError(
                    "Caption track is too large to process safely."
                )
            chunks: list[bytes] = []
            total = 0
            for chunk in response.iter_content(chunk_size=64 * 1024):
                _check_fetch_context()
                if not chunk:
                    continue
                total += len(chunk)
                if total > limit:
                    raise TranscriptRetrievalError(
                        "Caption track is too large to process safely."
                    )
                chunks.append(chunk)
            response._content = b"".join(chunks)
            response._content_consumed = True
            content_type = str(
                response.headers.get("Content-Type")
                or response.headers.get("content-type")
                or ""
            )
            if "charset=" not in content_type.lower():
                # requests defaults text/* without a charset to ISO-8859-1,
                # which silently mojibakes UTF-8 caption tracks.
                response.encoding = "utf-8-sig"
            return response
        except Exception:
            response.close()
            raise
        finally:
            if response._content_consumed:
                response.close()

    def get(self, url: str, **kwargs: object) -> requests.Response:
        kwargs["stream"] = True
        remaining = _check_fetch_context()
        kwargs["timeout"] = _bounded_request_timeout(
            kwargs.get("timeout", MAX_NETWORK_IDLE_TIMEOUT_SECONDS), remaining
        )
        response = super().get(url, **kwargs)
        try:
            _register_fetch_response(response)
            return self._materialize(response, _payload_limit(url))
        finally:
            _release_fetch_response(response)

    def post(self, url: str, **kwargs: object) -> requests.Response:
        kwargs["stream"] = True
        remaining = _check_fetch_context()
        kwargs["timeout"] = _bounded_request_timeout(
            kwargs.get("timeout", MAX_NETWORK_IDLE_TIMEOUT_SECONDS), remaining
        )
        response = super().post(url, **kwargs)
        try:
            _register_fetch_response(response)
            return self._materialize(response, _payload_limit(url))
        finally:
            _release_fetch_response(response)


class _DeadlineYoutubeDL(yt_dlp.YoutubeDL):
    """Check the fetch budget before each network request made by yt-dlp."""

    def build_request_director(self, handlers, preferences=None):
        wrapped_handlers = (
            _CancellableYtDlpRequestsRH if handler is YtDlpRequestsRH else handler
            for handler in handlers
        )
        return super().build_request_director(wrapped_handlers, preferences)

    def urlopen(self, req):
        remaining = _check_fetch_context()
        if remaining is not None:
            if isinstance(req, str):
                req = YtDlpRequest(req)
            if isinstance(req, YtDlpRequest):
                req.extensions["timeout"] = min(
                    remaining, MAX_NETWORK_IDLE_TIMEOUT_SECONDS
                )
        response = super().urlopen(req)
        cancellation = _FETCH_CANCELLED.get()
        if cancellation is not None:
            cancellation.register(response, persistent=True)
        return response


def extract_video_id(raw_value: str) -> str:
    candidate = clean_whitespace(raw_value)
    if not candidate:
        raise ValueError("Enter a YouTube URL or a video ID.")

    if VIDEO_ID_RE.fullmatch(candidate):
        return candidate

    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("That does not look like a valid YouTube URL or video ID.")
    host = parsed.netloc.lower()
    path_parts = [part for part in parsed.path.split("/") if part]

    if host in {"youtu.be", "www.youtu.be"} and path_parts:
        candidate_id = path_parts[0]
        if VIDEO_ID_RE.fullmatch(candidate_id):
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
            if video_id and VIDEO_ID_RE.fullmatch(video_id):
                return video_id
        if (
            path_parts
            and path_parts[0] in {"embed", "shorts", "live", "v"}
            and len(path_parts) > 1
        ):
            candidate_id = path_parts[1]
            if VIDEO_ID_RE.fullmatch(candidate_id):
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
        self.api = YouTubeTranscriptApi(http_client=_BoundedYouTubeSession())

    def fetch(
        self, source: str, preferred_languages: Iterable[str]
    ) -> TranscriptBundle:
        _check_fetch_context()
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
        except TranscriptTimeoutError:
            raise
        except Exception as error:  # noqa: BLE001 - either backend may fail with library-specific errors.
            _check_fetch_context()
            primary_error = error
            logger.info(
                "Primary transcript backend failed for %s: %s",
                video_id,
                type(error).__name__,
            )

        try:
            return self._fetch_with_ytdlp(video_id, source_url, languages)
        except TranscriptTimeoutError:
            raise
        except Exception as fallback_error:
            _check_fetch_context()
            raise TranscriptRetrievalError(
                "Transcript extraction failed with both backends.\n\n"
                f"Primary backend error: {type(primary_error).__name__}.\n"
                f"Fallback backend error: {type(fallback_error).__name__}."
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
        segments = self._normalize_segments(
            tuple(
                TranscriptSegment(
                    text=clean_whitespace(item.text),
                    start=item.start,
                    duration=item.duration,
                )
                for item in fetched
                if clean_whitespace(item.text)
            )
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
        video_title = self._fetch_video_title(source_url)
        _check_fetch_context()
        return TranscriptBundle(
            video_id=video_id,
            source_url=source_url,
            transcript_text=transcript_text,
            segments=segments,
            language_code=getattr(transcript, "language_code", "unknown"),
            language_name=getattr(transcript, "language", "Unknown"),
            is_generated=bool(getattr(transcript, "is_generated", False)),
            duration_seconds=_duration_seconds(segments),
            word_count=len(transcript_text.split()),
            video_title=video_title,
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
        video_title = clean_whitespace(str(info.get("title") or ""))
        if not video_title:
            video_title = self._fetch_video_title(source_url) or ""
        _check_fetch_context()
        return TranscriptBundle(
            video_id=video_id,
            source_url=source_url,
            transcript_text=transcript_text,
            segments=segments,
            language_code=language_code,
            language_name=language_name,
            is_generated=is_generated,
            duration_seconds=_duration_seconds(segments),
            word_count=len(transcript_text.split()),
            video_title=video_title or None,
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
        remaining = _check_fetch_context()
        options = {
            "skip_download": True,
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
            "socket_timeout": (
                min(MAX_NETWORK_IDLE_TIMEOUT_SECONDS, remaining)
                if remaining is not None
                else MAX_NETWORK_IDLE_TIMEOUT_SECONDS
            ),
        }
        cancellation = _FETCH_CANCELLED.get()
        try:
            with _DeadlineYoutubeDL(options) as downloader:
                info = downloader.extract_info(source_url, download=False)
        finally:
            if cancellation is not None:
                cancellation.release_persistent_responses()
        _check_fetch_context()
        return info

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
        response = self._get_response_with_retries(track_url, timeout=20, stream=True)
        raw_text = self._read_caption_response(response)
        stripped = raw_text.lstrip()
        extension = track_ext.lower().lstrip(".")

        if extension == "json3" or not stripped or stripped.startswith("{"):
            try:
                segments = self._segments_from_json3(json.loads(raw_text))
            except (TypeError, ValueError) as error:
                raise TranscriptRetrievalError(
                    "The JSON caption track was not valid JSON3 data."
                ) from error
        elif extension in {"srt", "vtt"} or stripped.startswith("WEBVTT"):
            segments = self._segments_from_vtt(raw_text)
        else:
            segments = self._segments_from_xml(raw_text)

        normalized_segments = self._normalize_segments(segments)
        deduped_segments: list[TranscriptSegment] = []
        seen_cues: set[tuple[str, float, float]] = set()
        for segment in normalized_segments:
            cue_key = (segment.text, segment.start, segment.duration)
            if cue_key in seen_cues:
                continue
            seen_cues.add(cue_key)
            deduped_segments.append(segment)

        if not deduped_segments:
            raise TranscriptRetrievalError(
                "A caption track was found, but it did not contain usable transcript text."
            )
        return tuple(deduped_segments)

    def _read_caption_response(self, response: requests.Response) -> str:
        """Read a caption response incrementally under the byte ceiling."""
        try:
            _check_fetch_context()
            headers = getattr(response, "headers", {})
            content_length = headers.get("Content-Length") or headers.get(
                "content-length"
            )
            try:
                too_large = (
                    content_length is not None
                    and int(content_length) > MAX_CAPTION_PAYLOAD_BYTES
                )
            except (TypeError, ValueError):
                too_large = False
            if too_large:
                raise TranscriptRetrievalError(
                    "Caption track is too large to process safely."
                )

            iterator = getattr(response, "iter_content", None)
            if callable(iterator):
                chunks: list[bytes] = []
                total = 0
                for chunk in iterator(chunk_size=64 * 1024):
                    _check_fetch_context()
                    if not chunk:
                        continue
                    if isinstance(chunk, str):
                        chunk = chunk.encode("utf-8")
                    total += len(chunk)
                    if total > MAX_CAPTION_PAYLOAD_BYTES:
                        raise TranscriptRetrievalError(
                            "Caption track is too large to process safely."
                        )
                    chunks.append(chunk)
                raw = b"".join(chunks)
            else:
                content = getattr(response, "content", None)
                if content is None:
                    content = str(getattr(response, "text", "") or "").encode("utf-8")
                if isinstance(content, str):
                    content = content.encode("utf-8")
                if len(content) > MAX_CAPTION_PAYLOAD_BYTES:
                    raise TranscriptRetrievalError(
                        "Caption track is too large to process safely."
                    )
                raw = content
            _check_fetch_context()
            content_type = str(
                headers.get("Content-Type") or headers.get("content-type") or ""
            )
            if "charset=" in content_type.lower():
                encoding = getattr(response, "encoding", None) or "utf-8-sig"
            else:
                # requests defaults text/* without a charset to ISO-8859-1,
                # which silently mojibakes non-ASCII captions; YouTube serves
                # UTF-8, so prefer it unless the server said otherwise.
                encoding = "utf-8-sig"
            try:
                return raw.decode(encoding)
            except (LookupError, UnicodeDecodeError) as error:
                raise TranscriptRetrievalError(
                    "Caption track returned invalid text encoding."
                ) from error
        finally:
            _release_fetch_response(response)

    def _normalize_segments(
        self, segments: Iterable[TranscriptSegment]
    ) -> tuple[TranscriptSegment, ...]:
        normalized: list[TranscriptSegment] = []
        for segment in segments:
            try:
                start = float(segment.start)
                duration = float(segment.duration)
            except (TypeError, ValueError):
                continue
            if (
                not math.isfinite(start)
                or not math.isfinite(duration)
                or start < 0
                or duration < 0
                or not math.isfinite(start + duration)
            ):
                continue
            text = clean_whitespace(str(segment.text or ""))
            if text:
                normalized.append(
                    TranscriptSegment(text=text, start=start, duration=duration)
                )
        normalized.sort(key=lambda segment: (segment.start, segment.duration))
        return tuple(normalized)

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
            if start is not None and end is not None and end >= start:
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
        except (ElementTree.ParseError, DefusedXmlException) as error:
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
        try:
            response = self._get_response_with_retries(
                "https://www.youtube.com/oembed",
                timeout=10,
                stream=True,
                params={"url": source_url, "format": "json"},
            )
            payload = json.loads(self._read_caption_response(response))
        except TranscriptTimeoutError:
            raise
        except (
            requests.RequestException,
            TranscriptRetrievalError,
            TypeError,
            ValueError,
        ):
            return None
        title = payload.get("title") if isinstance(payload, dict) else None
        return title.strip() if isinstance(title, str) and title.strip() else None

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
        finally:
            _release_fetch_response(response)
        _check_fetch_context()
        if not isinstance(payload, dict):
            raise TranscriptRetrievalError(
                "Caption endpoint returned a non-object JSON payload."
            )
        return payload

    def _get_response_with_retries(
        self,
        url: str,
        *,
        timeout: float,
        retries: int = 3,
        stream: bool = False,
        params: Mapping[str, str] | None = None,
    ) -> requests.Response:
        last_error: str | None = None
        for attempt in range(max(1, retries)):
            response: requests.Response | None = None
            try:
                remaining = _check_fetch_context()
                cancellation = _FETCH_CANCELLED.get()
                get = cancellation.session.get if cancellation else requests.get
                response = get(
                    url,
                    params=params,
                    timeout=_bounded_request_timeout(timeout, remaining),
                    stream=stream,
                )
                _register_fetch_response(response)
                response.raise_for_status()
                _check_fetch_context()
                return response
            except TranscriptTimeoutError:
                if response is not None:
                    _release_fetch_response(response)
                raise
            except requests.RequestException as error:
                if response is not None:
                    _release_fetch_response(response)
                status = getattr(getattr(error, "response", None), "status_code", None)
                last_error = type(error).__name__
                if status is not None:
                    last_error += f" HTTP {status}"
                if status is not None and 400 <= status < 500 and status != 429:
                    # Client errors cannot be fixed by retrying (404 dead video,
                    # 403 bot block); only rate limiting is worth another try.
                    break
                if attempt < max(1, retries) - 1:
                    delay = 0.5 * (2**attempt)
                    error_response = getattr(error, "response", None)
                    retry_after = (getattr(error_response, "headers", {}) or {}).get(
                        "Retry-After"
                    )
                    try:
                        if retry_after is not None:
                            delay = max(delay, float(retry_after))
                    except (TypeError, ValueError):
                        pass
                    delay = min(delay, MAX_RETRY_DELAY_SECONDS)
                    delay += random.uniform(0, 0.25)
                    remaining = _check_fetch_context()
                    if remaining is not None:
                        delay = min(delay, remaining)
                    time.sleep(delay)
        raise TranscriptRetrievalError(
            f"Failed to download caption track after {retries} attempts: {last_error}"
        )


def fetch_with_deadline(
    service: TranscriptService,
    source: str,
    preferred_languages: Iterable[str],
    *,
    timeout: float = TRANSCRIPT_FETCH_DEADLINE_SECONDS,
) -> TranscriptBundle:
    """Run transcript retrieval under a wall-clock bound.

    The worker cooperatively stops after cancellation. Active requests responses
    are closed when the caller's deadline expires, and admission is bounded so
    timed-out calls cannot accumulate behind occupied workers.
    """
    if not _FETCH_ADMISSION.acquire(blocking=False):
        raise TranscriptBusyError(
            "Transcript retrieval is busy. Please wait a moment and try again."
        )
    deadline = time.monotonic() + max(0.0, timeout)
    cancelled = _FetchCancellation()

    def worker() -> TranscriptBundle:
        deadline_token = _FETCH_DEADLINE.set(deadline)
        cancelled_token = _FETCH_CANCELLED.set(cancelled)
        try:
            _check_fetch_context()
            return service.fetch(source, preferred_languages)
        finally:
            cancelled.release_persistent_responses()
            cancelled.session.close()
            _FETCH_DEADLINE.reset(deadline_token)
            _FETCH_CANCELLED.reset(cancelled_token)

    try:
        future: Future = _FETCH_EXECUTOR.submit(worker)
    except BaseException:
        _FETCH_ADMISSION.release()
        cancelled.session.close()
        raise
    future.add_done_callback(lambda _future: _FETCH_ADMISSION.release())
    try:
        return future.result(timeout=max(0.0, deadline - time.monotonic()))
    except TimeoutError as error:
        cancelled.cancel()
        if future.cancel():
            cancelled.session.close()
        raise TranscriptTimeoutError(
            "Transcript retrieval timed out. YouTube may be slow or blocking "
            "this server; paste the transcript instead."
        ) from error


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


def _duration_seconds(segments: Iterable[TranscriptSegment]) -> float:
    return max((segment.end for segment in segments), default=0.0)
