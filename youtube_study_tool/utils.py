from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from html import escape

from youtube_study_tool.models import TranscriptSegment

STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "also",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "with",
    "would",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "like",
    "really",
    "kind",
    "sort",
    "actually",
    "okay",
    "yeah",
    "um",
    "uh",
}

WORD_RE = re.compile(r"[^\W\d_][\w'-]{2,}", re.UNICODE)
SENTENCE_END_RE = re.compile(r"[.!?]\s*$")
MARKDOWN_IMAGE_RE = re.compile(r"!\[([^]]*)\]\([^)]*\)", re.DOTALL)
MARKDOWN_LINK_RE = re.compile(r"\[([^]]*)\]\([^)]*\)", re.DOTALL)
MARKDOWN_REFERENCE_DEF_RE = re.compile(r"(?m)^\s{0,3}\[[^]\n]+\]:\s*\S.*$")
MARKDOWN_REFERENCE_USE_RE = re.compile(r"\[([^]\n]+)\]\[[^]\n]*\]")
BARE_URL_RE = re.compile(r"https?://[^\s<>()]+", re.IGNORECASE)


@dataclass(frozen=True)
class Passage:
    text: str
    start: float
    end: float


def format_seconds(total_seconds: float) -> str:
    total_seconds = max(0, int(total_seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def timestamp_url(video_id: str, start_seconds: float) -> str:
    seconds = max(0, int(start_seconds))
    return f"https://www.youtube.com/watch?v={video_id}&t={seconds}s"


def timestamp_reference(
    video_id: str, start_seconds: float, *, linked: bool = True
) -> str:
    label = format_seconds(start_seconds)
    if not linked:
        return label
    return f"[{label}]({timestamp_url(video_id, start_seconds)})"


def escape_html_text(value: object) -> str:
    return escape(str(value), quote=True)


def clean_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def encode_untrusted_json(value: object) -> str:
    """Serialize model input as JSON without emitting markup delimiters."""
    encoded = json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    return (
        encoded.replace("<", r"\u003c").replace(">", r"\u003e").replace("&", r"\u0026")
    )


def sanitize_untrusted_markdown(value: str) -> str:
    """Keep formatting while disabling links, images, and raw HTML from input."""
    sanitized = str(value)
    previous = None
    while sanitized != previous:
        previous = sanitized
        sanitized = MARKDOWN_IMAGE_RE.sub(r"\1", sanitized)
        sanitized = MARKDOWN_LINK_RE.sub(r"\1", sanitized)
        sanitized = MARKDOWN_REFERENCE_USE_RE.sub(r"\1", sanitized)
        sanitized = MARKDOWN_REFERENCE_DEF_RE.sub("[external link removed]", sanitized)
    sanitized = BARE_URL_RE.sub("[external link removed]", sanitized)
    return sanitized.replace("<", "&lt;").replace(">", "&gt;")


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in WORD_RE.finditer(text)]


def build_passages(
    segments: Sequence[TranscriptSegment],
    target_chars: int = 320,
) -> list[Passage]:
    if target_chars <= 0:
        raise ValueError("target_chars must be positive")

    passages: list[Passage] = []
    current_text: list[str] = []
    current_start: float | None = None
    current_end = 0.0

    def flush() -> None:
        nonlocal current_start, current_end, current_text
        if current_text and current_start is not None:
            passages.append(
                Passage(
                    text=clean_whitespace(" ".join(current_text)),
                    start=current_start,
                    end=current_end,
                )
            )
        current_text = []
        current_start = None

    def split_text(text: str) -> list[str]:
        parts: list[str] = []
        while len(text) > target_chars:
            cut = text.rfind(" ", 0, target_chars + 1)
            cut = cut if cut > 0 else target_chars
            parts.append(text[:cut].rstrip())
            text = text[cut:].lstrip()
        if text:
            parts.append(text)
        return parts

    for segment in segments:
        snippet = clean_whitespace(segment.text)
        if not snippet:
            continue

        # A malformed or unusually long caption cue must not become one
        # enormous passage that is repeated in every fallback section.
        if len(snippet) > target_chars:
            flush()
            for part in split_text(snippet):
                passages.append(
                    Passage(text=part, start=segment.start, end=segment.end)
                )
            continue

        if current_text and len(" ".join((*current_text, snippet))) > target_chars:
            flush()
        if current_start is None:
            current_start = segment.start
        current_text.append(snippet)
        current_end = segment.end
        joined = " ".join(current_text)
        if len(joined) >= target_chars or SENTENCE_END_RE.search(snippet):
            flush()

    flush()

    return passages


def keyword_frequencies(texts: Iterable[str], limit: int = 16) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for text in texts:
        for token in tokenize(text):
            if token in STOPWORDS or token.isnumeric():
                continue
            counter[token] += 1
    return counter.most_common(limit)


def rank_passages(passages: Sequence[Passage]) -> list[tuple[Passage, float]]:
    frequencies = dict(
        keyword_frequencies([passage.text for passage in passages], limit=200)
    )
    ranked: list[tuple[Passage, float]] = []
    for passage in passages:
        tokens = [token for token in tokenize(passage.text) if token not in STOPWORDS]
        if not tokens:
            continue
        score = sum(frequencies.get(token, 0) for token in tokens) / math.sqrt(
            len(tokens)
        )
        ranked.append((passage, score))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def select_key_passages(passages: Sequence[Passage], limit: int = 5) -> list[Passage]:
    if limit <= 0 or not passages:
        return []
    if len(passages) <= limit:
        return list(passages)

    ranked = rank_passages(passages)
    chosen: list[Passage] = []
    span = max(0.0, passages[-1].start - passages[0].start)
    # Keep salient passages spread across the source.  A fixed gap makes a
    # short lesson collapse to its opening and closing lines, hiding the
    # middle of the explanation from the generated summary.
    min_gap = max(1.0, span / max(limit - 1, 1))
    for passage, _score in ranked:
        if any(abs(existing.start - passage.start) < min_gap for existing in chosen):
            continue
        chosen.append(passage)
        if len(chosen) >= limit:
            break
    if len(chosen) < limit:
        for passage, _score in ranked:
            if passage in chosen:
                continue
            chosen.append(passage)
            if len(chosen) >= limit:
                break
    if not chosen:
        return list(passages[:limit])
    return sorted(chosen, key=lambda passage: passage.start)


def build_chunked_text(
    segments: Sequence[TranscriptSegment],
    max_chars: int = 10000,
    *,
    include_timestamps: bool = True,
) -> list[str]:
    chunks: list[str] = []
    current_lines: list[str] = []
    current_length = 0

    for segment in segments:
        text = clean_whitespace(segment.text)
        if not text:
            continue
        prefix = f"[{format_seconds(segment.start)}] " if include_timestamps else ""
        available = max_chars - len(prefix)
        if available <= 0:
            raise ValueError("max_chars must leave room for a transcript segment")
        while text:
            if len(text) <= available:
                part = text
                text = ""
            else:
                cut = text.rfind(" ", 0, available + 1)
                cut = cut if cut > 0 else available
                part = text[:cut].rstrip()
                text = text[cut:].lstrip()
            line = prefix + part
            if current_lines and current_length + len(line) + 1 > max_chars:
                chunks.append("\n".join(current_lines))
                current_lines = []
                current_length = 0
            current_lines.append(line)
            current_length += len(line) + 1

    if current_lines:
        chunks.append("\n".join(current_lines))

    return chunks


def pick_keyword_candidates(
    text: str, fallback_pool: Sequence[str]
) -> tuple[str, list[str]]:
    tokens = [token for token in tokenize(text) if token not in STOPWORDS]
    counts = Counter(token for token in tokens if len(token) >= 4)
    if counts:
        answer = counts.most_common(1)[0][0]
    elif tokens:
        answer = tokens[0]
    else:
        answer = "concept"

    distractors = [candidate for candidate in fallback_pool if candidate != answer]
    unique_distractors: list[str] = []
    for candidate in distractors:
        if candidate not in unique_distractors:
            unique_distractors.append(candidate)
        if len(unique_distractors) >= 3:
            break
    while len(unique_distractors) < 3:
        unique_distractors.append(f"option{len(unique_distractors) + 1}")
    return answer, unique_distractors


def blank_keyword(text: str, keyword: str) -> str:
    pattern = re.compile(rf"\b{re.escape(keyword)}\b", re.IGNORECASE)
    return pattern.sub("_____", text, count=1)


def stable_shuffle(values: list[str], seed_text: str) -> list[str]:
    randomizer = random.Random(seed_text)
    shuffled = list(values)
    randomizer.shuffle(shuffled)
    return shuffled
