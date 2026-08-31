from __future__ import annotations

from youtube_study_tool.models import (
    MAX_TRANSCRIPT_CHARS,
    TranscriptBundle,
    TranscriptSegment,
)
from youtube_study_tool.utils import clean_whitespace


def build_manual_transcript(text: str, *, title: str = "") -> TranscriptBundle:
    transcript_text = clean_whitespace(text)
    if not transcript_text:
        raise ValueError("Paste transcript text to build a study pack.")
    if len(transcript_text) > MAX_TRANSCRIPT_CHARS:
        raise ValueError(
            f"Transcript is too long; keep it under {MAX_TRANSCRIPT_CHARS:,} characters."
        )

    normalized_title = clean_whitespace(title) or "Pasted transcript"
    segment = TranscriptSegment(text=transcript_text, start=0.0, duration=0.0)
    return TranscriptBundle(
        video_id="pasted-transcript",
        source_url="",
        transcript_text=transcript_text,
        segments=(segment,),
        language_code="und",
        language_name="User-provided text",
        is_generated=False,
        duration_seconds=0.0,
        word_count=len(transcript_text.split()),
        video_title=normalized_title,
        source_label="Pasted transcript",
    )
