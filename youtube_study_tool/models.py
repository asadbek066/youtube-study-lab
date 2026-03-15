from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TranscriptSegment:
    text: str
    start: float
    duration: float

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass(frozen=True)
class TranscriptBundle:
    video_id: str
    source_url: str
    transcript_text: str
    segments: tuple[TranscriptSegment, ...]
    language_code: str
    language_name: str
    is_generated: bool
    duration_seconds: float
    word_count: int
    video_title: str | None = None


@dataclass(frozen=True)
class VideoClassification:
    video_type: str
    confidence: float
    reason: str
    best_summary_style: str
    best_note_style: str


@dataclass(frozen=True)
class AnalysisBundle:
    summary: str
    study_notes: str
    quiz: str
    provider: str
    model: str
    classification: VideoClassification
