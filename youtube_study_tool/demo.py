from __future__ import annotations

from youtube_study_tool.models import TranscriptBundle, TranscriptSegment

_DEMO_SEGMENTS = (
    TranscriptSegment(
        text="A neural network learns by looking at examples and adjusting small numerical weights inside the model.",
        start=0.0,
        duration=8.0,
    ),
    TranscriptSegment(
        text="First, the network receives an input, such as the pixels of a handwritten digit, and produces a prediction.",
        start=8.0,
        duration=9.0,
    ),
    TranscriptSegment(
        text="A loss function compares that prediction with the correct answer and turns the mistake into a single score.",
        start=17.0,
        duration=9.0,
    ),
    TranscriptSegment(
        text="Backpropagation works backward through the network to calculate which weights contributed to the error.",
        start=26.0,
        duration=9.0,
    ),
    TranscriptSegment(
        text="Gradient descent then nudges those weights in the direction that should reduce the next prediction error.",
        start=35.0,
        duration=9.0,
    ),
    TranscriptSegment(
        text="For example, repeated images of threes help the model discover useful patterns such as curves and connected strokes.",
        start=44.0,
        duration=10.0,
    ),
    TranscriptSegment(
        text="Training repeats this predict, measure, and update loop across many examples and multiple passes through the dataset.",
        start=54.0,
        duration=10.0,
    ),
    TranscriptSegment(
        text="The important lesson is that the network is not given a handwritten rule for every digit; it improves its internal representation from feedback.",
        start=64.0,
        duration=11.0,
    ),
)


def build_demo_transcript() -> TranscriptBundle:
    """Return a deterministic, network-free transcript for the instant demo."""
    transcript_text = " ".join(segment.text for segment in _DEMO_SEGMENTS)
    return TranscriptBundle(
        video_id="study-demo",
        source_url="",
        transcript_text=transcript_text,
        segments=_DEMO_SEGMENTS,
        language_code="en",
        language_name="English",
        is_generated=False,
        duration_seconds=_DEMO_SEGMENTS[-1].end,
        word_count=len(transcript_text.split()),
        video_title="How neural networks learn from examples",
        source_label="Built-in sample",
    )
