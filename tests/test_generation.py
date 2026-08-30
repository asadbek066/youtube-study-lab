import re

from youtube_study_tool.classification import heuristic_classification
from youtube_study_tool.demo import build_demo_transcript
from youtube_study_tool.generation import CHUNK_PROMPT, StudyPackGenerator
from youtube_study_tool.manual import build_manual_transcript


def _capture_llm_prompts(bundle, monkeypatch) -> list[str]:
    generator = StudyPackGenerator()
    generator.client = object()
    prompts: list[str] = []

    monkeypatch.setattr(
        generator, "_classify", lambda _: heuristic_classification(bundle)
    )

    def fake_complete(prompt: str, instructions: str, **_kwargs) -> str:
        prompts.append(prompt)
        if instructions == CHUNK_PROMPT:
            return "A transcript-grounded chunk summary."
        return "<summary>Summary</summary><study_notes>Notes</study_notes><quiz>Quiz</quiz>"

    monkeypatch.setattr(generator, "_complete", fake_complete)
    generator._generate_with_llm(bundle)
    return prompts


def test_pasted_transcript_never_sends_a_timecode_to_llm(monkeypatch) -> None:
    bundle = build_manual_transcript(
        "A model predicts an answer. The loss measures error. Gradient descent updates weights."
    )

    prompts = _capture_llm_prompts(bundle, monkeypatch)

    assert prompts
    assert all(
        not re.search(r"\b\d{2}:\d{2}(?::\d{2})?\b", prompt) for prompt in prompts
    )
    assert all("Approximate duration:" not in prompt for prompt in prompts)


def test_timed_video_retains_timecodes_in_llm_prompts(monkeypatch) -> None:
    prompts = _capture_llm_prompts(build_demo_transcript(), monkeypatch)

    assert any("[00:00]" in prompt for prompt in prompts)
    assert any("Approximate duration:" in prompt for prompt in prompts)
