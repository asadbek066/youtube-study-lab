from __future__ import annotations

import json
import logging
import re
from typing import Any

from google import genai
from google.genai import types as genai_types
from openai import OpenAI

from youtube_study_tool.classification import (
    CLASSIFIER_PROMPT,
    build_classification_prompt,
    heuristic_classification,
    parse_classification_json,
)
from youtube_study_tool.fallback import generate_fallback_bundle
from youtube_study_tool.models import AnalysisBundle, TranscriptBundle, VideoClassification
from youtube_study_tool.settings import LLMSettings, load_settings
from youtube_study_tool.utils import build_chunked_text, format_seconds

SECTION_RE = re.compile(r"<(?P<name>summary|study_notes|quiz)>\s*(?P<body>.*?)\s*</\1>", re.DOTALL)
logger = logging.getLogger(__name__)

LEARNING_ASSISTANT_PROMPT = """
Role: transcript editor for study materials.

Responsibilities:
1. analyze transcript content
2. generate accurate summaries
3. generate study notes
4. generate quizzes
5. adapt output style to transcript type

Rules:
- Be accurate and do not invent details.
- Preserve important meaning, examples, and steps.
- Remove filler, repetition, sponsor talk, and low-value transitions.
- Prefer clarity and completeness over style.
- For educational content, focus on teachable structure.
- For motivational content, focus on practical lessons and mindset principles.
- For tutorial content, preserve step order and dependencies.
- If context is long, compress before final generation.
- Always return requested structure.
- Keep output useful for revision and active recall.
""".strip()

CHUNK_PROMPT = """
{base_instructions}

You are helping a student learn from a YouTube transcript excerpt.
Summarize only what appears in the excerpt below.

Requirements:
- Capture the main point, supporting details, and examples/definitions when present.
- Keep it to 5-7 bullet points.
- Skip filler, repetition, and low-value transitions.
- Ignore greetings, sponsor talk, and housekeeping unless they materially affect the lesson.
- Do not invent facts.
""".strip().format(base_instructions=LEARNING_ASSISTANT_PROMPT)


class StudyPackGenerator:
    def __init__(self, settings: LLMSettings | None = None) -> None:
        self.settings = settings or load_settings()
        self.client = self._build_client()

    @property
    def provider_label(self) -> str:
        return self.settings.provider_label

    @property
    def model_name(self) -> str:
        return self.settings.active_model

    @property
    def is_ready(self) -> bool:
        return self.client is not None and self.settings.is_ready

    @property
    def status_message(self) -> str:
        return self.settings.status_message

    def generate(self, bundle: TranscriptBundle) -> AnalysisBundle:
        if self.is_ready:
            try:
                return self._generate_with_llm(bundle)
            except Exception:
                logger.exception("LLM generation failed, using fallback bundle.")
                return generate_fallback_bundle(bundle)
        return generate_fallback_bundle(bundle)

    def _build_client(self) -> Any | None:
        if not self.settings.is_ready:
            return None

        if self.settings.provider == "openai":
            kwargs: dict[str, Any] = {"api_key": self.settings.openai_api_key}
            if self.settings.openai_base_url:
                kwargs["base_url"] = self.settings.openai_base_url
            return OpenAI(**kwargs)

        if self.settings.provider == "azure_openai":
            return OpenAI(
                api_key=self.settings.azure_openai_api_key,
                base_url=self.settings.azure_openai_base_url,
            )

        if self.settings.provider == "gemini":
            return genai.Client(api_key=self.settings.gemini_api_key)

        return None

    def _generate_with_llm(self, bundle: TranscriptBundle) -> AnalysisBundle:
        classification = self._classify(bundle)
        chunk_summaries = self._chunk_summaries(bundle)
        source_text = "\n\n".join(
            f"Excerpt {index + 1} summary:\n{summary}" for index, summary in enumerate(chunk_summaries)
        )
        response_text = self._complete(
            prompt=(
                f"Video title: {bundle.video_title or 'Unknown'}\n"
                f"Transcript length: {bundle.word_count} words\n"
                f"Approximate duration: {format_seconds(bundle.duration_seconds)}\n\n"
                f"Classifier result:\n{json.dumps(_classification_to_dict(classification), ensure_ascii=True)}\n\n"
                f"{source_text}"
            ),
            instructions=_build_final_prompt(self.settings, classification),
            max_output_tokens=self.settings.final_max_output_tokens,
            temperature=self.settings.temperature,
        )
        sections = _parse_sections(response_text)
        if not all(sections.values()):
            raise ValueError("The model response did not contain all required sections.")
        return AnalysisBundle(
            summary=sections["summary"],
            study_notes=sections["study_notes"],
            quiz=sections["quiz"],
            provider=self.provider_label,
            model=self.model_name,
            classification=classification,
        )

    def _classify(self, bundle: TranscriptBundle) -> VideoClassification:
        if not self.client:
            return heuristic_classification(bundle)
        response_text = self._complete(
            prompt=build_classification_prompt(bundle),
            instructions=CLASSIFIER_PROMPT,
            max_output_tokens=300,
            temperature=0.1,
        )
        try:
            return parse_classification_json(response_text)
        except Exception:
            logger.warning("Classification parse failed, using heuristic classification.")
            return heuristic_classification(bundle)

    def _chunk_summaries(self, bundle: TranscriptBundle) -> list[str]:
        chunks = build_chunked_text(bundle.segments)
        if len(chunks) == 1:
            return [
                self._complete(
                    chunks[0],
                    instructions=CHUNK_PROMPT,
                    max_output_tokens=self.settings.chunk_max_output_tokens,
                    temperature=min(self.settings.temperature, 0.2),
                )
            ]

        summaries: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            prompt = f"Chunk {index} of {len(chunks)}\n\n{chunk}"
            summaries.append(
                self._complete(
                    prompt,
                    instructions=CHUNK_PROMPT,
                    max_output_tokens=self.settings.chunk_max_output_tokens,
                    temperature=min(self.settings.temperature, 0.2),
                )
            )
        return summaries

    def _complete(
        self,
        prompt: str,
        instructions: str,
        max_output_tokens: int,
        temperature: float,
    ) -> str:
        if not self.client:
            raise RuntimeError("No model client is configured.")

        if self.settings.provider in {"openai", "azure_openai"}:
            return self._complete_openai_family(prompt, instructions, max_output_tokens, temperature)
        if self.settings.provider == "gemini":
            return self._complete_gemini(prompt, instructions, max_output_tokens, temperature)
        raise RuntimeError(f"Unsupported provider: {self.settings.provider}")

    def _complete_openai_family(
        self,
        prompt: str,
        instructions: str,
        max_output_tokens: int,
        temperature: float,
    ) -> str:
        response = self.client.responses.create(
            model=self.model_name,
            instructions=instructions,
            input=prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
        if getattr(response, "output_text", ""):
            return response.output_text.strip()

        fragments: list[str] = []
        for item in getattr(response, "output", []):
            for content in getattr(item, "content", []):
                text = getattr(content, "text", None)
                if text:
                    fragments.append(text)
        return "\n".join(fragments).strip()

    def _complete_gemini(
        self,
        prompt: str,
        instructions: str,
        max_output_tokens: int,
        temperature: float,
    ) -> str:
        config_kwargs: dict[str, Any] = {
            "system_instruction": instructions,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        if self.model_name.startswith("gemini-2.5"):
            config_kwargs["thinking_config"] = genai_types.ThinkingConfig(thinking_budget=0)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=genai_types.GenerateContentConfig(**config_kwargs),
        )
        if getattr(response, "text", ""):
            return response.text.strip()

        fragments: list[str] = []
        for candidate in getattr(response, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", []) or []:
                text = getattr(part, "text", None)
                if text:
                    fragments.append(text)
        return "\n".join(fragments).strip()


def _parse_sections(response_text: str) -> dict[str, str]:
    sections = {"summary": "", "study_notes": "", "quiz": ""}
    for match in SECTION_RE.finditer(response_text):
        sections[match.group("name")] = match.group("body").strip()
    return sections


def _build_final_prompt(settings: LLMSettings, classification: VideoClassification) -> str:
    return f"""
{LEARNING_ASSISTANT_PROMPT}

Build a study pack from the provided transcript evidence.
Use only the evidence in the transcript or chunk summaries provided below.

Goals:
- Create a detailed summary of the transcript.
- Preserve all important ideas while removing repetition, sponsor talk, greetings, and fluff.
- Keep the original logic and sequence of the speaker's explanation.
- Include concrete examples when they genuinely improve understanding.
- Do not oversimplify.
- If the speaker gives steps, preserve them in order.
- If the speaker explains a framework, preserve the full framework.
- If the content is motivational, extract the practical lessons and mindset principles clearly.

Summary mode:
- Requested summary style: {settings.summary_style}
- Requested detail level: {settings.summary_detail}
- Classifier primary type: {classification.video_type}
- Classifier preferred summary style: {classification.best_summary_style}
- Classifier preferred note style: {classification.best_note_style}

Adaptation rules:
- If the transcript is a tutorial, build, walkthrough, recipe, or process video, emphasize the outcome, major steps, tools/prerequisites, decisions, and pitfalls. Do not list every micro-step unless essential.
- If the transcript is a coding walkthrough, emphasize stack, file/code changes, implementation order, debugging moments, and tradeoffs.
- If the transcript is motivational, mindset, or self-improvement content, emphasize the central message, practical actions, mindset shifts, and strongest examples. Avoid repeating the same encouragement in different words.
- If the transcript is an interview, organize the summary by topics and standout answers rather than strict chronology.
- If the transcript is commentary, emphasize the main claims, reasoning, and supporting examples.
- If the transcript is storytelling, preserve the arc: setup, turning points, and resolution.
- If the transcript is a lecture, emphasize the thesis, key concepts, examples, and recap.
- If the transcript is explanatory or educational, emphasize the thesis, core concepts, examples, and what someone should remember after watching.
- If summary style is not obvious or SUMMARY_STYLE is adaptive, infer the best structure from the transcript.

Output rules:
- Keep the wording accurate and study-friendly.
- Call out uncertainty instead of guessing.
- Make the quiz useful for active recall.
- Return exactly three tagged sections:
<summary>...</summary>
<study_notes>...</study_notes>
<quiz>...</quiz>

Formatting:
- Use Markdown inside each tag.
- The summary must start with `## Summary`.
- Inside `<summary>`, use exactly these sections and this order:
  `### 1. Overview`
  `### 2. Main ideas`
  `### 3. Step-by-step breakdown`
  `### 4. Important examples`
  `### 5. Practical takeaways`
  `### 6. One-paragraph compressed version`
- Keep the step-by-step breakdown in the speaker's original sequence.
- If the video is not literally procedural, use that section for the sequence in which the ideas unfold.
- For concise detail: keep the summary tight and selective.
- For balanced detail: cover the main ideas and the most useful supporting details.
- For deep detail: still stay selective, but include richer structure, key examples, and practical nuance.
- The study notes must start with `## Study Notes`.
- Inside `<study_notes>`, use exactly these sections and this order:
  `### 1. Topic`
  `### 2. Key concepts`
  `### 3. Important details`
  `### 4. Examples`
  `### 5. Common mistakes or misconceptions`
  `### 6. What to remember`
- Use concise but complete language.
- Preserve definitions, methods, frameworks, and key examples.
- Organize the notes for revision.
- Use bullet points only where useful, not by default everywhere.
- End with a strong revision-focused `What to remember` section.
- The quiz must start with `## Quiz`.
- Inside `<quiz>`, use exactly these sections and this order:
  `### 1. Multiple-choice questions`
  `### 2. Short-answer questions`
  `### 3. Application-based questions`
- Create exactly:
  10 multiple-choice questions,
  5 short-answer questions,
  3 application-based questions.
- Every question must be answerable from the transcript.
- Include a difficulty label for every question: easy, medium, or hard.
- Include an answer key and an explanation for every answer.
- Vary the difficulty across the full quiz instead of making everything the same level.
""".strip()


def _classification_to_dict(classification: VideoClassification) -> dict[str, object]:
    return {
        "video_type": classification.video_type,
        "confidence": classification.confidence,
        "reason": classification.reason,
        "best_summary_style": classification.best_summary_style,
        "best_note_style": classification.best_note_style,
    }
