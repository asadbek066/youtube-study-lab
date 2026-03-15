from __future__ import annotations

import json
import math
import re
from collections import Counter

from youtube_study_tool.models import TranscriptBundle, VideoClassification
from youtube_study_tool.utils import tokenize

VIDEO_TYPES = (
    "tutorial",
    "lecture",
    "motivational",
    "interview",
    "commentary",
    "storytelling",
    "coding walkthrough",
)

CLASSIFIER_PROMPT = """
You are analyzing a YouTube transcript.

Task:
Classify the transcript into one primary type:
- tutorial
- lecture
- motivational
- interview
- commentary
- storytelling
- coding walkthrough

Return JSON only:
{
  "video_type": "...",
  "confidence": 0.0,
  "reason": "...",
  "best_summary_style": "...",
  "best_note_style": "..."
}
""".strip()

STYLE_GUIDANCE = {
    "tutorial": (
        "condensed step-by-step summary with outcome, major steps, tools, and pitfalls",
        "procedural notes with prerequisites, steps, checkpoints, and mistakes to avoid",
    ),
    "lecture": (
        "concept-first summary with thesis, key ideas, examples, and recap",
        "structured study notes with definitions, frameworks, and review questions",
    ),
    "motivational": (
        "message-first summary with central theme, mindset shifts, and practical actions",
        "reflective notes with takeaways, action prompts, and memorable lines",
    ),
    "interview": (
        "topic-grouped summary with strongest answers, opinions, and themes",
        "speaker-based notes with topics, insights, and follow-up questions",
    ),
    "commentary": (
        "argument-focused summary with claims, evidence, and conclusions",
        "analytical notes with viewpoints, supporting points, and counterpoints",
    ),
    "storytelling": (
        "arc-based summary covering setup, turning points, conflict, and resolution",
        "narrative notes with characters, events, themes, and memorable moments",
    ),
    "coding walkthrough": (
        "build-log summary with stack, code changes, decisions, and debugging moments",
        "developer notes with code structure, commands, implementation choices, and gotchas",
    ),
}

KEYWORDS = {
    "tutorial": {
        "tutorial": 3.0,
        "how": 1.0,
        "guide": 2.0,
        "step": 2.0,
        "setup": 2.0,
        "install": 2.0,
        "workflow": 2.0,
        "process": 2.0,
        "tool": 1.0,
        "first": 1.0,
        "next": 1.0,
        "finally": 1.0,
        "build": 1.5,
        "create": 1.0,
        "make": 1.0,
    },
    "lecture": {
        "lecture": 3.0,
        "class": 2.0,
        "course": 2.0,
        "students": 1.5,
        "concept": 1.5,
        "theory": 2.0,
        "theorem": 2.0,
        "chapter": 1.5,
        "topic": 1.0,
        "explain": 1.5,
        "understand": 1.0,
        "framework": 1.0,
    },
    "motivational": {
        "motivation": 3.0,
        "motivational": 3.0,
        "mindset": 2.5,
        "discipline": 2.0,
        "success": 1.5,
        "dream": 1.5,
        "fear": 1.0,
        "believe": 1.5,
        "focus": 1.0,
        "life": 1.0,
        "excuses": 1.5,
        "comfort": 1.0,
        "goals": 1.0,
    },
    "interview": {
        "interview": 3.0,
        "host": 2.0,
        "guest": 2.0,
        "question": 1.5,
        "answer": 1.5,
        "asked": 1.0,
        "conversation": 1.0,
        "podcast": 1.5,
        "tell": 0.8,
    },
    "commentary": {
        "commentary": 3.0,
        "reaction": 2.0,
        "analysis": 1.5,
        "opinion": 1.5,
        "review": 1.0,
        "news": 1.0,
        "discuss": 1.0,
        "happened": 1.0,
        "breakdown": 1.5,
        "take": 1.0,
    },
    "storytelling": {
        "story": 3.0,
        "remember": 1.0,
        "happened": 1.0,
        "journey": 1.5,
        "character": 2.0,
        "once": 1.0,
        "then": 0.8,
        "after": 0.5,
        "suddenly": 1.0,
        "finally": 0.8,
    },
    "coding walkthrough": {
        "code": 3.0,
        "coding": 3.0,
        "function": 2.0,
        "variable": 2.0,
        "class": 1.5,
        "api": 2.0,
        "component": 2.0,
        "terminal": 2.0,
        "repository": 1.5,
        "repo": 1.5,
        "python": 2.0,
        "javascript": 2.0,
        "typescript": 2.0,
        "react": 2.0,
        "debug": 1.5,
        "bug": 1.5,
        "deploy": 1.5,
        "npm": 2.0,
        "package": 1.0,
        "database": 1.5,
    },
}


def build_classification_prompt(bundle: TranscriptBundle, max_chars: int = 12000) -> str:
    transcript_excerpt = bundle.transcript_text[:max_chars]
    return (
        f"Video title: {bundle.video_title or 'Unknown'}\n"
        f"Transcript language: {bundle.language_name}\n"
        f"Approximate transcript length: {bundle.word_count} words\n\n"
        f"Transcript excerpt:\n{transcript_excerpt}"
    )


def parse_classification_json(raw_text: str) -> VideoClassification:
    candidate = raw_text.strip()
    if candidate.startswith("```"):
        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", candidate, re.DOTALL)
        if fenced:
            candidate = fenced.group(1)
    if not candidate.startswith("{"):
        object_match = re.search(r"\{.*\}", candidate, re.DOTALL)
        if object_match:
            candidate = object_match.group(0)

    data = json.loads(candidate)
    return normalize_classification(data)


def normalize_classification(data: dict[str, object]) -> VideoClassification:
    video_type = str(data.get("video_type", "")).strip().lower()
    if video_type not in VIDEO_TYPES:
        raise ValueError(f"Unsupported video_type: {video_type!r}")

    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError) as error:
        raise ValueError("confidence must be numeric") from error

    reason = str(data.get("reason", "")).strip()
    best_summary_style = str(data.get("best_summary_style", "")).strip()
    best_note_style = str(data.get("best_note_style", "")).strip()
    if not reason or not best_summary_style or not best_note_style:
        raise ValueError("classification response is missing required fields")

    return VideoClassification(
        video_type=video_type,
        confidence=max(0.0, min(1.0, confidence)),
        reason=reason,
        best_summary_style=best_summary_style,
        best_note_style=best_note_style,
    )


def heuristic_classification(bundle: TranscriptBundle) -> VideoClassification:
    text = f"{bundle.video_title or ''} {bundle.transcript_text}".lower()
    token_counts = Counter(tokenize(text))
    scores: dict[str, float] = {}

    for video_type, weights in KEYWORDS.items():
        score = 0.0
        for keyword, weight in weights.items():
            if " " in keyword:
                score += text.count(keyword) * weight
            else:
                score += token_counts.get(keyword, 0) * weight
        scores[video_type] = score

    # Distinguish tutorial from coding walkthrough when code-heavy language is present.
    if scores["coding walkthrough"] >= 4:
        scores["tutorial"] *= 0.7
    if scores["interview"] >= 3 and text.count("?") >= 2:
        scores["interview"] += 1.5
    if scores["storytelling"] >= 2 and any(token in token_counts for token in {"i", "we", "he", "she"}):
        scores["storytelling"] += 0.5

    top_type, top_score = max(scores.items(), key=lambda item: item[1])
    ordered_scores = sorted(scores.values(), reverse=True)
    second_score = ordered_scores[1] if len(ordered_scores) > 1 else 0.0

    if top_score <= 0:
        top_type = "commentary"
        top_score = 1.0
        second_score = 0.5

    margin = top_score - second_score
    confidence = 1 / (1 + math.exp(-(margin + top_score * 0.15)))
    confidence = round(max(0.35, min(0.96, confidence)), 2)

    summary_style, note_style = STYLE_GUIDANCE[top_type]
    reason = _build_reason(top_type, scores, token_counts)
    return VideoClassification(
        video_type=top_type,
        confidence=confidence,
        reason=reason,
        best_summary_style=summary_style,
        best_note_style=note_style,
    )


def _build_reason(video_type: str, scores: dict[str, float], token_counts: Counter[str]) -> str:
    keywords = KEYWORDS[video_type]
    anchors = sorted(
        ((token, token_counts.get(token, 0)) for token in keywords if " " not in token),
        key=lambda item: item[1],
        reverse=True,
    )
    positive = [token for token, count in anchors if count > 0][:3]
    anchor_text = ", ".join(positive) if positive else "overall phrasing and structure"
    return (
        f"Primary signals point to {video_type} based on recurring cues such as {anchor_text}; "
        f"its score was highest against the other candidate types."
    )
