from __future__ import annotations

from youtube_study_tool.classification import heuristic_classification
from youtube_study_tool.models import AnalysisBundle, TranscriptBundle
from youtube_study_tool.utils import (
    Passage,
    blank_keyword,
    build_passages,
    format_seconds,
    keyword_frequencies,
    pick_keyword_candidates,
    select_key_passages,
    stable_shuffle,
    timestamp_url,
    tokenize,
)


def generate_fallback_bundle(bundle: TranscriptBundle) -> AnalysisBundle:
    classification = heuristic_classification(bundle)
    passages = build_passages(bundle.segments)
    key_passages = _deduplicate_passages(select_key_passages(passages, limit=5))
    quiz_passages = _quiz_passages(bundle, passages)
    keywords = [keyword for keyword, _count in keyword_frequencies([bundle.transcript_text], limit=12)]
    content_style = _normalize_style_for_fallback(classification.video_type)

    summary = _build_summary(bundle, key_passages, keywords, content_style)
    study_notes = _build_study_notes(bundle, key_passages, keywords)
    quiz = _build_quiz(bundle, quiz_passages, keywords, content_style)
    return AnalysisBundle(
        summary=summary,
        study_notes=study_notes,
        quiz=quiz,
        provider="Heuristic fallback",
        model=f"local-rules/{content_style}",
        classification=classification,
    )


def _build_summary(
    bundle: TranscriptBundle,
    passages: list[Passage],
    keywords: list[str],
    content_style: str,
) -> str:
    lines = ["## Summary"]
    if not passages:
        lines.append("- The transcript was fetched, but there was not enough content to summarize.")
        return "\n".join(lines)

    lines.append("### 1. Overview")
    lines.append(_build_overview(passages))
    lines.append("")

    lines.append("### 2. Main ideas")
    for bullet in _build_main_ideas(passages):
        lines.append(f"- {bullet}")
    lines.append("")

    lines.append("### 3. Step-by-step breakdown")
    for step in _build_step_breakdown(bundle, passages):
        lines.append(f"- {step}")
    lines.append("")

    lines.append("### 4. Important examples")
    for example in _build_important_examples(bundle, passages):
        lines.append(f"- {example}")
    lines.append("")

    lines.append("### 5. Practical takeaways")
    for takeaway in _build_practical_takeaways(content_style, keywords, passages):
        lines.append(f"- {takeaway}")
    lines.append("")

    lines.append("### 6. One-paragraph compressed version")
    lines.append(_build_compressed_paragraph(passages))
    return "\n".join(lines)


def _build_study_notes(
    bundle: TranscriptBundle,
    passages: list[Passage],
    keywords: list[str],
) -> str:
    lines = ["## Study Notes"]

    lines.append("### 1. Topic")
    lines.append(_build_topic(bundle, passages))
    lines.append("")

    lines.append("### 2. Key concepts")
    for concept in _build_key_concepts(passages, keywords):
        lines.append(f"- {concept}")
    lines.append("")

    lines.append("### 3. Important details")
    for detail in _build_important_details(passages):
        lines.append(f"- {detail}")
    lines.append("")

    lines.append("### 4. Examples")
    for example in _build_important_examples(bundle, passages):
        lines.append(f"- {example}")
    lines.append("")

    lines.append("### 5. Common mistakes or misconceptions")
    for item in _build_common_mistakes(passages, keywords):
        lines.append(f"- {item}")
    lines.append("")

    lines.append("### 6. What to remember")
    for item in _build_what_to_remember(passages, keywords):
        lines.append(f"- {item}")

    return "\n".join(lines)


def _checkpoint_segments(bundle: TranscriptBundle, count: int) -> list[tuple[float, str]]:
    if not bundle.segments:
        return []
    stride = max(1, len(bundle.segments) // count)
    checkpoints: list[tuple[float, str]] = []
    for index in range(0, len(bundle.segments), stride):
        segment = bundle.segments[index]
        if not segment.text:
            continue
        checkpoints.append((segment.start, segment.text))
        if len(checkpoints) >= count:
            break
    return checkpoints


def _review_prompts(passages: list[Passage], keywords: list[str]) -> list[str]:
    prompts: list[str] = []
    for passage in passages[:3]:
        clue = passage.text.rstrip(".")
        prompts.append(f"How would you explain this idea in your own words: \"{clue}\"?")
    for keyword in keywords[:3]:
        prompts.append(f"Why is `{keyword}` important in the overall lesson?")
    return prompts[:6] or ["What is the central lesson of this video?"]


def _build_topic(bundle: TranscriptBundle, passages: list[Passage]) -> str:
    if bundle.video_title:
        return bundle.video_title
    if passages:
        return passages[0].text
    return "The transcript did not contain enough content to identify the topic clearly."


def _build_key_concepts(passages: list[Passage], keywords: list[str]) -> list[str]:
    concepts: list[str] = []
    for passage in passages[:3]:
        concepts.append(passage.text)
    for keyword in keywords[:3]:
        concepts.append(f"`{keyword}` is one of the repeated concepts anchoring the lesson.")
    return concepts[:5] or ["No strong concepts could be extracted from the transcript."]


def _build_important_details(passages: list[Passage]) -> list[str]:
    if not passages:
        return ["The transcript did not contain enough detail for reliable study notes."]
    details: list[str] = []
    for passage in passages[:4]:
        details.append(passage.text)
    return details


def _build_overview(passages: list[Passage]) -> str:
    selected = [passage.text.rstrip(".") for passage in passages[:2]]
    if not selected:
        return "The video contains too little transcript text for a reliable overview."
    if len(selected) == 1:
        return f"The main idea is: {selected[0]}."
    return f"{selected[0]}. {selected[1]}."


def _motivational_takeaways(keywords: list[str]) -> list[str]:
    anchors = keywords[:3] or ["discipline", "action", "focus"]
    return [
        f"Translate `{anchors[0]}` into one concrete habit you can repeat this week.",
        f"Notice where `{anchors[1]}` shows up as a practical rather than purely emotional idea.",
        f"Use `{anchors[2]}` as a prompt for self-reflection after the video.",
    ]


def _build_main_ideas(passages: list[Passage]) -> list[str]:
    return [passage.text for passage in passages[:4]] or ["The transcript did not contain enough material to identify the main ideas."]


def _build_step_breakdown(bundle: TranscriptBundle, passages: list[Passage]) -> list[str]:
    steps: list[str] = []
    for passage in passages[:4]:
        steps.append(f"[{format_seconds(passage.start)}]({timestamp_url(bundle.video_id, passage.start)}) {passage.text}")
    return steps or ["The transcript did not provide enough sequential detail to map the breakdown."]


def _build_important_examples(bundle: TranscriptBundle, passages: list[Passage]) -> list[str]:
    markers = ("for example", "for instance", "imagine", "like", "such as", "let's say")
    examples: list[str] = []
    for passage in passages:
        lower_text = passage.text.lower()
        if any(marker in lower_text for marker in markers):
            examples.append(passage.text)
    if not examples:
        for passage in passages[1:3]:
            examples.append(
                f"[{format_seconds(passage.start)}]({timestamp_url(bundle.video_id, passage.start)}) {passage.text}"
            )
    return examples[:3] or ["No clear standalone examples appeared in the transcript."]


def _build_common_mistakes(passages: list[Passage], keywords: list[str]) -> list[str]:
    if not passages:
        return ["No reliable misconceptions could be inferred from the transcript."]

    anchors = keywords[:3] or ["main idea", "sequence", "example"]
    mistakes = [
        f"Do not confuse the main point with side comments or repeated phrasing around `{anchors[0]}`.",
        f"Avoid skipping the sequence of ideas, especially around `{anchors[1]}`.",
        f"Do not treat a single example involving `{anchors[2]}` as the whole concept.",
    ]
    if any("not" in passage.text.lower() or "avoid" in passage.text.lower() for passage in passages):
        mistakes.append("Pay attention to places where the speaker explicitly warns against an approach or misunderstanding.")
    return mistakes[:4]


def _build_practical_takeaways(
    content_style: str,
    keywords: list[str],
    passages: list[Passage],
) -> list[str]:
    if content_style == "motivational":
        return _motivational_takeaways(keywords)
    if content_style == "tutorial":
        anchors = keywords[:3] or ["setup", "build", "review"]
        return [
            f"Use `{anchors[0]}` as the first checkpoint before moving to later steps.",
            f"Treat `{anchors[1]}` as a core implementation theme to revisit while practicing.",
            f"Review `{anchors[2]}` after finishing the process so the sequence stays clear.",
        ]
    anchors = keywords[:3] or ["main idea", "example", "sequence"]
    takeaways = [
        f"Remember `{anchors[0]}` as one of the core concepts repeated across the transcript.",
        f"Use `{anchors[1]}` to explain the topic back in your own words.",
        f"Preserve the sequence around `{anchors[2]}` when reviewing the lesson.",
    ]
    if passages:
        takeaways.append(f"Revisit the early explanation first, then follow the later supporting details in order.")
    return takeaways[:4]


def _build_what_to_remember(passages: list[Passage], keywords: list[str]) -> list[str]:
    reminders: list[str] = []
    if passages:
        reminders.append("Keep the speaker's original sequence in mind when revising the topic.")
        reminders.append(passages[0].text)
    for keyword in keywords[:3]:
        reminders.append(f"Remember `{keyword}` as one of the key anchors in the transcript.")
    return reminders[:5] or ["Remember the core topic and the order in which the explanation develops."]


def _build_compressed_paragraph(passages: list[Passage]) -> str:
    text = " ".join(passage.text.rstrip(".") for passage in passages[:3]).strip()
    if not text:
        return "The transcript was too short to compress into a reliable paragraph."
    return f"{text}."


def _deduplicate_passages(passages: list[Passage]) -> list[Passage]:
    unique: list[Passage] = []
    seen_signatures: set[str] = set()
    for passage in passages:
        signature = " ".join(tokenize(passage.text.lower())[:20])
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        unique.append(passage)
    return unique


def _quiz_passages(bundle: TranscriptBundle, passages: list[Passage]) -> list[Passage]:
    if len(passages) >= 3:
        return passages
    segment_passages = [
        Passage(text=segment.text, start=segment.start, end=segment.end)
        for segment in bundle.segments
        if segment.text
    ]
    return segment_passages or passages


def _normalize_style_for_fallback(video_type: str) -> str:
    if video_type in {"tutorial", "coding walkthrough"}:
        return "tutorial"
    if video_type == "motivational":
        return "motivational"
    return "general"


def _build_quiz(
    bundle: TranscriptBundle,
    passages: list[Passage],
    keywords: list[str],
    content_style: str,
) -> str:
    lines = ["## Quiz"]
    if not passages:
        lines.append("The transcript was too short to generate a reliable quiz.")
        return "\n".join(lines)

    fallback_pool = keywords or [token for token in tokenize(bundle.transcript_text) if len(token) >= 4][:12]
    lines.append("### 1. Multiple-choice questions")
    for block in _build_multiple_choice_questions(bundle, passages, fallback_pool):
        lines.extend(block)
        lines.append("")

    lines.append("### 2. Short-answer questions")
    for block in _build_short_answer_questions(bundle, passages, keywords):
        lines.extend(block)
        lines.append("")

    lines.append("### 3. Application-based questions")
    for block in _build_application_questions(bundle, passages, keywords, content_style):
        lines.extend(block)
        lines.append("")

    return "\n".join(lines).rstrip()


def _build_multiple_choice_questions(
    bundle: TranscriptBundle,
    passages: list[Passage],
    fallback_pool: list[str],
) -> list[list[str]]:
    questions: list[list[str]] = []
    for index in range(10):
        question_number = index + 1
        difficulty = _difficulty_label(index)
        passage = passages[index % len(passages)]
        answer, distractors = pick_keyword_candidates(passage.text, fallback_pool)
        options = stable_shuffle([answer, *distractors], f"{bundle.video_id}-mcq-{question_number}")
        answer_letter = "ABCD"[options.index(answer)]

        if index % 3 == 0:
            stem = blank_keyword(passage.text, answer)
            prompt = f"{question_number}. [{difficulty}] Which option best completes this transcript statement?"
        elif index % 3 == 1:
            stem = passage.text
            prompt = f"{question_number}. [{difficulty}] Which concept is emphasized in this transcript segment?"
        else:
            stem = f"Focus segment at {format_seconds(passage.start)}: {passage.text}"
            prompt = f"{question_number}. [{difficulty}] Which keyword best matches the main focus here?"

        block = [prompt, stem]
        for option_index, option in enumerate(options):
            block.append(f"{'ABCD'[option_index]}. {option}")
        block.append(f"Answer: {answer_letter}. {answer}")
        block.append(
            f"Explanation: Around {format_seconds(passage.start)}, the transcript directly centers on `{answer}`, which makes it the best supported answer."
        )
        questions.append(block)
    return questions


def _build_short_answer_questions(
    bundle: TranscriptBundle,
    passages: list[Passage],
    keywords: list[str],
) -> list[list[str]]:
    anchors = keywords[:3] or ["main idea", "supporting detail", "sequence"]
    details = passages[:3]
    compressed = _build_compressed_paragraph(passages)
    misconceptions = _build_common_mistakes(passages, keywords)

    prompts_and_answers = [
        (
            "11. [easy] What is the main topic of the transcript?",
            _build_topic(bundle, passages),
            "The title/opening of the transcript identifies this as the central subject.",
        ),
        (
            f"12. [medium] Name one key concept the speaker emphasizes.",
            anchors[0],
            f"`{anchors[0]}` appears repeatedly and acts as one of the main anchors of the transcript.",
        ),
        (
            f"13. [medium] What important detail does the speaker give about the topic?",
            details[0].text if details else _build_topic(bundle, passages),
            "This detail is stated directly in one of the early high-signal transcript passages.",
        ),
        (
            "14. [hard] How does the transcript progress from the opening idea to later points?",
            compressed,
            "This answer preserves the sequence of the strongest early-to-mid transcript points.",
        ),
        (
            "15. [hard] What is one mistake or misconception a learner should avoid?",
            misconceptions[0],
            "This is supported by the transcript's emphasis and by the way the explanation frames the topic.",
        ),
    ]

    blocks: list[list[str]] = []
    for prompt, answer, explanation in prompts_and_answers:
        blocks.append([prompt, f"Answer: {answer}", f"Explanation: {explanation}"])
    return blocks


def _build_application_questions(
    bundle: TranscriptBundle,
    passages: list[Passage],
    keywords: list[str],
    content_style: str,
) -> list[list[str]]:
    anchors = keywords[:3] or ["core idea", "sequence", "example"]
    first_step = passages[0].text if passages else _build_topic(bundle, passages)
    later_step = passages[1].text if len(passages) > 1 else first_step

    if content_style == "tutorial":
        prompts_and_answers = [
            (
                "16. [medium] If you were following this process yourself, what should you focus on first?",
                first_step,
                "The earliest step in the transcript is the safest first action because the speaker presents it before later stages.",
            ),
            (
                "17. [hard] If you got stuck halfway through, how would the transcript suggest getting back on track?",
                later_step,
                "The transcript gives a clear sequence, so returning to the next major step is the most supported move.",
            ),
            (
                "18. [hard] How could you apply the same method to a similar project?",
                f"Reuse the sequence built around `{anchors[0]}` and keep the same overall order of steps.",
                "The transcript emphasizes the method and ordering more than a one-off isolated detail.",
            ),
        ]
    elif content_style == "motivational":
        prompts_and_answers = [
            (
                "16. [medium] If someone wanted to apply the message of the transcript this week, what should they start with?",
                f"Start with the principle around `{anchors[0]}` and turn it into one repeatable action.",
                "The transcript frames this idea as a practical anchor rather than just an abstract slogan.",
            ),
            (
                "17. [hard] How would you use the transcript's mindset advice when motivation is low?",
                f"Return to the lesson around `{anchors[1]}` and use it as the first mental reset before taking action.",
                "This follows the transcript's emphasis on principle-first action.",
            ),
            (
                "18. [hard] How could you explain the transcript's message to a friend in a real-life situation?",
                f"Use `{anchors[2]}` as the key principle and connect it to the transcript's practical lesson.",
                "The transcript's strongest motivational value comes from applying the principle, not repeating the wording.",
            ),
        ]
    else:
        prompts_and_answers = [
            (
                "16. [medium] If you had to teach this topic to a beginner, what would you explain first?",
                first_step,
                "The transcript introduces this point early, which makes it the most grounded place to begin.",
            ),
            (
                "17. [hard] If you were reviewing this topic under time pressure, what idea would you prioritize?",
                f"Prioritize `{anchors[0]}` because it is one of the recurring anchors across the transcript.",
                "Repeated emphasis is a strong signal that this idea matters most for review.",
            ),
            (
                "18. [hard] How would you apply one principle from the transcript to a related example?",
                f"Apply the sequence or concept around `{anchors[1]}` and keep the same logic used by the speaker.",
                "This stays answerable from the transcript because it uses the speaker's own framework or sequence.",
            ),
        ]

    blocks: list[list[str]] = []
    for prompt, answer, explanation in prompts_and_answers:
        blocks.append([prompt, f"Answer: {answer}", f"Explanation: {explanation}"])
    return blocks


def _difficulty_label(index: int) -> str:
    pattern = ("easy", "easy", "medium", "medium", "hard", "easy", "medium", "hard", "medium", "hard")
    return pattern[index % len(pattern)]
