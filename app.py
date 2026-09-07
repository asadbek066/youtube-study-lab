from __future__ import annotations

import logging
from dataclasses import asdict
from textwrap import dedent

import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import (
    CouldNotRetrieveTranscript,
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeTranscriptApiException,
)

from youtube_study_tool.demo import build_demo_transcript
from youtube_study_tool.fallback import generate_fallback_bundle
from youtube_study_tool.generation import StudyPackGenerator
from youtube_study_tool.manual import build_manual_transcript
from youtube_study_tool.models import (
    MAX_TRANSCRIPT_CHARS,
    AnalysisBundle,
    TranscriptBundle,
)
from youtube_study_tool.transcripts import (
    TranscriptRetrievalError,
    TranscriptService,
    normalize_languages,
)
from youtube_study_tool.utils import (
    escape_html_text,
    format_seconds,
    sanitize_untrusted_markdown,
    timestamp_reference,
)

load_dotenv()
logger = logging.getLogger(__name__)

MAX_RENDERED_SEGMENTS = 200
MAX_TRANSCRIPT_PREVIEW_CHARS = 60_000
MAX_PAID_SUBMISSIONS_PER_SESSION = 3

st.set_page_config(
    page_title="YouTube Study Lab",
    page_icon="YT",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Source+Serif+4:wght@400;600;700&display=swap');

    :root {
        --paper: #fcf8ef;
        --ink: #17222d;
        --accent: #9a4f2f;
        --accent-soft: #f5d3b6;
        --card: rgba(255, 255, 255, 0.82);
        --border: rgba(23, 34, 45, 0.1);
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(248, 209, 163, 0.6), transparent 30%),
            radial-gradient(circle at top right, rgba(175, 214, 197, 0.55), transparent 28%),
            linear-gradient(180deg, #fff8eb 0%, #f3efe5 48%, #edf4ef 100%);
        color: var(--ink);
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3 {
        font-family: "Space Grotesk", sans-serif;
        color: var(--ink);
        letter-spacing: -0.02em;
    }

    p, li, label, .stMarkdown, .stTextInput, .stTextArea {
        font-family: "Source Serif 4", serif;
    }

    .hero {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 28px;
        padding: 1.6rem 1.7rem;
        box-shadow: 0 20px 60px rgba(62, 43, 31, 0.08);
        margin-bottom: 1.25rem;
    }

    .hero-kicker {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        background: var(--accent-soft);
        color: var(--ink);
        font-family: "Space Grotesk", sans-serif;
        font-size: 0.84rem;
        margin-bottom: 0.65rem;
    }

    .hero h1 {
        margin-bottom: 0.35rem;
    }

    .hero-copy {
        max-width: 760px;
        font-size: 1.08rem;
        line-height: 1.6;
        margin-bottom: 1rem;
    }

    .trust-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
    }

    .trust-pill {
        background: rgba(23, 34, 45, 0.06);
        border: 1px solid var(--border);
        border-radius: 999px;
        color: var(--ink);
        font-family: "Space Grotesk", sans-serif;
        font-size: 0.78rem;
        font-weight: 500;
        padding: 0.38rem 0.66rem;
    }

    .section-label {
        color: var(--accent);
        font-family: "Space Grotesk", sans-serif;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        margin: 0.2rem 0 0.5rem;
        text-transform: uppercase;
    }

    div.stButton > button[kind="primary"] {
        background: var(--accent);
        border-color: var(--accent);
        color: #ffffff;
        font-family: "Space Grotesk", sans-serif;
        font-weight: 700;
    }

    div.stButton > button[kind="primary"]:hover {
        background: #7f3f26;
        border-color: #7f3f26;
        color: #ffffff;
    }

    .meta-card {
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid var(--border);
        border-radius: 22px;
        padding: 1rem 1.1rem;
        min-height: 100%;
    }

    .meta-label {
        font-family: "Space Grotesk", sans-serif;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.72;
    }

    .meta-value {
        font-family: "Space Grotesk", sans-serif;
        font-size: 1.2rem;
        margin-top: 0.3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_hero() -> None:
    st.markdown(
        """
        <section class="hero">
            <div class="hero-kicker">One link. A complete study pack.</div>
            <h1>YouTube Study Lab</h1>
            <p class="hero-copy">Turn a captioned YouTube video into a structured summary, revision notes, and an active-recall quiz—without scrubbing through the timeline again.</p>
            <div class="trust-row">
                <span class="trust-pill">No API key required</span>
                <span class="trust-pill">Timestamped sources</span>
                <span class="trust-pill">18-question quiz</span>
                <span class="trust-pill">Markdown export</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_meta(bundle: TranscriptBundle, analysis: AnalysisBundle) -> None:
    col1, col2, col3, col4 = st.columns(4)
    cards = [
        ("Transcript", bundle.language_name),
        (
            "Source",
            bundle.source_label
            or ("Auto captions" if bundle.is_generated else "Manual captions"),
        ),
        ("Type", analysis.classification.video_type.title()),
        ("Generator", f"{analysis.provider} ({analysis.model})"),
    ]
    for column, (label, value) in zip((col1, col2, col3, col4), cards):
        safe_label = escape_html_text(label)
        safe_value = escape_html_text(value)
        column.markdown(
            f"""
            <div class="meta-card">
                <div class="meta-label">{safe_label}</div>
                <div class="meta-value">{safe_value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_transcript_tab(bundle: TranscriptBundle) -> None:
    st.download_button(
        label="Download transcript (.txt)",
        data=bundle.transcript_text,
        file_name=f"{bundle.video_id}-transcript.txt",
        mime="text/plain",
        use_container_width=True,
    )
    preview = bundle.transcript_text[:MAX_TRANSCRIPT_PREVIEW_CHARS]
    if len(bundle.transcript_text) > MAX_TRANSCRIPT_PREVIEW_CHARS:
        preview += "\n\n[Preview truncated; download the transcript for the full text.]"
    st.text_area("Transcript preview", value=preview, height=320)
    if bundle.duration_seconds > 0:
        with st.expander("Timestamped transcript"):
            for segment in bundle.segments[:MAX_RENDERED_SEGMENTS]:
                reference = timestamp_reference(
                    bundle.video_id,
                    segment.start,
                    linked=bool(bundle.source_url),
                )
                if bundle.source_url:
                    st.markdown(reference)
                st.text(segment.text)
            if len(bundle.segments) > MAX_RENDERED_SEGMENTS:
                st.caption(
                    f"Showing the first {MAX_RENDERED_SEGMENTS} caption segments. "
                    "Download the transcript for the complete text."
                )
    else:
        st.caption("Timestamps are unavailable for pasted transcript text.")


def render_classification_tab(analysis: AnalysisBundle) -> None:
    classification = analysis.classification
    st.markdown(f"### {classification.video_type.title()}")
    st.markdown(f"**Confidence:** {classification.confidence:.2f}")
    st.text(f"Reason: {classification.reason}")
    st.text(f"Best summary style: {classification.best_summary_style}")
    st.text(f"Best note style: {classification.best_note_style}")
    st.json(asdict(classification))


def compile_study_pack(bundle: TranscriptBundle, analysis: AnalysisBundle) -> str:
    title = sanitize_untrusted_markdown(bundle.video_title or bundle.video_id)
    classification_reason = sanitize_untrusted_markdown(analysis.classification.reason)
    return dedent(
        f"""
        # {title}

        Source: {bundle.source_url or bundle.source_label or "Unknown"}
        Transcript language: {bundle.language_name} ({bundle.language_code})
        Duration: {format_seconds(bundle.duration_seconds)}
        Generated with: {analysis.provider} ({analysis.model})
        Video type: {analysis.classification.video_type} ({analysis.classification.confidence:.2f})
        Classification reason: {classification_reason}

        {sanitize_untrusted_markdown(analysis.summary)}

        {sanitize_untrusted_markdown(analysis.study_notes)}

        {sanitize_untrusted_markdown(analysis.quiz)}
        """
    ).strip()


def generate_study_pack(
    generator: StudyPackGenerator, bundle: TranscriptBundle
) -> AnalysisBundle:
    """Cap paid submissions per Streamlit session and keep the local fallback available."""
    if not generator.is_ready:
        return generator.generate(bundle)
    used = int(st.session_state.get("paid_generation_submissions", 0))
    if used >= MAX_PAID_SUBMISSIONS_PER_SESSION:
        st.info(
            "The provider-session limit has been reached. This pack uses local generation "
            "so repeated submissions cannot create unbounded paid calls."
        )
        return generate_fallback_bundle(bundle)
    st.session_state["paid_generation_submissions"] = used + 1
    return generator.generate(bundle)


def run() -> None:
    render_hero()
    transcript_service = TranscriptService()
    generator = StudyPackGenerator()

    with st.sidebar:
        st.header("Study settings")
        language_input = st.text_input(
            "Preferred transcript languages", value="en,en-US,en-GB"
        )

        if generator.settings.provider == "heuristic":
            st.success(
                "No-key mode ready. Study packs are generated with local rules after transcript retrieval."
            )
        elif generator.is_ready:
            st.success(f"{generator.provider_label} is ready.")
        else:
            st.warning(
                f"{generator.provider_label} is not fully configured; no-key mode will be used."
            )

        with st.expander("Provider details"):
            st.caption(
                "Provider and model are controlled through `.env`; keys are never entered in this interface."
            )
            st.markdown(f"**Provider:** {generator.provider_label}")
            st.markdown(f"**Model/deployment:** `{generator.model_name}`")
            st.markdown(
                f"**Summary profile:** `{generator.settings.summary_style}` / `{generator.settings.summary_detail}`"
            )
            st.caption(generator.status_message)

        st.caption(
            "A public caption track is required when you use your own YouTube link."
        )

    st.markdown('<div class="section-label">Start here</div>', unsafe_allow_html=True)
    demo_requested = st.button(
        "See a complete study pack instantly",
        key="instant-demo",
        use_container_width=True,
        type="primary",
    )
    st.caption(
        "Uses an original sample transcript and local generation—no YouTube request or API key."
    )

    st.markdown(
        '<div class="section-label">Or use your own video</div>', unsafe_allow_html=True
    )
    with st.form("analyze-form"):
        source = st.text_input(
            "YouTube URL or video ID",
            placeholder="https://www.youtube.com/watch?v=...",
        )
        submitted = st.form_submit_button("Build study pack", use_container_width=True)

    with st.expander("YouTube blocked? Paste a transcript instead"):
        st.caption(
            "Useful on hosted servers where YouTube blocks transcript requests. No timestamps are invented."
        )
        with st.form("manual-transcript-form"):
            manual_title = st.text_input(
                "Title",
                placeholder="My lecture notes",
                key="manual-title",
            )
            manual_text = st.text_area(
                "Transcript text",
                placeholder="Paste public or personal transcript text here...",
                height=180,
                max_chars=MAX_TRANSCRIPT_CHARS,
                key="manual-transcript",
            )
            manual_submitted = st.form_submit_button(
                "Build from pasted transcript",
                use_container_width=True,
                key="manual-submit",
            )

    if demo_requested or submitted or manual_submitted:
        # Results belong to the last submitted input. Clear them before any
        # validation or retrieval so a failed submission cannot resurrect an
        # older study pack on the next rerun.
        st.session_state.pop("transcript_bundle", None)
        st.session_state.pop("analysis_bundle", None)

    if demo_requested:
        with st.spinner("Loading the network-free demo..."):
            transcript = build_demo_transcript()
            analysis = generate_fallback_bundle(transcript)
        st.session_state["transcript_bundle"] = transcript
        st.session_state["analysis_bundle"] = analysis
        st.toast("Instant demo ready", icon="✅")
    elif submitted:
        if not source.strip():
            st.warning("Paste a YouTube URL or ID to get started.")
            return

        languages = normalize_languages(language_input)
        try:
            with st.spinner("Pulling transcript from YouTube..."):
                transcript = transcript_service.fetch(source, languages)
            with st.spinner("Building summary, notes, and quiz..."):
                analysis = generate_study_pack(generator, transcript)
        except ValueError as error:
            st.error(str(error))
            return
        except (
            NoTranscriptFound,
            TranscriptsDisabled,
            CouldNotRetrieveTranscript,
            TranscriptRetrievalError,
            YouTubeTranscriptApiException,
        ) as error:
            logger.warning("Transcript extraction failed: %s", error, exc_info=True)
            st.error(
                "Transcript extraction failed. The video may be unavailable, "
                "blocked, or missing a public caption track."
            )
            return
        except Exception:  # keep the UI alive for provider failures.
            logger.exception("Unexpected error while building a YouTube study pack")
            st.error("Could not build the study pack. Please try again.")
            return

        st.session_state["transcript_bundle"] = transcript
        st.session_state["analysis_bundle"] = analysis
    elif manual_submitted:
        try:
            transcript = build_manual_transcript(manual_text, title=manual_title)
            with st.spinner("Building summary, notes, and quiz..."):
                analysis = generate_study_pack(generator, transcript)
        except ValueError as error:
            st.error(str(error))
            return
        except Exception:  # keep the UI alive for generation failures.
            logger.exception("Unexpected error while building a pasted study pack")
            st.error("Could not build the study pack. Please try again.")
            return

        st.session_state["transcript_bundle"] = transcript
        st.session_state["analysis_bundle"] = analysis

    transcript_bundle = st.session_state.get("transcript_bundle")
    analysis_bundle = st.session_state.get("analysis_bundle")
    if not transcript_bundle or not analysis_bundle:
        st.markdown(
            """
            ### What this app does
            - Fetches public YouTube captions or accepts pasted transcript text
            - Builds a structured summary for fast review
            - Generates study notes for revision
            - Creates quiz questions for active recall
            """
        )
        return

    title = transcript_bundle.video_title or transcript_bundle.video_id
    st.markdown(f"### {sanitize_untrusted_markdown(title)}")
    if transcript_bundle.source_url:
        st.video(transcript_bundle.source_url)
    elif transcript_bundle.source_label == "Pasted transcript":
        st.info(
            "Pasted transcript: the study pack was generated without a YouTube request or invented timestamps."
        )
    else:
        st.info(
            "Instant demo: an original sample transcript is being processed locally without YouTube or an API key."
        )
    render_meta(transcript_bundle, analysis_bundle)

    pack_text = compile_study_pack(transcript_bundle, analysis_bundle)
    st.download_button(
        label="Download complete study pack (.md)",
        data=pack_text,
        file_name=f"{transcript_bundle.video_id}-study-pack.md",
        mime="text/markdown",
        use_container_width=True,
    )

    summary_tab, notes_tab, quiz_tab, classification_tab, transcript_tab = st.tabs(
        ["Summary", "Study Notes", "Quiz", "Classification", "Transcript"]
    )
    with summary_tab:
        st.markdown(sanitize_untrusted_markdown(analysis_bundle.summary))
    with notes_tab:
        st.markdown(sanitize_untrusted_markdown(analysis_bundle.study_notes))
    with quiz_tab:
        st.markdown(sanitize_untrusted_markdown(analysis_bundle.quiz))
    with classification_tab:
        render_classification_tab(analysis_bundle)
    with transcript_tab:
        render_transcript_tab(transcript_bundle)


if __name__ == "__main__":
    run()
