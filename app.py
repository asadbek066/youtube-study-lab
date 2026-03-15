from __future__ import annotations

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

from youtube_study_tool.generation import StudyPackGenerator
from youtube_study_tool.models import AnalysisBundle, TranscriptBundle
from youtube_study_tool.transcripts import TranscriptService, normalize_languages
from youtube_study_tool.utils import format_seconds, timestamp_url

load_dotenv()

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
        --accent: #c96a3f;
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
            <div class="hero-kicker">Transcript -> Summary -> Notes -> Quiz</div>
            <h1>YouTube Study Lab</h1>
            <p>Paste any YouTube link with captions and turn it into a study pack. The app extracts the transcript, builds a concise summary, turns the lesson into notes, and creates quiz questions for review.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_meta(bundle: TranscriptBundle, analysis: AnalysisBundle) -> None:
    col1, col2, col3, col4 = st.columns(4)
    cards = [
        ("Transcript", bundle.language_name),
        ("Source", "Auto captions" if bundle.is_generated else "Manual captions"),
        ("Type", analysis.classification.video_type.title()),
        ("Generator", f"{analysis.provider} ({analysis.model})"),
    ]
    for column, (label, value) in zip((col1, col2, col3, col4), cards):
        column.markdown(
            f"""
            <div class="meta-card">
                <div class="meta-label">{label}</div>
                <div class="meta-value">{value}</div>
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
    st.text_area("Transcript text", value=bundle.transcript_text, height=320)
    with st.expander("Timestamped transcript"):
        for segment in bundle.segments:
            st.markdown(
                f"[{format_seconds(segment.start)}]({timestamp_url(bundle.video_id, segment.start)}) {segment.text}"
            )


def render_classification_tab(analysis: AnalysisBundle) -> None:
    classification = analysis.classification
    st.markdown(f"### {classification.video_type.title()}")
    st.markdown(f"**Confidence:** {classification.confidence:.2f}")
    st.markdown(f"**Reason:** {classification.reason}")
    st.markdown(f"**Best summary style:** {classification.best_summary_style}")
    st.markdown(f"**Best note style:** {classification.best_note_style}")
    st.json(asdict(classification))


def compile_study_pack(bundle: TranscriptBundle, analysis: AnalysisBundle) -> str:
    return dedent(
        f"""
        # {bundle.video_title or bundle.video_id}

        Source: {bundle.source_url}
        Transcript language: {bundle.language_name} ({bundle.language_code})
        Duration: {format_seconds(bundle.duration_seconds)}
        Generated with: {analysis.provider} ({analysis.model})
        Video type: {analysis.classification.video_type} ({analysis.classification.confidence:.2f})
        Classification reason: {analysis.classification.reason}

        {analysis.summary}

        {analysis.study_notes}

        {analysis.quiz}
        """
    ).strip()


def run() -> None:
    render_hero()
    transcript_service = TranscriptService()
    generator = StudyPackGenerator()

    with st.sidebar:
        st.header("Options")
        language_input = st.text_input("Preferred transcript languages", value="en,en-US,en-GB")
        st.caption("LLM provider and model are controlled from `.env`.")
        if generator.is_ready:
            st.success(generator.status_message)
        else:
            st.info(generator.status_message)

        st.markdown(f"**Configured provider:** {generator.provider_label}")
        st.markdown(f"**Configured model/deployment:** `{generator.model_name}`")
        st.markdown(
            f"**Summary profile:** `{generator.settings.summary_style}` / `{generator.settings.summary_detail}`"
        )

        st.caption(
            "Some videos do not expose captions publicly. In those cases YouTube may block transcript extraction."
        )

    with st.form("analyze-form"):
        source = st.text_input(
            "YouTube URL or video ID",
            placeholder="https://www.youtube.com/watch?v=...",
        )
        submitted = st.form_submit_button("Build study pack", use_container_width=True)

    if submitted:
        if not source.strip():
            st.warning("Paste a YouTube URL or ID to get started.")
            return

        languages = normalize_languages(language_input)
        try:
            with st.spinner("Pulling transcript from YouTube..."):
                transcript = transcript_service.fetch(source, languages)
            with st.spinner("Building summary, notes, and quiz..."):
                analysis = generator.generate(transcript)
        except ValueError as error:
            st.error(str(error))
            return
        except (NoTranscriptFound, TranscriptsDisabled, CouldNotRetrieveTranscript, YouTubeTranscriptApiException) as error:
            st.error(f"Transcript extraction failed: {error}")
            return
        except Exception as error:
            st.error(f"Something unexpected happened: {error}")
            return

        st.session_state["transcript_bundle"] = transcript
        st.session_state["analysis_bundle"] = analysis

    transcript_bundle = st.session_state.get("transcript_bundle")
    analysis_bundle = st.session_state.get("analysis_bundle")
    if not transcript_bundle or not analysis_bundle:
        st.markdown(
            """
            ### What this tool does
            - Extracts the transcript from a YouTube video with captions
            - Produces a high-signal summary
            - Turns the lesson into study notes
            - Generates a review quiz you can use for active recall
            """
        )
        return

    title = transcript_bundle.video_title or transcript_bundle.video_id
    st.subheader(title)
    st.video(transcript_bundle.source_url)
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
        st.markdown(analysis_bundle.summary)
    with notes_tab:
        st.markdown(analysis_bundle.study_notes)
    with quiz_tab:
        st.markdown(analysis_bundle.quiz)
    with classification_tab:
        render_classification_tab(analysis_bundle)
    with transcript_tab:
        render_transcript_tab(transcript_bundle)


if __name__ == "__main__":
    run()
