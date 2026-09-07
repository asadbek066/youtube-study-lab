from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_landing_page_surfaces_the_no_key_value_proposition() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=15).run()

    page_copy = "\n".join(element.value for element in app.markdown)
    assert "One link. A complete study pack." in page_copy
    assert "No API key required" in page_copy
    assert "Timestamped sources" in page_copy
    assert "18-question quiz" in page_copy
    assert "Markdown export" in page_copy
    assert app.button(key="instant-demo").label == "See a complete study pack instantly"
    assert not app.exception


def test_instant_demo_button_builds_a_complete_study_pack() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=15).run()

    app.button(key="instant-demo").click().run()

    transcript = app.session_state["transcript_bundle"]
    analysis = app.session_state["analysis_bundle"]
    assert transcript.video_id == "study-demo"
    assert transcript.source_url == ""
    assert "## Summary" in analysis.summary
    assert "## Study Notes" in analysis.study_notes
    assert "## Quiz" in analysis.quiz
    assert not app.exception


def test_manual_transcript_form_builds_a_source_free_study_pack() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=15).run()
    transcript_text = (
        "A neural network starts with adjustable weights. It makes a prediction from an input example. "
        "The prediction is compared with the correct target to calculate a loss. Backpropagation measures "
        "how each weight contributed to that error. Gradient descent updates the weights a little at a time. "
        "Repeating this process across many examples helps the network learn useful patterns and generalize."
    )

    app.text_input(key="manual-title").input("My pasted lecture")
    app.text_area(key="manual-transcript").input(transcript_text)
    app.button(key="manual-submit").click().run()

    transcript = app.session_state["transcript_bundle"]
    analysis = app.session_state["analysis_bundle"]
    assert transcript.video_title == "My pasted lecture"
    assert transcript.source_label == "Pasted transcript"
    assert transcript.source_url == ""
    assert "## Summary" in analysis.summary
    assert "youtube.com/watch?v=pasted-transcript" not in analysis.summary
    assert not app.exception


def test_failed_manual_submission_clears_the_previous_pack() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=15).run()
    app.text_area(key="manual-transcript").input("A useful lesson about testing.")
    app.button(key="manual-submit").click().run()
    assert "transcript_bundle" in app.session_state

    app.text_area(key="manual-transcript").input("")
    app.button(key="manual-submit").click().run()

    assert "transcript_bundle" not in app.session_state
    assert "analysis_bundle" not in app.session_state
    assert not app.exception


def test_markdown_export_neutralizes_untrusted_metadata() -> None:
    from app import compile_study_pack
    from youtube_study_tool.fallback import generate_fallback_bundle
    from youtube_study_tool.manual import build_manual_transcript

    bundle = build_manual_transcript(
        "A lesson explains tests and feedback.", title="[click](javascript:alert(1))"
    )
    analysis = generate_fallback_bundle(bundle)

    exported = compile_study_pack(bundle, analysis)

    assert "javascript:" not in exported
    assert "alert(1)" not in exported
