"""Process-wide reuse of the transcript service and study-pack generator."""

from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import app as app_module
from youtube_study_tool.generation import StudyPackGenerator
from youtube_study_tool.manual import build_manual_transcript
from youtube_study_tool.settings import load_settings, settings_fingerprint

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
SECRET = "sk-test-SECRET-value-1234567890"


@pytest.fixture(autouse=True)
def _fresh_caches(monkeypatch: pytest.MonkeyPatch):
    for name in (
        "LLM_PROVIDER",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "LLM_TEMPERATURE",
        "LLM_MODEL",
        "SUMMARY_STYLE",
        "SUMMARY_DETAIL",
    ):
        monkeypatch.delenv(name, raising=False)
    app_module.st.cache_resource.clear()
    yield
    app_module.st.cache_resource.clear()


def test_same_settings_reuse_one_generator() -> None:
    assert app_module.get_generator() is app_module.get_generator()


def test_transcript_service_is_reused() -> None:
    assert app_module.get_transcript_service() is app_module.get_transcript_service()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("LLM_TEMPERATURE", "0.9"),
        ("LLM_MODEL", "another-model"),
        ("SUMMARY_STYLE", "tutorial"),
        ("SUMMARY_DETAIL", "deep"),
        ("OPENAI_BASE_URL", "https://proxy.example/v1"),
        ("LLM_CHUNK_MAX_OUTPUT_TOKENS", "900"),
        ("LLM_FINAL_MAX_OUTPUT_TOKENS", "5000"),
    ],
)
def test_behaviour_affecting_setting_yields_a_new_generator(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    first = app_module.get_generator()
    monkeypatch.setenv(name, value)
    second = app_module.get_generator()

    assert second is not first
    assert app_module.get_generator() is second


def test_api_key_change_yields_a_new_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    first = app_module.get_generator()
    monkeypatch.setenv("OPENAI_API_KEY", SECRET + "-rotated")

    assert app_module.get_generator() is not first


def test_api_key_never_appears_in_the_cache_key_or_reprs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    settings = load_settings()
    fingerprint = settings_fingerprint(settings)

    assert SECRET not in fingerprint
    assert SECRET not in repr(settings)
    assert settings.openai_api_key == SECRET  # still available to the SDK client
    # The only value Streamlit hashes is the digest; the settings travel in an
    # underscore-prefixed (unhashed) parameter.
    parameters = app_module._cached_generator.__wrapped__.__code__.co_varnames[:2]
    assert parameters == ("fingerprint", "_settings")
    assert fingerprint != settings_fingerprint(
        replace(settings, openai_api_key=SECRET + "x")
    )


def test_evicted_resources_are_closed() -> None:
    closed: list[str] = []

    class Resource:
        def close(self) -> None:
            closed.append("closed")

    app_module._release_resource(Resource())
    app_module._release_resource(object())  # no close(): ignored

    assert closed == ["closed"]


def test_clearing_the_cache_closes_the_cached_service() -> None:
    service = app_module.get_transcript_service()
    closed: list[bool] = []
    service.close = lambda: closed.append(True)  # type: ignore[method-assign]

    app_module.st.cache_resource.clear()

    assert closed == [True]


def test_generator_state_is_per_thread() -> None:
    generator = StudyPackGenerator()
    seen: dict[str, tuple[str, int]] = {}
    barrier = threading.Barrier(2)

    def worker(name: str, calls: int) -> None:
        generator.last_fallback_reason = name
        generator._provider_calls = calls
        barrier.wait(timeout=5)  # both threads have written before either reads
        seen[name] = (generator.last_fallback_reason, generator.provider_calls_used)

    threads = [
        threading.Thread(target=worker, args=("a", 1)),
        threading.Thread(target=worker, args=("b", 2)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert seen == {"a": ("a", 1), "b": ("b", 2)}
    assert generator.last_fallback_reason == ""  # main thread untouched


def test_shared_generator_keeps_concurrent_generations_apart(monkeypatch) -> None:
    generator = StudyPackGenerator()
    barrier = threading.Barrier(2)
    outcomes: dict[str, tuple[str, int]] = {}

    def fake_llm(self_bundle):
        generator._provider_calls = 7 if self_bundle.video_title == "slow" else 1
        barrier.wait(timeout=5)
        raise RuntimeError("provider down for " + self_bundle.video_title)

    monkeypatch.setattr(generator, "_generate_with_llm", fake_llm)
    monkeypatch.setattr(StudyPackGenerator, "is_ready", property(lambda self: True))

    def run(title: str) -> None:
        bundle = build_manual_transcript(
            "A model learns from examples and feedback.", title=title
        )
        generator.generate(bundle)
        outcomes[title] = (
            generator.last_fallback_reason,
            generator.provider_calls_used,
        )

    threads = [threading.Thread(target=run, args=(t,)) for t in ("slow", "fast")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert outcomes == {
        "slow": ("the provider request failed", 7),
        "fast": ("the provider request failed", 1),
    }


def test_shared_transcript_service_serves_two_threads(monkeypatch) -> None:
    service = app_module.get_transcript_service()
    results: dict[str, str] = {}
    barrier = threading.Barrier(2)

    def fake_api_fetch(video_id, source_url, languages):
        barrier.wait(timeout=5)
        return (video_id, source_url, languages)

    monkeypatch.setattr(service, "_fetch_with_youtube_transcript_api", fake_api_fetch)

    def run(video_id: str) -> None:
        results[video_id] = service.fetch(video_id, ["en"])[0]

    threads = [
        threading.Thread(target=run, args=(v,)) for v in ("aaaaaaaaaaa", "bbbbbbbbbbb")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert results == {"aaaaaaaaaaa": "aaaaaaaaaaa", "bbbbbbbbbbb": "bbbbbbbbbbb"}


def test_reruns_do_not_build_a_second_service_or_generator(monkeypatch) -> None:
    constructed: list[str] = []
    real_generator_init = StudyPackGenerator.__init__
    real_service_init = app_module.TranscriptService.__init__

    def counting_generator_init(self, *args, **kwargs):
        constructed.append("generator")
        real_generator_init(self, *args, **kwargs)

    def counting_service_init(self, *args, **kwargs):
        constructed.append("service")
        real_service_init(self, *args, **kwargs)

    monkeypatch.setattr(StudyPackGenerator, "__init__", counting_generator_init)
    monkeypatch.setattr(app_module.TranscriptService, "__init__", counting_service_init)

    app = AppTest.from_file(str(APP_PATH), default_timeout=15).run()
    app.run()
    app.text_input[0].set_value("en").run()

    assert sorted(constructed) == ["generator", "service"]
    assert not app.exception
