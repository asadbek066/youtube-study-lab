from youtube_study_tool.settings import load_settings


def test_settings_use_provider_specific_model(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")

    settings = load_settings()

    assert settings.provider == "gemini"
    assert settings.active_model == "gemini-2.5-flash"
    assert settings.is_ready is True


def test_settings_allow_global_model_override(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "azure_openai")
    monkeypatch.setenv("LLM_MODEL", "study-pack-deployment")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example-resource.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "unused-default")

    settings = load_settings()

    assert settings.active_model == "study-pack-deployment"
    assert settings.azure_openai_base_url == "https://example-resource.openai.azure.com/openai/v1/"
    assert settings.is_ready is True


def test_invalid_provider_falls_back_to_heuristic(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "something-else")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    settings = load_settings()

    assert settings.provider == "heuristic"
    assert "unsupported" in settings.status_message.lower()


def test_summary_settings_normalize_invalid_values(monkeypatch) -> None:
    monkeypatch.setenv("SUMMARY_STYLE", "unknown")
    monkeypatch.setenv("SUMMARY_DETAIL", "lots")

    settings = load_settings()

    assert settings.summary_style == "adaptive"
    assert settings.summary_detail == "balanced"


def test_generation_values_are_clamped(monkeypatch) -> None:
    monkeypatch.setenv("LLM_TEMPERATURE", "9")
    monkeypatch.setenv("LLM_CHUNK_MAX_OUTPUT_TOKENS", "1")
    monkeypatch.setenv("LLM_FINAL_MAX_OUTPUT_TOKENS", "999999")

    settings = load_settings()

    assert settings.temperature == 2.0
    assert settings.chunk_max_output_tokens == 64
    assert settings.final_max_output_tokens == 16000
