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
    monkeypatch.setenv(
        "AZURE_OPENAI_ENDPOINT", "https://example-resource.openai.azure.com"
    )
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "unused-default")

    settings = load_settings()

    assert settings.active_model == "study-pack-deployment"
    assert (
        settings.azure_openai_base_url
        == "https://example-resource.openai.azure.com/openai/v1/"
    )
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


def test_invalid_openai_base_url_is_reported_as_a_config_error(monkeypatch) -> None:
    from dataclasses import replace

    settings = replace(
        load_settings(),
        requested_provider="openai",
        provider="openai",
        openai_api_key="key",
        openai_model="gpt-4o-mini",
        openai_base_url="ftp://not-http.example",
    )

    assert settings.config_error == "OPENAI_BASE_URL must be a valid http(s) URL."
    assert not settings.is_ready


def test_valid_openai_base_url_is_accepted(monkeypatch) -> None:
    from dataclasses import replace

    settings = replace(
        load_settings(),
        requested_provider="openai",
        provider="openai",
        openai_api_key="key",
        openai_model="gpt-4o-mini",
        openai_base_url="https://gateway.example/v1",
    )

    assert settings.config_error is None
    assert settings.is_ready


def test_invalid_azure_endpoint_is_reported_as_a_config_error() -> None:
    from dataclasses import replace

    settings = replace(
        load_settings(),
        requested_provider="azure_openai",
        provider="azure_openai",
        azure_openai_api_key="key",
        azure_openai_endpoint="https://[bad",
        azure_openai_deployment="deployment",
    )

    assert settings.config_error == "AZURE_OPENAI_ENDPOINT must be a valid http(s) URL."
    assert not settings.is_ready


def test_control_characters_in_url_are_rejected() -> None:
    from dataclasses import replace

    settings = replace(
        load_settings(),
        requested_provider="openai",
        provider="openai",
        openai_api_key="key",
        openai_model="gpt-4o-mini",
        openai_base_url="https://gateway.example/v1\nheader",
    )

    assert settings.config_error == "OPENAI_BASE_URL must be a valid http(s) URL."


def test_provider_specific_url_validation_ignores_inactive_providers() -> None:
    from dataclasses import replace

    gemini_settings = replace(
        load_settings(),
        requested_provider="gemini",
        provider="gemini",
        gemini_api_key="key",
        gemini_model="gemini-2.5-flash",
        openai_base_url="not-a-url",
    )

    assert gemini_settings.config_error is None
    assert gemini_settings.is_ready


def test_invalid_port_is_reported_as_a_config_error() -> None:
    from dataclasses import replace

    settings = replace(
        load_settings(),
        requested_provider="openai",
        provider="openai",
        openai_api_key="key",
        openai_model="gpt-4o-mini",
        openai_base_url="https://gateway.example:abc/v1",
    )

    assert settings.config_error == "OPENAI_BASE_URL must be a valid http(s) URL."
