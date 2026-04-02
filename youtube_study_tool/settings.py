from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

SUPPORTED_PROVIDERS = {"heuristic", "openai", "azure_openai", "gemini"}
SUPPORTED_SUMMARY_STYLES = {"adaptive", "tutorial", "motivational", "general"}
SUPPORTED_SUMMARY_DETAILS = {"concise", "balanced", "deep"}
PROVIDER_LABELS = {
    "heuristic": "Heuristic fallback",
    "openai": "OpenAI",
    "azure_openai": "Azure OpenAI",
    "gemini": "Gemini",
}


def _read_env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _read_float(name: str, default: float) -> float:
    value = _read_env(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _read_int(name: str, default: int) -> int:
    value = _read_env(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _clamp_float(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _read_choice(name: str, default: str, allowed: set[str]) -> str:
    value = _read_env(name, default).lower()
    return value if value in allowed else default


@dataclass(frozen=True)
class LLMSettings:
    requested_provider: str
    provider: str
    llm_model_override: str
    openai_api_key: str
    openai_model: str
    openai_base_url: str
    azure_openai_api_key: str
    azure_openai_endpoint: str
    azure_openai_deployment: str
    gemini_api_key: str
    gemini_model: str
    temperature: float
    chunk_max_output_tokens: int
    final_max_output_tokens: int
    summary_style: str
    summary_detail: str

    @property
    def provider_label(self) -> str:
        return PROVIDER_LABELS[self.provider]

    @property
    def active_model(self) -> str:
        if self.provider == "heuristic":
            return "local-rules"
        if self.llm_model_override:
            return self.llm_model_override
        if self.provider == "openai":
            return self.openai_model or "gpt-4o-mini"
        if self.provider == "azure_openai":
            return self.azure_openai_deployment
        if self.provider == "gemini":
            return self.gemini_model or "gemini-2.5-flash"
        return "local-rules"

    @property
    def azure_openai_base_url(self) -> str:
        endpoint = self.azure_openai_endpoint.rstrip("/")
        if not endpoint:
            return ""
        if endpoint.endswith("/openai/v1"):
            return f"{endpoint}/"
        return f"{endpoint}/openai/v1/"

    @property
    def config_error(self) -> str | None:
        if self.requested_provider and self.requested_provider not in SUPPORTED_PROVIDERS:
            return (
                f"LLM_PROVIDER={self.requested_provider!r} is unsupported. "
                "Use heuristic, openai, azure_openai, or gemini."
            )

        if self.provider == "heuristic":
            return None

        if not self.active_model:
            if self.provider == "azure_openai":
                return "Set LLM_MODEL or AZURE_OPENAI_DEPLOYMENT."
            if self.provider == "gemini":
                return "Set LLM_MODEL or GEMINI_MODEL."
            return "Set LLM_MODEL or OPENAI_MODEL."

        if self.provider == "openai" and not self.openai_api_key:
            return "OPENAI_API_KEY is missing."

        if self.provider == "azure_openai":
            if not self.azure_openai_api_key:
                return "AZURE_OPENAI_API_KEY is missing."
            if not self.azure_openai_endpoint:
                return "AZURE_OPENAI_ENDPOINT is missing."

        if self.provider == "gemini" and not self.gemini_api_key:
            return "GEMINI_API_KEY or GOOGLE_API_KEY is missing."

        return None

    @property
    def is_ready(self) -> bool:
        return self.provider != "heuristic" and self.config_error is None

    @property
    def status_message(self) -> str:
        if self.provider == "heuristic":
            if self.requested_provider and self.requested_provider not in SUPPORTED_PROVIDERS:
                return f"{self.config_error} Using heuristic fallback."
            return "Heuristic mode is active. Set LLM_PROVIDER in .env to openai, azure_openai, or gemini to use an API provider."

        if self.is_ready:
            return f"{self.provider_label} ready with `{self.active_model}`."

        return f"{self.provider_label} is selected, but {self.config_error} Using heuristic fallback."


def load_settings() -> LLMSettings:
    requested_provider = _read_env("LLM_PROVIDER").lower()
    if not requested_provider:
        requested_provider = "openai" if _read_env("OPENAI_API_KEY") else "heuristic"

    provider = requested_provider if requested_provider in SUPPORTED_PROVIDERS else "heuristic"
    temperature = _clamp_float(_read_float("LLM_TEMPERATURE", 0.3), 0.0, 2.0)
    chunk_max_output_tokens = _clamp_int(_read_int("LLM_CHUNK_MAX_OUTPUT_TOKENS", 450), 64, 4000)
    final_max_output_tokens = _clamp_int(_read_int("LLM_FINAL_MAX_OUTPUT_TOKENS", 2600), 128, 16000)

    return LLMSettings(
        requested_provider=requested_provider,
        provider=provider,
        llm_model_override=_read_env("LLM_MODEL"),
        openai_api_key=_read_env("OPENAI_API_KEY"),
        openai_model=_read_env("OPENAI_MODEL", "gpt-4o-mini"),
        openai_base_url=_read_env("OPENAI_BASE_URL"),
        azure_openai_api_key=_read_env("AZURE_OPENAI_API_KEY"),
        azure_openai_endpoint=_read_env("AZURE_OPENAI_ENDPOINT"),
        azure_openai_deployment=_read_env("AZURE_OPENAI_DEPLOYMENT"),
        gemini_api_key=_read_env("GEMINI_API_KEY") or _read_env("GOOGLE_API_KEY"),
        gemini_model=_read_env("GEMINI_MODEL", "gemini-2.5-flash"),
        temperature=temperature,
        chunk_max_output_tokens=chunk_max_output_tokens,
        final_max_output_tokens=final_max_output_tokens,
        summary_style=_read_choice("SUMMARY_STYLE", "adaptive", SUPPORTED_SUMMARY_STYLES),
        summary_detail=_read_choice("SUMMARY_DETAIL", "balanced", SUPPORTED_SUMMARY_DETAILS),
    )
