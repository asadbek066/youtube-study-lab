# Changelog

Notable changes to YouTube Study Lab are documented here.

The project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and uses semantic versioning for releases.

## [Unreleased]

### Added

- One-click, network-free instant demo with an original sample transcript.
- Pasted-transcript fallback for environments where YouTube blocks server-side caption retrieval.
- Streamlit interaction tests for the landing page and generated study-pack flow.
- Product screenshots, an animated walkthrough, an architecture diagram, and a GitHub social-preview image.
- Structured bug and feature request forms, a pull request checklist, and Dependabot configuration.
- Process-wide provider-call budget (`LLM_MAX_PROVIDER_CALLS_PER_HOUR`), plus
  generation (300 s), request (120 s), and transcript-fetch (90 s) time bounds.
- Visible notice when a configured provider falls back to local generation.
- Hash-pinned runtime and development locks (`requirements.lock`,
  `requirements-dev.lock`) installed with `--require-hashes` in CI.

### Changed

- Bound caption downloads before parsing, kept provider input data in escaped
  JSON, and routed oversized LLM jobs to the local fallback.
- Reworked the landing page around a clearer no-key value proposition.
- Rebuilt the README with a 30-second quick start, generation-mode documentation, privacy notes, and current limitations.
- Prevented dead or fabricated timestamp links for source-free demo and pasted-transcript inputs.
- Escaped dynamic metadata before rendering it in custom HTML cards.
- Expanded CI coverage to Python 3.11–3.14 with read-only workflow permissions,
  dependency-consistency checks, and byte-compilation.
- Updated `yt-dlp`, `google-genai`, `ruff`, and `pytest` to versions without the
  known advisories reported by `pip-audit`.
- Corrected the MIT license holder name.
