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

### Changed

- Reworked the landing page around a clearer no-key value proposition.
- Rebuilt the README with a 30-second quick start, generation-mode documentation, privacy notes, and current limitations.
- Prevented dead or fabricated timestamp links for source-free demo and pasted-transcript inputs.
- Escaped dynamic metadata before rendering it in custom HTML cards.
- Expanded CI coverage to Python 3.11 and 3.12 with read-only workflow permissions.
- Updated `yt-dlp` and `pytest` to versions without the known advisories reported by `pip-audit`.
- Corrected the MIT license holder name.
