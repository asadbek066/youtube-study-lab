# youtube-study-lab

Streamlit app that turns a YouTube video's captions, or pasted transcript text, into a
summary, study notes, an 18-question quiz and a Markdown export. Transcripts come from
`youtube-transcript-api` with a `yt-dlp` fallback. Generation runs locally by default
(`LLM_PROVIDER=heuristic`, no provider calls) or through OpenAI, Azure OpenAI or Gemini.

## Commands

CI (`.github/workflows/ci.yml`) runs these on Python 3.11, 3.12, 3.13 and 3.14:

```bash
python -m pip install --require-hashes -r requirements-dev.lock
python -m pip check
python -m ruff check app.py youtube_study_tool tests
python -m ruff format --check app.py youtube_study_tool tests
python -m compileall -q app.py youtube_study_tool tests
python -m pytest tests -q
```

Run the app locally: `streamlit run app.py`

`CONTRIBUTING.md` lists the same lint, format, test and compile commands with
`.\.venv\Scripts\python` on Windows. It omits `pip check`; the CI list above is the
reference.

## Layout

- `app.py`: Streamlit UI, session state, submission and provider-call limits.
- `youtube_study_tool/transcripts.py`: video ID and URL validation, caption fetching,
  `yt-dlp` fallback, shared fetch pool, deadlines and byte caps.
- `youtube_study_tool/generation.py`: OpenAI, Azure OpenAI and Gemini clients, chunked
  generation, output structure and citation checks, process-wide call budget.
- `youtube_study_tool/classification.py`, `fallback.py`: content classification and the
  local heuristic generator.
- `youtube_study_tool/models.py`, `settings.py`, `utils.py`: data types, env-driven
  settings and shared helpers.
- `tests/`: pytest suite. `tests/conftest.py` only adds the repository root to `sys.path`.
- `requirements.txt`, `requirements-dev.txt`: direct dependencies. `*.lock` files are
  their hash-pinned compiled outputs.
- `.env.example`: provider variable names and defaults. `docs/assets/`: README images.

## Rules for changes

- Tests must not contact YouTube or any model provider. Provider calls are mocked.
  `tests/conftest.py` has no network guard, so new tests must mock explicitly.
- Provider configuration is environment-driven. Use only variable names from
  `.env.example`, never commit `.env`, and never add a key entry to the Streamlit UI.
- After changing `requirements.txt` or `requirements-dev.txt`, recompile both locks.
  Locks are compiled with `--python-version 3.11`:

  ```bash
  uv pip compile requirements.txt --universal --python-version 3.11 --generate-hashes --output-file requirements.lock
  uv pip compile requirements-dev.txt --universal --python-version 3.11 --generate-hashes --output-file requirements-dev.lock
  ```

- CI installs `requirements-dev.lock` with `--require-hashes`, then runs `pip check`.
  Both must pass.
- Ruff lint, `ruff format --check` and `compileall` gate CI. Formatting is checked, not
  applied, so the format check must pass as written.
- Add or update tests when changing generation or transcript handling.
- Keep provider configuration env-driven and outputs structured and revision-friendly.
- Update the README when behavior or configuration changes. Confirm the app still
  launches with `streamlit run app.py` before opening a pull request.
- `main` is branch-protected, so every change goes through a pull request.
