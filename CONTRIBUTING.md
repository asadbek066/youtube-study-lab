# Contributing

Thanks for contributing to YouTube Study Lab.

## Development setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install --require-hashes -r requirements-dev.lock
copy .env.example .env
```

`requirements-dev.lock` is a hash-pinned lock for the runtime plus dev tools.
When adding or bumping a direct dependency, update `requirements.txt` or
`requirements-dev.txt` and recompile the locks:

```bash
uv pip compile requirements.txt --universal --python-version 3.11 \
  --generate-hashes --output-file requirements.lock
uv pip compile requirements-dev.txt --universal --python-version 3.11 \
  --generate-hashes --output-file requirements-dev.lock
```

## Run locally

```powershell
streamlit run app.py
```

## Tests

```powershell
.\.venv\Scripts\python -m ruff check app.py youtube_study_tool tests
.\.venv\Scripts\python -m ruff format --check app.py youtube_study_tool tests
.\.venv\Scripts\python -m pytest tests -q
.\.venv\Scripts\python -m compileall -q app.py youtube_study_tool tests
```

## Guidelines

- Keep environment-specific secrets out of git.
- Prefer small, focused pull requests.
- Add or update tests when changing generation logic or transcript handling.
- Preserve the env-driven provider configuration model.
- Keep outputs structured and revision-friendly.

## Before opening a pull request

- Run the lint, formatting, test, and compile commands above locally.
- Confirm the app still launches with `streamlit run app.py`.
- Update README docs if behavior or configuration changed.
