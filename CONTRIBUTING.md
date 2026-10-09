# Contributing

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
python -m pip install --require-hashes -r requirements-dev.lock
cp .env.example .env             # optional: only needed for an API provider
streamlit run app.py
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

Dependabot version-update pull requests are switched off; bump dependencies by
hand and recompile the locks.

## Checks

These are the commands CI runs:

```bash
python -m pip check
python -m ruff check app.py youtube_study_tool tests
python -m ruff format --check app.py youtube_study_tool tests
python -m compileall -q app.py youtube_study_tool tests
python -m pytest tests -q
```

The test suite blocks real network access; tests that need YouTube or a model
provider must use the fakes in `tests/`.

## Pull requests

- Keep them small and focused.
- Add or update tests when changing generation logic or transcript handling.
- Update the README when behaviour or configuration changes.
- Never commit `.env` or API keys.
