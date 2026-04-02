# Contributing

Thanks for contributing to YouTube Study Lab.

## Development setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

## Run locally

```powershell
streamlit run app.py
```

## Tests

```powershell
.\.venv\Scripts\python -m pytest tests -q
```

## Guidelines

- Keep environment-specific secrets out of git.
- Prefer small, focused pull requests.
- Add or update tests when changing generation logic or transcript handling.
- Preserve the env-driven provider configuration model.
- Keep outputs structured and revision-friendly.

## Before opening a pull request

- Run the tests locally.
- Confirm the app still launches with `streamlit run app.py`.
- Update README docs if behavior or configuration changed.
