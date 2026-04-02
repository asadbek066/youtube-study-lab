# YouTube Study Lab

YouTube Study Lab converts a captioned YouTube video into a revision pack:

- transcript
- structured summary
- study notes
- quiz set
- transcript classification

It is built for students who want faster review loops from long-form videos.

## Features

- Dual transcript ingestion:
  - primary: `youtube-transcript-api`
  - fallback: `yt-dlp` caption track parsing
- Transcript classification (`tutorial`, `lecture`, `motivational`, `interview`, `commentary`, `storytelling`, `coding walkthrough`)
- Multi-provider generation from env config:
  - `heuristic` (local no-API mode)
  - `openai`
  - `azure_openai`
  - `gemini`
- Downloadable transcript and full study pack markdown

## Tech Stack

- Python 3.12
- Streamlit
- `youtube-transcript-api`
- `yt-dlp`
- OpenAI Python SDK
- Google GenAI SDK
- pytest

## Processing Flow

1. Parse URL or video ID.
2. Fetch transcript via primary backend; fallback to `yt-dlp` when needed.
3. Classify transcript type.
4. Chunk long transcripts for stable generation.
5. Generate `Summary`, `Study Notes`, and `Quiz`.
6. Render and export outputs.

## Local Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
streamlit run app.py
```

## Provider Configuration

All provider settings are env-driven.

```env
LLM_PROVIDER=heuristic
LLM_MODEL=
SUMMARY_STYLE=adaptive
SUMMARY_DETAIL=balanced
```

### OpenAI

```env
LLM_PROVIDER=openai
LLM_MODEL=
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
```

### Azure OpenAI

```env
LLM_PROVIDER=azure_openai
LLM_MODEL=
AZURE_OPENAI_API_KEY=your_azure_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_DEPLOYMENT=your-deployment-name
```

### Gemini

```env
LLM_PROVIDER=gemini
LLM_MODEL=
GEMINI_API_KEY=your_gemini_key_here
GEMINI_MODEL=gemini-2.5-flash
```

## Output Shape

- Summary:
  - Overview
  - Main ideas
  - Step-by-step breakdown
  - Important examples
  - Practical takeaways
  - One-paragraph compressed version
- Study notes:
  - Topic
  - Key concepts
  - Important details
  - Examples
  - Common mistakes or misconceptions
  - What to remember
- Quiz:
  - 10 multiple-choice
  - 5 short-answer
  - 3 application-based
  - answer + explanation per question

## Notes

- Captions must exist on the target video.
- Some videos block transcript access entirely.
- Provider misconfiguration automatically falls back to heuristic mode.

## Testing

```powershell
.\.venv\Scripts\python -m pytest tests -q
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT. See [LICENSE](LICENSE).
