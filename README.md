# YouTube Study Lab

YouTube Study Lab turns a YouTube video into a study pack:

- extracts the transcript
- summarizes the content
- generates study notes
- generates quizzes

It is built for people who want to turn long YouTube videos into something they can review, revise, and apply quickly.

## What it produces

For each supported video, the app can generate:

- a structured summary
- revision-friendly study notes
- a mixed-difficulty quiz
- transcript-based video classification

The generator adapts to video type, including:

- tutorial
- lecture
- motivational
- interview
- commentary
- storytelling
- coding walkthrough

The app supports four generation modes from one `.env` file:

- `heuristic`: no API calls, local summaries/notes/quizzes
- `openai`: OpenAI Responses API
- `azure_openai`: Azure OpenAI via the OpenAI-compatible endpoint
- `gemini`: Google Gemini via the official `google-genai` SDK

## Stack

- Python 3.12
- Streamlit
- `youtube-transcript-api`
- `yt-dlp`
- OpenAI Python SDK
- Google GenAI SDK

## How it works

1. Paste a YouTube URL with captions enabled.
2. The app extracts the transcript from YouTube.
   If the primary transcript backend is blocked by YouTube, the app falls back to caption extraction through `yt-dlp`.
3. It classifies the transcript by video type.
4. It compresses long transcript context when needed.
5. It generates:
   - a detailed summary
   - study notes
   - quizzes
6. You can download the transcript or the complete study pack.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
streamlit run app.py
```

## Provider setup

Provider selection now lives entirely in `.env`. In normal use, you only change `LLM_PROVIDER`, and optionally `LLM_MODEL` if you want to override the provider-specific default.

```env
LLM_PROVIDER=heuristic
LLM_MODEL=
```

You can also tune how the summary behaves:

```env
SUMMARY_STYLE=adaptive
SUMMARY_DETAIL=balanced
```

- `SUMMARY_STYLE=adaptive` lets the app shape the summary around the video itself.
- `SUMMARY_STYLE=tutorial` is useful when you mostly summarize build/process/how-to videos.
- `SUMMARY_STYLE=motivational` is useful when you want the output to focus on core message and practical mindset shifts.
- `SUMMARY_DETAIL=concise` gives a tight TL;DR.
- `SUMMARY_DETAIL=balanced` keeps the best signal without feeling too compressed.
- `SUMMARY_DETAIL=deep` gives a fuller summary without turning into a line-by-line retelling.

### OpenAI

```env
LLM_PROVIDER=openai
LLM_MODEL=
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
```

### Azure OpenAI

`AZURE_OPENAI_DEPLOYMENT` should be your Azure deployment name.

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

## Output format

### Summary

The summary is generated in this structure:

1. Overview
2. Main ideas
3. Step-by-step breakdown
4. Important examples
5. Practical takeaways
6. One-paragraph compressed version

### Study notes

The study notes are generated in this structure:

1. Topic
2. Key concepts
3. Important details
4. Examples
5. Common mistakes or misconceptions
6. What to remember

### Quiz

The quiz is generated in this structure:

- 10 multiple-choice questions
- 5 short-answer questions
- 3 application-based questions
- answer and explanation for every question
- mixed difficulty: easy, medium, hard

## Example output

Example summary shape:

```md
## Summary
### 1. Overview
This video explains the main topic and why it matters.

### 2. Main ideas
- Core idea one
- Core idea two

### 3. Step-by-step breakdown
- Step 1
- Step 2

### 4. Important examples
- Example from the transcript

### 5. Practical takeaways
- Actionable lesson

### 6. One-paragraph compressed version
A compact revision-ready version of the transcript.
```

Example quiz item shape:

```md
1. [medium] Which concept best matches the transcript segment?
A. Option one
B. Option two
C. Option three
D. Option four
Answer: B. Option two
Explanation: This is the best answer because the transcript explicitly supports it.
```

## Notes

- Transcript extraction depends on captions being available for the video.
- Some YouTube videos block transcript access entirely.
- The app now has a secondary transcript fallback path via `yt-dlp`, which helps with some YouTube IP-block cases.
- If the selected provider is misconfigured, the app falls back to heuristic mode automatically.
- `LLM_MODEL` is optional. If left blank, the app uses `OPENAI_MODEL`, `AZURE_OPENAI_DEPLOYMENT`, or `GEMINI_MODEL` depending on `LLM_PROVIDER`.
- The summary is now content-aware: tutorial/build videos get step-oriented summaries, while motivational videos get message-and-action oriented summaries.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for local setup, testing, and pull request guidance.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

## Tests

```powershell
.\.venv\Scripts\python -m pytest
```
