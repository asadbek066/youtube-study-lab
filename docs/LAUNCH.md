# Launch Runbook

The live demo is available at https://yowassup.streamlit.app/. The checklist below is the release gate; mark each item only after the current deployment has been tested and the evidence recorded. Tailor community posts to each community's rules.

## Preflight

- [ ] Main-branch CI is green.
- [ ] The instant demo works in a clean browser without an API key.
- [ ] At least one public captioned video works through the hosted deployment.
- [ ] `docs/assets/social-preview.png` is configured as the GitHub social preview.
- [ ] Repository description, topics, and homepage URL are current.
- [ ] The release and deployment URLs are copied below.
- [ ] No API key, token, or private transcript appears in screenshots or logs.

## X thread

### 1/5

I built YouTube Study Lab, an open-source Streamlit app that turns captioned YouTube videos into summaries, revision notes, quizzes, video-type classification, and a downloadable Markdown study pack.

Demo: https://yowassup.streamlit.app/
Code: https://github.com/asadbek066/youtube-study-lab

### 2/5

Paste a YouTube URL or video ID. The app retrieves the captions, classifies the video, and builds a revision pack with timestamped sources.

It recognizes tutorials, lectures, interviews, commentary, storytelling, motivational videos, and coding walkthroughs.

### 3/5

It works without an API key in local heuristic mode. There is also a one-click demo that makes no YouTube or model-provider request.

OpenAI, Azure OpenAI, and Gemini are optional generation providers configured through environment variables.

### 4/5

The honest constraint is transcript access: the video needs public captions, and YouTube can block transcript retrieval from some hosted environments.

The app tries `youtube-transcript-api` first, then falls back to caption tracks found through `yt-dlp`.

### 5/5

The project is Python, Streamlit, tested with pytest, and MIT licensed.

Reproducible bug reports and focused contributions are welcome:
https://github.com/asadbek066/youtube-study-lab

## LinkedIn

I built YouTube Study Lab, an open-source Python and Streamlit app for turning captioned YouTube videos into study material.

Paste a YouTube URL and it produces:

- a structured summary
- revision notes
- an 18-question active-recall quiz
- video-type classification
- timestamped sources
- a downloadable Markdown study pack

It runs without an API key in local heuristic mode, and the built-in instant demo makes no YouTube or model-provider request. OpenAI, Azure OpenAI, and Gemini are optional.

There are honest limits: the source video must expose captions, and YouTube may block transcript retrieval from hosted environments. The app uses `youtube-transcript-api` first and `yt-dlp` caption tracks as a fallback.

Demo: https://yowassup.streamlit.app/
Repository: https://github.com/asadbek066/youtube-study-lab

The project is MIT licensed. Practical bug reports and focused contributions are welcome.

## Show HN

**Title:** Show HN: YouTube Study Lab – turn captioned videos into notes and quizzes

I built YouTube Study Lab, an open-source Streamlit app that converts captioned YouTube videos into revision packs.

Given a YouTube URL or video ID, it retrieves the transcript and generates a structured summary, study notes, an 18-question quiz, and video-type classification. The complete pack can be downloaded as Markdown.

The default heuristic mode runs without an API key, and the built-in instant demo makes no external request. OpenAI, Azure OpenAI, and Gemini are supported through environment configuration.

The transcript pipeline uses `youtube-transcript-api`, with `yt-dlp` caption tracks as a fallback. It still depends on the video exposing public captions, and YouTube may block transcript access from some hosted environments.

Demo: https://yowassup.streamlit.app/
Source: https://github.com/asadbek066/youtube-study-lab

I would especially value feedback on transcript reliability, output structure, and useful export formats.

## Reddit-safe draft

**Title:** I made an open-source tool that turns captioned YouTube videos into study packs

I have been working on YouTube Study Lab, a Python and Streamlit app for converting captioned YouTube videos into summaries, revision notes, quizzes, and a downloadable Markdown study pack.

It supports a zero-key heuristic mode and includes an instant demo, so an API account is not required. OpenAI, Azure OpenAI, and Gemini can be configured for model-generated output.

Technical details:

- accepts a YouTube URL or video ID
- uses `youtube-transcript-api`, with `yt-dlp` caption tracks as a fallback
- classifies lectures, tutorials, interviews, commentary, stories, motivational videos, and coding walkthroughs
- exports the complete pack as Markdown
- has pytest coverage and GitHub Actions CI

It only works when captions are publicly accessible, and YouTube may block transcript retrieval from some hosted environments.

Demo: https://yowassup.streamlit.app/
Source: https://github.com/asadbek066/youtube-study-lab

I am sharing it in case it is useful to students or developers working on transcript tools. Technical criticism and reproducible bug reports are welcome.

> Post only in a relevant community whose self-promotion rules permit project links. Rewrite the opening for that community instead of cross-posting identical text.

## Follow-up content

1. **One video, multiple generation modes:** compare heuristic, OpenAI, and Gemini output with the same public lecture and publish settings, runtime, and obvious differences.
2. **Transcript reliability report:** document manual captions, generated captions, language fallback, unavailable captions, and hosting restrictions.
3. **Pipeline walkthrough:** explain URL parsing → transcript retrieval → classification → chunking → generation → Markdown export using the architecture diagram and source links.
4. **Before and after:** show raw caption text beside the resulting summary, notes, and quiz with reproducible settings.
5. **Evidence-based release notes:** publish one concrete improvement at a time with its test, issue, screenshot, or measured result.

## Rules

- No purchased stars or followers.
- No star-for-star groups or automated engagement.
- No mass direct messages.
- No identical cross-post spam.
- Do not claim a live workflow was tested unless it was actually tested.
- Reply helpfully to real feedback, issues, and pull requests.
