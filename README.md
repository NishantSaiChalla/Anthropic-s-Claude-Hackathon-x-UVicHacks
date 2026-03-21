# Ellipsis Health Voice Check-In MVP

Hackathon prototype inspired by Ellipsis Health's use of voice biomarkers for early mental health signal detection.

This project asks a user to record a short daily voice note, uploads the recording to GPT for transcription and transcript analysis, combines that with lightweight vocal cues such as pace, silence, and energy variation, and returns a simple emotional-state flag for the day.

## Problem

Depression, anxiety, and chronic stress often show up gradually. Most tools depend on long-form journaling, questionnaires, or infrequent clinical touchpoints. That creates a gap between how someone feels day to day and when support is offered.

## MVP idea

The product reduces the check-in to one habit: a 30-second daily voice note.

From that note, the app blends two signals:

- Vocal biomarkers: speaking pace, energy variation, and silence ratio
- Language signals: transcript sentiment and emotional cue extraction

The result is a daily flag such as stable, elevated, or high attention, plus a short rationale and a seven-entry trendline stored locally in the browser.

## Why this is a strong hackathon angle

- Voice-based mental-health monitoring is distinctive and memorable in a demo setting.
- The product is easy to understand in under a minute.
- The browser handles recording and vocal heuristics, so the demo feels live.
- GPT transcribes the actual recording and analyzes the transcript on the backend.
- A text-only fallback path keeps the app usable even without an OpenAI key.

## Core features

- 30-second voice-note recorder
- Live microphone level meter during capture
- Server-side GPT transcription of the saved recording
- Automatic browser speech recognition as a draft when available
- Manual transcript fallback when GPT transcription is unavailable
- Audio-derived pacing and vocal energy heuristics
- GPT-powered recording analysis through the OpenAI API
- Anthropic and local heuristic text fallback paths when OpenAI is not configured
- Combined emotional-state scoring
- Seven-check-in local history for trend demos
- Non-diagnostic product framing and disclaimer

## How it works

1. The user records a short voice note in the browser.
2. The app uploads the saved recording to OpenAI for transcription.
3. The frontend extracts simple vocal metrics from the recorded audio.
4. The backend sends the transcript to GPT for emotional-state analysis.
5. The app combines vocal and text signals into a single daily emotional-state card.
6. The result is saved locally to build a lightweight trendline.

## Tech stack

- Frontend: vanilla HTML, CSS, JavaScript
- Backend: Node.js and Express
- AI: OpenAI for audio transcription and transcript analysis, with Anthropic as an optional text-only fallback
- Local persistence: browser localStorage

## Project structure

- public/index.html: main interface
- public/styles.css: visual design and responsive layout
- public/app.js: recording flow, speech recognition, vocal analysis, and UI rendering
- server.js: static server, audio-upload transcription, and transcript-analysis API

## Local setup

1. Install dependencies.

```bash
npm install
```

2. Create a local environment file.

```bash
copy .env.example .env
```

3. Add your OpenAI API key to .env.

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_TRANSCRIPTION_MODEL=gpt-4o-mini-transcribe
OPENAI_ANALYSIS_MODEL=gpt-4.1-mini
PORT=3000
```

4. Start the app.

```bash
npm run dev
```

5. Open http://localhost:3000 in a browser.

## Environment variables

- OPENAI_API_KEY: enables recording upload, transcription, and GPT transcript analysis
- OPENAI_TRANSCRIPTION_MODEL: optional audio transcription model override
- OPENAI_ANALYSIS_MODEL: optional transcript analysis model override
- ANTHROPIC_API_KEY: optional text-only fallback provider
- ANTHROPIC_MODEL: optional Anthropic model override
- PORT: local server port, default is 3000

## Demo script

1. Explain that the app asks for one 30-second voice note per day.
2. Record a note that includes a clear emotional signal.
3. Show the transcript fill automatically from the uploaded recording, or type a short summary manually if no key is configured.
4. Run analysis and point out both signal sources: vocal cues and GPT transcript analysis.
5. Show the daily status card and recent trendline.
6. Mention that the current MVP is non-diagnostic and designed for early emotional drift detection, not clinical diagnosis.

## Current limitations

- Recording upload analysis requires a valid OpenAI API key.
- Speech recognition draft text depends on browser support.
- Vocal biomarker scoring is heuristic, not clinically validated.
- History is local to one browser and device.
- This prototype does not include clinician workflows, authentication, or crisis escalation.

## Future extensions

- Secure user accounts and cloud history sync
- Multimodal analysis that directly consumes audio content alongside transcript cues
- Longitudinal scoring over weeks instead of isolated daily check-ins
- Personalized baselines per user
- Referral or support-resource workflows for repeated elevated signals

## Important disclaimer

This project is a hackathon prototype for wellness check-ins. It is not a diagnostic, treatment, or emergency-response system.
