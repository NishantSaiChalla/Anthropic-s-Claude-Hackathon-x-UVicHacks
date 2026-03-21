# Ellipsis Health Voice Check-In MVP

Hackathon prototype inspired by Ellipsis Health's use of voice biomarkers for early mental health signal detection.

This project asks a user to record a short daily voice note, extracts lightweight vocal cues such as pace, silence, and energy variation, combines that with transcript sentiment analysis, and returns a simple emotional-state flag for the day.

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
- The browser handles recording and audio heuristics, so the demo feels live.
- Claude analysis makes the transcript response feel contextual instead of rule-based.
- A fallback heuristic keeps the app usable even without an API key.

## Core features

- 30-second voice-note recorder
- Live microphone level meter during capture
- Automatic browser speech recognition when available
- Manual transcript fallback when speech recognition is unavailable
- Audio-derived pacing and vocal energy heuristics
- Claude-powered transcript analysis through Anthropic's SDK
- Local fallback transcript analysis when Claude is not configured
- Combined emotional-state scoring
- Seven-check-in local history for trend demos
- Non-diagnostic product framing and disclaimer

## How it works

1. The user records a short voice note in the browser.
2. The app captures transcript text through browser speech recognition when supported.
3. The frontend extracts simple vocal metrics from the recorded audio.
4. The backend sends the transcript to Claude, or uses a local heuristic fallback.
5. The app combines vocal and text signals into a single daily emotional-state card.
6. The result is saved locally to build a lightweight trendline.

## Tech stack

- Frontend: vanilla HTML, CSS, JavaScript
- Backend: Node.js and Express
- AI: Anthropic Claude via @anthropic-ai/sdk
- Local persistence: browser localStorage

## Project structure

- public/index.html: main interface
- public/styles.css: visual design and responsive layout
- public/app.js: recording flow, speech recognition, vocal analysis, and UI rendering
- server.js: static server and transcript-analysis API

## Local setup

1. Install dependencies.

```bash
npm install
```

2. Create a local environment file.

```bash
copy .env.example .env
```

3. Add your Anthropic API key to .env.

```env
ANTHROPIC_API_KEY=your_api_key_here
ANTHROPIC_MODEL=claude-3-7-sonnet-latest
PORT=3000
```

4. Start the app.

```bash
npm run dev
```

5. Open http://localhost:3000 in a browser.

## Environment variables

- ANTHROPIC_API_KEY: enables Claude transcript analysis
- ANTHROPIC_MODEL: optional model override
- PORT: local server port, default is 3000

## Demo script

1. Explain that the app asks for one 30-second voice note per day.
2. Record a note that includes a clear emotional signal.
3. Show the transcript fill automatically, or type a short summary manually.
4. Run analysis and point out both signal sources: vocal cues and transcript sentiment.
5. Show the daily status card and recent trendline.
6. Mention that the current MVP is non-diagnostic and designed for early emotional drift detection, not clinical diagnosis.

## Current limitations

- Speech recognition depends on browser support.
- Vocal biomarker scoring is heuristic, not clinically validated.
- History is local to one browser and device.
- This prototype does not include clinician workflows, authentication, or crisis escalation.

## Future extensions

- Secure user accounts and cloud history sync
- Real speech-to-text service instead of browser-only transcription
- Longitudinal scoring over weeks instead of isolated daily check-ins
- Personalized baselines per user
- Referral or support-resource workflows for repeated elevated signals

## Important disclaimer

This project is a hackathon prototype for wellness check-ins. It is not a diagnostic, treatment, or emergency-response system.
