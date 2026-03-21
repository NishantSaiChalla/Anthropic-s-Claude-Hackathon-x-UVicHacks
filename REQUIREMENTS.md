# Requirements

## System prerequisites

- Node.js 20 or newer
- npm 10 or newer
- A modern browser with microphone access enabled
- An OpenAI API key for recording transcription and GPT-based analysis

## Install

Run the bootstrap script from PowerShell:

```powershell
.\requirements.ps1
```

Or double-click:

```text
requirements.cmd
```

This will:

- install all npm dependencies
- create `.env` from `.env.example` if `.env` does not exist

Manual install still works:

```bash
npm install
```

## Environment setup

Create a `.env` file in the project root with:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_TRANSCRIPTION_MODEL=gpt-4o-mini-transcribe
OPENAI_ANALYSIS_MODEL=gpt-4.1-mini
ANTHROPIC_API_KEY=
ANTHROPIC_MODEL=claude-3-7-sonnet-latest
PORT=3000
```

## Run locally

Development mode:

```bash
npm run dev
```

Production-style start:

```bash
npm start
```

Open the app at:

```text
http://localhost:3000
```

## If port 3000 is already in use

Use a different port for the current shell:

```powershell
$env:PORT=3001
npm run dev
```

## Notes

- If `OPENAI_API_KEY` is missing, recording upload analysis will not run.
- Browser speech recognition is optional and only provides a draft transcript.
- The core project dependencies are managed through `package.json` and `package-lock.json`.