import dotenv from 'dotenv';
import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import Anthropic from '@anthropic-ai/sdk';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const app = express();
const port = Number(process.env.PORT || 3000);
const anthropicApiKey = process.env.ANTHROPIC_API_KEY;
const anthropicModel = process.env.ANTHROPIC_MODEL || 'claude-3-7-sonnet-latest';
const anthropic = anthropicApiKey ? new Anthropic({ apiKey: anthropicApiKey }) : null;

app.use(express.json({ limit: '1mb' }));
app.use(express.static(path.join(__dirname, 'public')));

app.get('/api/health', (_request, response) => {
  response.json({ ok: true, claudeConfigured: Boolean(anthropic) });
});

app.post('/api/analyze-text', async (request, response) => {
  const transcript = String(request.body?.transcript || '').trim();

  if (!transcript) {
    response.status(400).json({ error: 'Transcript is required.' });
    return;
  }

  try {
    const analysis = anthropic
      ? await analyzeTranscriptWithClaude(transcript)
      : analyzeTranscriptHeuristically(transcript);

    response.json({
      ...analysis,
      model: anthropic ? anthropicModel : 'local-heuristic',
      analyzedAt: new Date().toISOString()
    });
  } catch (error) {
    response.status(500).json({
      error: 'Text analysis failed.',
      details: error instanceof Error ? error.message : 'Unknown error.'
    });
  }
});

app.get('*', (_request, response) => {
  response.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(port, () => {
  console.log(`Ellipsis Health MVP listening on http://localhost:${port}`);
});

function analyzeTranscriptHeuristically(transcript) {
  const text = transcript.toLowerCase();
  const scores = {
    depression: keywordScore(text, [
      { cue: 'empty', weight: 1 },
      { cue: 'hopeless', weight: 1.4 },
      { cue: 'numb', weight: 1 },
      { cue: 'drained', weight: 1 },
      { cue: 'worthless', weight: 1.6 },
      { cue: 'tired', weight: 0.6 },
      { cue: 'sad', weight: 0.8 },
      { cue: 'barely sleeping', weight: 1.2 },
      { cue: 'cannot get out of bed', weight: 1.8 }
    ]),
    anxiety: keywordScore(text, [
      { cue: 'anxious', weight: 1.2 },
      { cue: 'worried', weight: 0.9 },
      { cue: 'panic', weight: 1.4 },
      { cue: 'overthinking', weight: 1.1 },
      { cue: 'uneasy', weight: 0.8 },
      { cue: 'stressed', weight: 0.8 },
      { cue: 'nervous', weight: 0.8 },
      { cue: 'racing thoughts', weight: 1.3 },
      { cue: 'can\'t switch off', weight: 1.2 }
    ]),
    stress: keywordScore(text, [
      { cue: 'deadline', weight: 0.8 },
      { cue: 'pressure', weight: 0.8 },
      { cue: 'overwhelmed', weight: 1.2 },
      { cue: 'burned out', weight: 1.5 },
      { cue: 'exhausted', weight: 1 },
      { cue: 'too much', weight: 0.7 },
      { cue: 'no bandwidth', weight: 1.2 },
      { cue: 'spread thin', weight: 1.1 }
    ])
  };

  const rankedScores = Object.entries(scores).sort((left, right) => right[1] - left[1]);
  const dominant = rankedScores[0];
  const hasMixedCues = rankedScores[1] && dominant[1] > 0 && Math.abs(dominant[1] - rankedScores[1][1]) <= 0.25;
  const negativity = Math.min(1, (scores.depression + scores.anxiety + scores.stress) / 4.5);
  const concernLevel = negativity >= 0.8 ? 'high' : negativity >= 0.28 ? 'moderate' : 'low';

  return {
    source: 'heuristic',
    emotionalState: dominant[1] > 0 ? (hasMixedCues ? 'mixed distress' : dominant[0]) : 'steady',
    concernLevel,
    sentimentScore: Number((0.15 - negativity).toFixed(2)),
    rationale: dominant[1] > 0
      ? hasMixedCues
        ? 'Keyword patterns suggest overlapping anxiety, stress, or low-mood cues.'
        : `Keyword patterns suggest ${dominant[0]} cues in the transcript.`
      : 'Transcript appears broadly neutral with limited distress language.',
    supportiveMessage: concernLevel === 'high'
      ? 'Consider surfacing supportive resources or suggesting the user check in with someone they trust.'
      : 'Encourage another short check-in tomorrow to build a trend line.'
  };
}

function keywordScore(text, words) {
  return words.reduce((score, entry) => score + (text.includes(entry.cue) ? entry.weight : 0), 0);
}

async function analyzeTranscriptWithClaude(transcript) {
  const message = await anthropic.messages.create({
    model: anthropicModel,
    max_tokens: 400,
    temperature: 0.2,
    system: [
      'You analyze short daily mental wellness voice-note transcripts for a hackathon MVP.',
      'You are not diagnosing or making medical claims.',
      'Return only valid JSON with keys: emotionalState, concernLevel, sentimentScore, rationale, supportiveMessage.'
    ].join(' '),
    messages: [
      {
        role: 'user',
        content: [
          {
            type: 'text',
            text: [
              'Transcript:',
              transcript,
              '',
              'Requirements:',
              '- concernLevel must be one of low, moderate, high',
              '- sentimentScore must be a number from -1 to 1',
              '- rationale must be under 35 words',
              '- supportiveMessage must be under 30 words',
              '- no markdown, no code fences'
            ].join('\n')
          }
        ]
      }
    ]
  });

  const text = message.content
    .filter((block) => block.type === 'text')
    .map((block) => block.text)
    .join('')
    .trim();

  const parsed = JSON.parse(stripJsonEnvelope(text));

  return {
    source: 'claude',
    emotionalState: String(parsed.emotionalState || 'steady'),
    concernLevel: normalizeConcernLevel(parsed.concernLevel),
    sentimentScore: clampNumber(parsed.sentimentScore, -1, 1),
    rationale: String(parsed.rationale || 'Claude analyzed the transcript for emotional cues.'),
    supportiveMessage: String(parsed.supportiveMessage || 'Invite the user to check in again tomorrow.')
  };
}

function stripJsonEnvelope(text) {
  return text.replace(/^```json\s*/i, '').replace(/^```\s*/i, '').replace(/```$/i, '').trim();
}

function normalizeConcernLevel(value) {
  const normalized = String(value || 'low').toLowerCase();
  if (normalized === 'high' || normalized === 'moderate' || normalized === 'low') {
    return normalized;
  }

  return 'low';
}

function clampNumber(value, min, max) {
  const parsed = Number(value);
  if (Number.isNaN(parsed)) {
    return 0;
  }

  return Math.max(min, Math.min(max, Number(parsed.toFixed(2))));
}
