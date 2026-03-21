import dotenv from 'dotenv';
import express from 'express';
import multer from 'multer';
import OpenAI from 'openai';
import { toFile } from 'openai/uploads';
import path from 'path';
import { fileURLToPath } from 'url';
import Anthropic from '@anthropic-ai/sdk';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const app = express();
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 12 * 1024 * 1024
  }
});

const port = Number(process.env.PORT || 3000);
const openAiApiKey = String(process.env.OPENAI_API_KEY || '').trim();
const openAiTranscriptionModel = process.env.OPENAI_TRANSCRIPTION_MODEL || 'gpt-4o-mini-transcribe';
const openAiAnalysisModel = process.env.OPENAI_ANALYSIS_MODEL || 'gpt-4.1-mini';
const openAiAudioModel = process.env.OPENAI_AUDIO_MODEL || 'gpt-4o-audio-preview';
const anthropicApiKey = String(process.env.ANTHROPIC_API_KEY || '').trim();
const anthropicModel = process.env.ANTHROPIC_MODEL || 'claude-3-7-sonnet-latest';

const openai = isConfiguredApiKey(openAiApiKey)
  ? new OpenAI({ apiKey: openAiApiKey })
  : null;
const anthropic = isConfiguredApiKey(anthropicApiKey)
  ? new Anthropic({ apiKey: anthropicApiKey })
  : null;

app.use(express.json({ limit: '1mb' }));
app.use(express.static(path.join(__dirname, 'public')));

app.get('/api/health', (_request, response) => {
  response.json({
    ok: true,
    openAiConfigured: Boolean(openai),
    anthropicConfigured: Boolean(anthropic),
    transcriptionModel: openai ? openAiTranscriptionModel : null,
    analysisModel: getAnalysisModelName()
  });
});

app.post('/api/transcribe-recording', upload.single('audio'), async (request, response) => {
  if (!request.file) {
    response.status(400).json({ error: 'Audio recording is required.' });
    return;
  }

  if (!openai) {
    response.status(400).json({ error: 'OPENAI_API_KEY is required for recording transcription.' });
    return;
  }

  try {
    const transcript = await transcribeAudioWithOpenAI(request.file);

    response.json({
      transcript,
      source: 'openai-audio',
      model: openAiTranscriptionModel,
      transcribedAt: new Date().toISOString()
    });
  } catch (error) {
    response.status(500).json({
      error: 'Recording transcription failed.',
      details: error instanceof Error ? error.message : 'Unknown error.'
    });
  }
});

app.post('/api/analyze-recording', upload.single('audio'), async (request, response) => {
  if (!request.file) {
    response.status(400).json({ error: 'Audio recording is required.' });
    return;
  }

  if (!openai) {
    response.status(400).json({ error: 'OPENAI_API_KEY is required for recording analysis.' });
    return;
  }

  try {
    const transcriptDraft = String(request.body?.transcript || '').trim();
    const mode = String(request.body?.mode || 'advanced');

    if (mode === 'direct') {
      const directResult = await analyzeAudioDirectly(request.file);
      response.json({
        transcript: directResult.transcript,
        textAnalysis: { ...directResult.analysis, model: openAiAudioModel },
        transcription: { source: 'openai-audio-direct', model: openAiAudioModel },
        analyzedAt: new Date().toISOString()
      });
      return;
    }

    const transcript = (await transcribeAudioWithOpenAI(request.file)) || transcriptDraft;

    if (!transcript) {
      response.status(400).json({ error: 'The recording could not be transcribed.' });
      return;
    }

    const analysis = await analyzeTranscriptWithPreferredModel(transcript, mode);

    response.json({
      transcript,
      textAnalysis: {
        ...analysis,
        model: getAnalysisModelName()
      },
      transcription: {
        source: 'openai-audio',
        model: openAiTranscriptionModel
      },
      analyzedAt: new Date().toISOString()
    });
  } catch (error) {
    response.status(500).json({
      error: 'Recording analysis failed.',
      details: error instanceof Error ? error.message : 'Unknown error.'
    });
  }
});

app.post('/api/analyze-text', async (request, response) => {
  const transcript = String(request.body?.transcript || '').trim();

  if (!transcript) {
    response.status(400).json({ error: 'Transcript is required.' });
    return;
  }

  try {
    const analysis = await analyzeTranscriptWithPreferredModel(transcript, String(request.body?.mode || 'advanced'));

    response.json({
      ...analysis,
      model: getAnalysisModelName(),
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

async function analyzeAudioDirectly(file) {
  const base64 = file.buffer.toString('base64');
  const fmt = getAudioFormat(file.mimetype);

  const completion = await openai.chat.completions.create({
    model: openAiAudioModel,
    modalities: ['text'],
    messages: [
      {
        role: 'system',
        content: [
          'You analyze short daily mental wellness voice-note recordings for a hackathon MVP.',
          'Listen to the audio directly. Pay attention not just to the words spoken, but also to vocal',
          'tone, pace, energy level, hesitations, tremors, and emotional quality in the voice itself.',
          'You are not diagnosing or making medical claims.',
          'Respond with a single raw JSON object — no markdown, no code fences — with keys:',
          'transcript, emotionalState, concernLevel, sentimentScore, rationale, feedback, wellnessTips, supportiveMessage.',
          'transcript must be the verbatim speech content. wellnessTips must be an array of 2 or 3 short practical tips.'
        ].join(' ')
      },
      {
        role: 'user',
        content: [
          {
            type: 'input_audio',
            input_audio: { data: base64, format: fmt }
          },
          {
            type: 'text',
            text: [
              'Transcribe and analyze this voice note. Return only a raw JSON object.',
              'Requirements:',
              '- concernLevel must be one of low, moderate, high',
              '- sentimentScore must be a number from -1 to 1',
              '- rationale must mention both vocal delivery AND content, under 40 words',
              '- feedback must be under 45 words and sound supportive but direct',
              '- wellnessTips must contain 2 or 3 items, each under 18 words',
              '- supportiveMessage must be under 30 words',
              '- no markdown, no code fences, no extra text'
            ].join('\n')
          }
        ]
      }
    ]
  });

  let parsed;
  try {
    parsed = JSON.parse(stripJsonEnvelope(completion.choices[0]?.message?.content || '{}'));
  } catch {
    return {
      transcript: '',
      analysis: { ...analyzeTranscriptHeuristically(''), source: 'heuristic' }
    };
  }

  return {
    transcript: String(parsed.transcript || '').trim(),
    analysis: {
      source: 'openai-audio-direct',
      emotionalState: String(parsed.emotionalState || 'steady'),
      concernLevel: normalizeConcernLevel(parsed.concernLevel),
      sentimentScore: clampNumber(parsed.sentimentScore, -1, 1),
      rationale: String(parsed.rationale || 'GPT-4o analyzed the audio recording directly.'),
      feedback: String(parsed.feedback || 'Take one supportive step and check back in later.'),
      wellnessTips: normalizeTips(parsed.wellnessTips),
      supportiveMessage: String(parsed.supportiveMessage || 'Check in again tomorrow to build a trend.')
    }
  };
}

function getAudioFormat(mimetype) {
  const base = String(mimetype || 'audio/webm').split(';')[0].split('/')[1] || 'webm';
  const aliases = { mpeg: 'mp3', 'x-wav': 'wav', 'x-m4a': 'm4a' };
  return aliases[base] || base;
}

function getAnalysisModelName() {
  if (openai) {
    return openAiAnalysisModel;
  }

  if (anthropic) {
    return anthropicModel;
  }

  return 'local-heuristic';
}

async function transcribeAudioWithOpenAI(file) {
  const audioFile = await toFile(file.buffer, file.originalname || 'voice-note.webm', {
    type: file.mimetype || 'audio/webm'
  });

  const transcript = await openai.audio.transcriptions.create({
    file: audioFile,
    model: openAiTranscriptionModel
  });

  return String(transcript.text || '').trim();
}

async function analyzeTranscriptWithPreferredModel(transcript, mode = 'advanced') {
  if (mode === 'basic') {
    return analyzeTranscriptHeuristically(transcript);
  }

  if (openai) {
    return analyzeTranscriptWithOpenAI(transcript);
  }

  if (anthropic) {
    return analyzeTranscriptWithClaude(transcript);
  }

  return analyzeTranscriptHeuristically(transcript);
}

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
  const emotionalState = dominant?.[1] > 0 ? dominant[0] : 'steady';
  const hasMixedCues = rankedScores[1] && dominant[1] > 0 && Math.abs(dominant[1] - rankedScores[1][1]) <= 0.25;
  const negativity = Math.min(1, (scores.depression + scores.anxiety + scores.stress) / 4.5);
  const concernLevel = negativity >= 0.8 ? 'high' : negativity >= 0.28 ? 'moderate' : 'low';

  return {
    source: 'heuristic',
    emotionalState: dominant?.[1] > 0 ? (hasMixedCues ? 'mixed distress' : emotionalState) : 'steady',
    concernLevel,
    sentimentScore: Number((0.15 - negativity).toFixed(2)),
    rationale: dominant?.[1] > 0
      ? hasMixedCues
        ? 'Keyword patterns suggest overlapping anxiety, stress, or low-mood cues.'
        : `Keyword patterns suggest ${emotionalState} cues in the transcript.`
      : 'Transcript appears broadly neutral with limited distress language.',
    feedback: concernLevel === 'high'
      ? 'Your recording suggests a heavier emotional load today. Keep expectations narrow and reduce nonessential pressure where possible.'
      : concernLevel === 'moderate'
        ? 'Your recording suggests some strain today. A small reset and a lower-friction plan could help stabilize the day.'
        : 'Your recording sounds relatively steady today. Keep the routine that is helping you stay grounded.',
    wellnessTips: buildHeuristicTips({ concernLevel, emotionalState }),
    supportiveMessage: concernLevel === 'high'
      ? 'Consider checking in with someone you trust if the strain keeps building.'
      : 'Encourage another short check-in tomorrow to build a trend line.'
  };
}

function keywordScore(text, words) {
  return words.reduce((score, entry) => score + (text.includes(entry.cue) ? entry.weight : 0), 0);
}

async function analyzeTranscriptWithOpenAI(transcript) {
  const completion = await openai.chat.completions.create({
    model: openAiAnalysisModel,
    temperature: 0.2,
    response_format: { type: 'json_object' },
    messages: [
      {
        role: 'system',
        content: [
          'You analyze short daily mental wellness voice-note transcripts for a hackathon MVP.',
          'You are not diagnosing or making medical claims.',
          'Return valid JSON only with keys emotionalState, concernLevel, sentimentScore, rationale, feedback, wellnessTips, supportiveMessage.',
          'wellnessTips must be an array of 2 or 3 short practical tips.'
        ].join(' ')
      },
      {
        role: 'user',
        content: [
          'Transcript:',
          transcript,
          '',
          'Requirements:',
          '- concernLevel must be one of low, moderate, high',
          '- sentimentScore must be a number from -1 to 1',
          '- rationale must be under 35 words',
          '- feedback must be under 45 words and sound supportive but direct',
          '- wellnessTips must contain 2 or 3 items, each under 18 words',
          '- supportiveMessage must be under 30 words',
          '- no markdown, no code fences'
        ].join('\n')
      }
    ]
  });

  let parsed;
  try {
    parsed = JSON.parse(stripJsonEnvelope(completion.choices[0]?.message?.content || '{}'));
  } catch {
    return {
      ...analyzeTranscriptHeuristically(transcript),
      source: 'heuristic'
    };
  }

  return {
    source: 'openai',
    emotionalState: String(parsed.emotionalState || 'steady'),
    concernLevel: normalizeConcernLevel(parsed.concernLevel),
    sentimentScore: clampNumber(parsed.sentimentScore, -1, 1),
    rationale: String(parsed.rationale || 'GPT analyzed the transcript for emotional cues.'),
    feedback: String(parsed.feedback || 'The recording suggests a useful moment to pause and choose one supportive next step.'),
    wellnessTips: normalizeTips(parsed.wellnessTips),
    supportiveMessage: String(parsed.supportiveMessage || 'Invite the user to check in again tomorrow.')
  };
}

async function analyzeTranscriptWithClaude(transcript) {
  const message = await anthropic.messages.create({
    model: anthropicModel,
    max_tokens: 400,
    temperature: 0.2,
    system: [
      'You analyze short daily mental wellness voice-note transcripts for a hackathon MVP.',
      'You are not diagnosing or making medical claims.',
      'Return only valid JSON with keys: emotionalState, concernLevel, sentimentScore, rationale, feedback, wellnessTips, supportiveMessage.',
      'wellnessTips must be an array of 2 or 3 short practical tips.'
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
              '- feedback must be under 45 words and sound supportive but direct',
              '- wellnessTips must contain 2 or 3 items, each under 18 words',
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

  let parsed;
  try {
    parsed = JSON.parse(stripJsonEnvelope(text));
  } catch {
    return {
      ...analyzeTranscriptHeuristically(transcript),
      source: 'heuristic'
    };
  }

  return {
    source: 'claude',
    emotionalState: String(parsed.emotionalState || 'steady'),
    concernLevel: normalizeConcernLevel(parsed.concernLevel),
    sentimentScore: clampNumber(parsed.sentimentScore, -1, 1),
    rationale: String(parsed.rationale || 'Claude analyzed the transcript for emotional cues.'),
    feedback: String(parsed.feedback || 'The recording suggests a useful moment to slow down and choose one supportive action.'),
    wellnessTips: normalizeTips(parsed.wellnessTips),
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

function normalizeTips(value) {
  if (Array.isArray(value)) {
    const cleaned = value
      .map((entry) => String(entry || '').trim())
      .filter(Boolean)
      .slice(0, 3);

    if (cleaned.length >= 2) {
      return cleaned;
    }
  }

  return [
    'Pause for one slow minute of breathing.',
    'Shrink the next task to one manageable step.',
    'Check in again later if the strain increases.'
  ];
}

function buildHeuristicTips({ concernLevel, emotionalState }) {
  if (concernLevel === 'high') {
    return [
      'Reduce today to one or two essential tasks.',
      'Reach out to someone you trust for support.',
      'Step away briefly and reset your breathing.'
    ];
  }

  if (concernLevel === 'moderate') {
    return emotionalState === 'anxiety' || emotionalState === 'mixed distress'
      ? [
          'Slow your breathing for one minute.',
          'Write the next smallest task only.',
          'Limit one avoidable stress trigger today.'
        ]
      : [
          'Take a short walk or body reset.',
          'Keep your next step simple and concrete.',
          'Check in again tonight for any change.'
        ];
  }

  return [
    'Keep the routine that is helping today.',
    'Protect one short recovery break later.',
    'Record another check-in tomorrow for trend tracking.'
  ];
}

function isConfiguredApiKey(key) {
  if (!key) {
    return false;
  }

  const normalized = key.toLowerCase();
  const placeholders = new Set([
    'your_api_key_here',
    'your-api-key-here',
    'your_openai_api_key_here',
    'changeme'
  ]);

  return !placeholders.has(normalized);
}
