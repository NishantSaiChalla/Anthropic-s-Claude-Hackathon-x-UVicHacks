import dotenv from 'dotenv';
import express from 'express';
import multer from 'multer';
import OpenAI from 'openai';
import { toFile } from 'openai/uploads';
import path from 'path';
import { fileURLToPath } from 'url';
import { HfInference } from '@huggingface/inference';

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
const openAiTalkModel = process.env.OPENAI_TALK_MODEL || 'gpt-4o';
const openAiTtsModel = process.env.OPENAI_TTS_MODEL || 'tts-1';
const openAiTtsVoice = process.env.OPENAI_TTS_VOICE || 'nova';
const huggingFaceApiKey = String(process.env.HUGGINGFACE_API_KEY || '').trim();
const huggingFaceModel = process.env.HUGGINGFACE_EMOTION_MODEL || 'j-hartmann/emotion-english-distilroberta-base';

const openai = isConfiguredApiKey(openAiApiKey)
  ? new OpenAI({ apiKey: openAiApiKey })
  : null;
const huggingFaceConfigured = isConfiguredApiKey(huggingFaceApiKey);

app.use(express.json({ limit: '1mb' }));
app.use(express.static(path.join(__dirname, 'public')));

app.get('/api/health', (_request, response) => {
  response.json({
    ok: true,
    openAiConfigured: Boolean(openai),
    huggingFaceConfigured,
    huggingFaceModel: huggingFaceConfigured ? huggingFaceModel : null,
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

    const emotionResult = await detectEmotionsWithHuggingFace(transcript);
    const emotionContext = emotionResult?.dominant?.label || 'neutral';
    
    const analysis = await analyzeTranscriptWithPreferredModel(transcript, mode, emotionContext);

    response.json({
      transcript,
      textAnalysis: {
        ...analysis,
        model: getAnalysisModelName()
      },
      emotionDetection: emotionResult,
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
    const mode = String(request.body?.mode || 'advanced');
    const emotionResult = await detectEmotionsWithHuggingFace(transcript);
    const emotionContext = emotionResult?.dominant?.label || 'neutral';
    
    const analysis = await analyzeTranscriptWithPreferredModel(transcript, mode, emotionContext);

    response.json({
      ...analysis,
      emotionDetection: emotionResult,
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

app.post('/api/emotion-detect', async (request, response) => {
  const text = String(request.body?.text || '').trim();

  if (!text) {
    response.status(400).json({ error: 'Text is required.' });
    return;
  }

  try {
    const result = await detectEmotionsWithHuggingFace(text);
    response.json(result);
  } catch (error) {
    response.status(500).json({
      error: 'Emotion detection failed.',
      details: error instanceof Error ? error.message : 'Unknown error.'
    });
  }
});

// Streaming chat — emits sentences via SSE as GPT generates them so TTS can start immediately
app.post('/api/chat-stream', async (request, response) => {
  const messages = Array.isArray(request.body?.messages) ? request.body.messages : [];
  const turnCount = Math.max(0, Number(request.body?.turnCount || 0));

  response.set({
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive'
  });

  const send = (obj) => response.write(`data: ${JSON.stringify(obj)}\n\n`);

  if (!openai) {
    const fallbacks = [
      "Hi, I'm really glad you're here. How are you feeling right now, in this moment?",
      "I hear you. Can you tell me a bit more about what's been sitting with you lately?",
      "That makes a lot of sense. How long have you been carrying this feeling?",
      "Thank you for trusting me with that. What do you think is underneath it all?",
      "It sounds like you've been doing a lot on your own. What would feel like even a small relief right now?",
      "I really appreciate how open you've been today. It takes courage to look at these things honestly. What's one small, gentle thing you could do for yourself after we finish?"
    ];
    const userTurns = messages.filter((m) => m.role === 'user').length;
    const idx = Math.min(userTurns, fallbacks.length - 1);
    send({ sentence: fallbacks[idx] });
    send({ done: true, session_complete: idx === fallbacks.length - 1 });
    response.end();
    return;
  }

  // Phase guidance steers the therapist through the session arc
  let phaseGuidance;
  if (turnCount === 0) {
    phaseGuidance = 'PHASE — Opening: Greet the user warmly, create a safe and non-judgmental space, and ask one gentle open question about how they are feeling today.';
  } else if (turnCount <= 2) {
    phaseGuidance = 'PHASE — Building rapport: Listen carefully. Reflect back what you are hearing in your own words to show you understand. Gently probe with one open question to identify the core thing that is weighing on them.';
  } else if (turnCount <= 5) {
    phaseGuidance = 'PHASE — Exploring: You are starting to understand their core concern. Validate their experience, help them name what they are feeling more precisely, and gently guide them deeper. Do not rush to solutions yet.';
  } else if (turnCount <= 8) {
    phaseGuidance = 'PHASE — Working through: You have a clear picture of the issue. Help them examine it from a new angle, find one concrete insight, or identify a small coping step they could actually take. Keep responses warm and human.';
  } else {
    phaseGuidance = 'PHASE — Closing: You have explored the issue together. Reflect the progress made in this conversation, offer one last piece of affirmation or a small concrete next step, and bring the session to a warm and supportive close. When you are done, append exactly the token [END_SESSION] at the very end of your final sentence — nothing after it.';
  }

  try {
    const systemPrompt = [
      'You are a warm, empathetic therapist conducting a voice-based emotional check-in session.',
      'Use evidence-based techniques: accurately reflect feelings, validate emotions without judgment, and gently explore the "why" beneath what the user shares.',
      'Ask open-ended questions that invite genuine depth — not yes/no answers.',
      'Mirror the user\'s emotional tone: slow and grounding when anxious or overwhelmed, warm and present when low.',
      'Never interpret or label their experience prematurely — follow their lead and check your understanding.',
      'Keep every response to 2-3 natural spoken sentences. Always end with exactly one thoughtful open question.',
      'Never use lists, bullet points, headers, or clinical jargon. Speak as a caring human therapist would — unhurried and genuinely curious.',
      'Do not give advice unless explicitly invited. Your role is to listen, reflect, and guide the user toward their own insight.',
      'If the user expresses serious distress, suicidal thoughts, or self-harm, compassionately encourage them to contact a crisis line or mental health professional.',
      '',
      phaseGuidance
    ].join(' ');

    const stream = await openai.chat.completions.create({
      model: openAiTalkModel,
      temperature: 0.8,
      max_tokens: 200,
      stream: true,
      messages: [{ role: 'system', content: systemPrompt }, ...messages]
    });

    let buffer = '';
    let sessionComplete = false;
    // Sentence boundary: ends with . ! ? optionally followed by quote/space
    const sentenceEnd = /[.!?]["']?\s/;

    for await (const chunk of stream) {
      const token = chunk.choices[0]?.delta?.content || '';
      buffer += token;

      // Detect therapist signalling end of session
      if (buffer.includes('[END_SESSION]')) {
        const beforeMarker = buffer.slice(0, buffer.indexOf('[END_SESSION]')).trim();
        buffer = '';
        sessionComplete = true;

        // Flush everything before the marker as sentences
        if (beforeMarker) {
          let tmp = beforeMarker;
          let m;
          while ((m = sentenceEnd.exec(tmp)) !== null) {
            const s = tmp.slice(0, m.index + 1).trim();
            tmp = tmp.slice(m.index + m[0].length).trimStart();
            if (s) send({ sentence: s });
          }
          if (tmp.trim()) send({ sentence: tmp.trim() });
        }
        break;
      }

      // Flush complete sentences as they arrive
      let match;
      while ((match = sentenceEnd.exec(buffer)) !== null) {
        const sentence = buffer.slice(0, match.index + 1).trim();
        buffer = buffer.slice(match.index + match[0].length).trimStart();
        if (sentence) send({ sentence });
      }
    }

    // Flush any remaining text (only if session didn't signal complete)
    if (!sessionComplete) {
      const remaining = buffer.trim();
      if (remaining) send({ sentence: remaining });
    }

    send({ done: true, ...(sessionComplete && { session_complete: true }) });
    response.end();
  } catch (error) {
    send({ error: 'Chat failed.' });
    response.end();
  }
});

app.get('/api/tts', async (request, response) => {
  const text = String(request.query.text || '').trim().slice(0, 500);

  if (!text) {
    response.status(400).end();
    return;
  }

  if (!openai) {
    response.status(503).json({ error: 'OPENAI_API_KEY is required for TTS.' });
    return;
  }

  try {
    const speech = await openai.audio.speech.create({
      model: openAiTtsModel,
      voice: openAiTtsVoice,
      input: text,
      response_format: 'mp3'
    });

    const buffer = Buffer.from(await speech.arrayBuffer());
    response.set('Content-Type', 'audio/mpeg');
    response.set('Cache-Control', 'public, max-age=300');
    response.send(buffer);
  } catch (error) {
    response.status(500).json({
      error: 'TTS failed.',
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
              '- rationale should explain your thoughts gently, like a caring friend (under 40 words)',
              '- feedback should feel like a warm, supportive text message from a best friend helping you overcome a problem (under 45 words)',
              '- wellnessTips must contain 2 or 3 very simple, easy actions, each under 18 words',
              '- supportiveMessage must be an encouraging, friendly sign-off (under 30 words)',
              '- no medical jargon, keep it extremely natural and conversational',
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

async function analyzeTranscriptWithPreferredModel(transcript, mode = 'advanced', emotionContext = 'neutral') {
  if (mode === 'basic') {
    return analyzeTranscriptHeuristically(transcript, emotionContext);
  }

  if (openai) {
    return analyzeTranscriptWithOpenAI(transcript, emotionContext);
  }
  return analyzeTranscriptHeuristically(transcript, emotionContext);
}

function analyzeTranscriptHeuristically(transcript, emotionContext = 'neutral') {
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

async function analyzeTranscriptWithOpenAI(transcript, emotionContext = 'neutral') {
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
          `- rationale should explain the emotional cues gently, addressing their dominant emotion: ${emotionContext} (under 35 words)`,
          `- feedback should read like a warm, supportive text message from a caring friend comforting them through their ${emotionContext} (under 45 words)`,
          '- wellnessTips must contain 2 or 3 very simple, easy actions, each under 18 words',
          '- supportiveMessage must be an encouraging, friendly sign-off (under 30 words)',
          '- no medical jargon, keep it extremely natural and conversational',
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

async function detectEmotionsWithHuggingFace(text) {
  if (!huggingFaceConfigured) {
    return {
      source: 'unavailable',
      configured: false,
      emotions: null,
      dominant: null,
      message: 'HUGGINGFACE_API_KEY is not configured. Add it to .env to enable ML-based emotion detection.'
    };
  }

  try {
    const hf = new HfInference(huggingFaceApiKey);
    const result = await hf.textClassification({
      model: huggingFaceModel,
      inputs: text
    });

    const emotions = {};
    let dominant = { label: 'neutral', score: 0 };

    for (const entry of result) {
      const label = String(entry.label || '').toLowerCase();
      const score = Number(entry.score || 0);
      emotions[label] = Number(score.toFixed(4));
      if (score > dominant.score) {
        dominant = { label, score };
      }
    }

    return {
      source: 'huggingface',
      configured: true,
      model: huggingFaceModel,
      emotions,
      dominant: {
        label: dominant.label,
        score: Number(dominant.score.toFixed(4))
      }
    };
  } catch (error) {
    console.error('HuggingFace emotion detection failed:', error.message);
    return {
      source: 'huggingface-error',
      configured: true,
      emotions: null,
      dominant: null,
      error: error instanceof Error ? error.message : 'Unknown error.'
    };
  }
}
