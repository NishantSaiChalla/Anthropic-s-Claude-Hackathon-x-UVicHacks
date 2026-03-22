const RECORDING_LIMIT_SECONDS = 30;
const HISTORY_KEY = 'ellipsis-health-history';
const SPEECH_WARNING_MESSAGE = 'Browser speech recognition is unavailable. Recordings still work, but type your transcript manually.';

const RECORDING_MODE = 'record';
const UPLOAD_MODE = 'upload';
const TALK_MODE = 'talk';
const ADVANCED_ANALYSIS = 'advanced';

const elements = {
  startButton: document.querySelector('#startButton'),
  stopButton: document.querySelector('#stopButton'),
  transcribeButton: document.querySelector('#transcribeButton'),
  modeRecordButton: document.querySelector('#modeRecordButton'),
  modeUploadButton: document.querySelector('#modeUploadButton'),
  recordingPanel: document.querySelector('#recordingPanel'),
  recordingButtons: document.querySelector('#recordingButtons'),
  uploadPanel: document.querySelector('#uploadPanel'),
  uploadDropZone: document.querySelector('#uploadDropZone'),
  chooseFileButton: document.querySelector('#chooseFileButton'),
  audioUploadInput: document.querySelector('#audioUploadInput'),
  speechWarning: document.querySelector('#speechWarning'),
  uploadInfo: document.querySelector('#uploadInfo'),
  analyzeButton: document.querySelector('#analyzeButton'),
  clearTranscriptButton: document.querySelector('#clearTranscriptButton'),
  transcriptInput: document.querySelector('#transcriptInput'),
  transcriptStatus: document.querySelector('#transcriptStatus'),
  analysisStatus: document.querySelector('#analysisStatus'),
  timerPill: document.querySelector('#timerPill'),
  meterFill: document.querySelector('#meterFill'),
  meterCaption: document.querySelector('#meterCaption'),
  playback: document.querySelector('#playback'),
  resultEmpty: document.querySelector('#resultEmpty'),
  resultCard: document.querySelector('#resultCard'),
  resultBanner: document.querySelector('#resultBanner'),
  resultHeadline: document.querySelector('#resultHeadline'),
  resultSummary: document.querySelector('#resultSummary'),

  feedbackNarrative: document.querySelector('#feedbackNarrative'),
  tipsList: document.querySelector('#tipsList'),
  deepSummaryToggle: document.querySelector('#deepSummaryToggle'),
  deepSummarySection: document.querySelector('#deepSummarySection'),
  rationaleText: document.querySelector('#rationaleText'),
  emotionCard: document.querySelector('#emotionCard'),
  emotionModelLabel: document.querySelector('#emotionModelLabel'),
  emotionBars: document.querySelector('#emotionBars'),
  emotionDominant: document.querySelector('#emotionDominant'),
  analysisAdvancedButton: document.querySelector('#analysisAdvancedButton'),
  analysisModeHint: document.querySelector('#analysisModeHint'),
  historyList: document.querySelector('#historyList'),
  sparklineWrap: document.querySelector('#sparklineWrap'),
  historySparkline: document.querySelector('#historySparkline'),
  historyNudge: document.querySelector('#historyNudge'),
  streakCounter: document.querySelector('#streakCounter'),
  streakNumber: document.querySelector('#streakNumber'),
  restDayButton: document.querySelector('#restDayButton'),
  resetTrendlineButton: document.querySelector('#resetTrendlineButton'),
  modeTalkButton: document.querySelector('#modeTalkButton'),
  talkPanel: document.querySelector('#talkPanel'),
  conversationList: document.querySelector('#conversationList'),
  talkPhasePill: document.querySelector('#talkPhasePill'),
  talkTurnCount: document.querySelector('#talkTurnCount'),
  talkLiveTranscript: document.querySelector('#talkLiveTranscript'),
  talkStatus: document.querySelector('#talkStatus'),
  talkMicButton: document.querySelector('#talkMicButton'),
  endTalkButton: document.querySelector('#endTalkButton')
};

let mediaRecorder = null;
let mediaStream = null;
let audioChunks = [];
let audioBlob = null;
let audioFileName = 'voice-note.webm';
let playbackUrl = null;
let recordingTimer = null;
let meterLoop = null;
let secondsRemaining = RECORDING_LIMIT_SECONDS;
let recognition = null;
let finalTranscript = '';
let audioContext = null;
let analyser = null;
let sourceNode = null;
let isTranscribing = false;
let hasRecordedTranscription = false;
let captureMode = RECORDING_MODE;
let analysisMode = ADVANCED_ANALYSIS;
let serverCapabilities = {
  openAiConfigured: false,
  huggingFaceConfigured: false,
  huggingFaceModel: null,
  analysisModel: 'local-heuristic',
  transcriptionModel: null
};

// Talk mode state
let conversationHistory = [];
let talkRecognition = null;
let talkAudio = null;
let talkAbortController = null;
let isBotSpeaking = false;
let isTalkListening = false;
let pendingSessionComplete = false;

boot();

async function boot() {
  renderHistory();
  wireEvents();
  await loadServerCapabilities();
}

function wireEvents() {
  elements.startButton.addEventListener('click', startRecording);
  elements.stopButton.addEventListener('click', stopRecording);
  elements.transcribeButton.addEventListener('click', () => {
    transcribeRecording({ forceRefresh: true });
  });
  elements.modeRecordButton.addEventListener('click', () => setCaptureMode(RECORDING_MODE));
  elements.modeUploadButton.addEventListener('click', () => setCaptureMode(UPLOAD_MODE));
  elements.modeTalkButton.addEventListener('click', () => setCaptureMode(TALK_MODE));
  elements.talkMicButton.addEventListener('click', startUserListening);
  elements.endTalkButton.addEventListener('click', endTalkSession);
  elements.analysisAdvancedButton.addEventListener('click', () => setAnalysisMode(ADVANCED_ANALYSIS));
  elements.deepSummaryToggle.addEventListener('click', () => {
    const isOpen = !elements.deepSummarySection.hidden;
    elements.deepSummarySection.hidden = isOpen;
    elements.deepSummaryToggle.textContent = isOpen ? 'See deeper analysis ▾' : 'Hide deeper analysis ▴';
    elements.deepSummaryToggle.classList.toggle('open', !isOpen);
  });
  elements.chooseFileButton.addEventListener('click', () => elements.audioUploadInput.click());
  elements.audioUploadInput.addEventListener('change', (event) => {
    const file = event.target.files?.[0];
    if (file) {
      handleUploadedFile(file);
    }
  });
  elements.analyzeButton.addEventListener('click', analyzeEntry);
  elements.restDayButton.addEventListener('click', logRestDay);
  elements.resetTrendlineButton.addEventListener('click', resetTrendline);
  elements.clearTranscriptButton.addEventListener('click', () => {
    finalTranscript = '';
    elements.transcriptInput.value = '';
    setTranscriptStatus('Transcript cleared. Re-transcribe or type a summary.');
  });

  setupDropZone();
  refreshCapabilityWarnings();
}

function setCaptureMode(mode) {
  const leavingTalk = captureMode === TALK_MODE && mode !== TALK_MODE;

  if (captureMode === mode) {
    return;
  }

  captureMode = mode;
  const usingRecord = captureMode === RECORDING_MODE;
  const usingUpload = captureMode === UPLOAD_MODE;
  const usingTalk = captureMode === TALK_MODE;

  elements.modeRecordButton.classList.toggle('active', usingRecord);
  elements.modeRecordButton.setAttribute('aria-selected', String(usingRecord));
  elements.modeUploadButton.classList.toggle('active', usingUpload);
  elements.modeUploadButton.setAttribute('aria-selected', String(usingUpload));
  elements.modeTalkButton.classList.toggle('active', usingTalk);
  elements.modeTalkButton.setAttribute('aria-selected', String(usingTalk));

  elements.recordingPanel.hidden = !usingRecord;
  elements.recordingButtons.hidden = !usingRecord;
  elements.uploadPanel.hidden = !usingUpload;
  elements.talkPanel.hidden = !usingTalk;

  if (leavingTalk) {
    leaveTalkMode();
  }

  if (usingTalk) {
    enterTalkMode();
  } else if (!usingRecord) {
    stopRecording();
    updateTimer(RECORDING_LIMIT_SECONDS);
    if (usingUpload) elements.meterCaption.textContent = 'Upload mode ready';
  } else if (!audioBlob) {
    elements.meterCaption.textContent = 'Microphone idle';
  }

  refreshCapabilityWarnings();
}

function setAnalysisMode(mode) {
  analysisMode = mode;
  elements.analysisAdvancedButton.classList.toggle('active', mode === ADVANCED_ANALYSIS);
  elements.analysisAdvancedButton.setAttribute('aria-selected', String(mode === ADVANCED_ANALYSIS));
  elements.analysisModeHint.textContent = 'GPT-powered emotional analysis with personalised feedback and tips';
}

function setupDropZone() {
  const zone = elements.uploadDropZone;
  if (!zone) {
    return;
  }

  const preventDefaults = (event) => {
    event.preventDefault();
    event.stopPropagation();
  };

  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach((name) => {
    zone.addEventListener(name, preventDefaults);
  });

  ['dragenter', 'dragover'].forEach((name) => {
    zone.addEventListener(name, () => zone.classList.add('drag-over'));
  });

  ['dragleave', 'drop'].forEach((name) => {
    zone.addEventListener(name, () => zone.classList.remove('drag-over'));
  });

  zone.addEventListener('drop', (event) => {
    const file = event.dataTransfer?.files?.[0];
    if (file) {
      handleUploadedFile(file);
    }
  });

  zone.addEventListener('click', () => elements.audioUploadInput.click());
  zone.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      elements.audioUploadInput.click();
    }
  });
}

function refreshCapabilityWarnings() {
  const hasSpeechApi = Boolean(window.SpeechRecognition || window.webkitSpeechRecognition);
  const showSpeechWarning = captureMode === RECORDING_MODE && !hasSpeechApi;
  elements.speechWarning.hidden = !showSpeechWarning;
  elements.speechWarning.textContent = SPEECH_WARNING_MESSAGE;

  elements.uploadInfo.hidden = captureMode !== UPLOAD_MODE;
  elements.uploadInfo.textContent = serverCapabilities.openAiConfigured
    ? 'Uploaded audio can also be sent to GPT for transcription and analysis.'
    : 'Uploaded audio works, but add a transcript manually unless OPENAI_API_KEY is configured.';

  if (!audioBlob || isTranscribing) {
    elements.transcribeButton.disabled = true;
    return;
  }

  elements.transcribeButton.disabled = !serverCapabilities.openAiConfigured;
}

async function handleUploadedFile(file) {
  const looksAudio = file.type.startsWith('audio/') || file.type.startsWith('video/') || /\.(wav|mp3|m4a|ogg|webm|aac|flac|mp4|mov|mkv)$/i.test(file.name);
  if (!looksAudio) {
    elements.analysisStatus.textContent = 'Please choose a valid audio or video file.';
    return;
  }

  await finalizeRecording(file, { skipTranscription: !serverCapabilities.openAiConfigured });
  setTranscriptStatus(
    serverCapabilities.openAiConfigured
      ? 'Uploaded audio loaded. GPT transcription can run now.'
      : 'Uploaded audio loaded. Add a transcript manually before analysis.'
  );
  elements.analysisStatus.textContent = serverCapabilities.openAiConfigured
    ? `Loaded ${file.name}. Ready to transcribe or analyze.`
    : `Loaded ${file.name}. Add a transcript, then analyze.`;
}

async function loadServerCapabilities() {
  try {
    const response = await fetch('/api/health');
    if (!response.ok) {
      throw new Error('Health check failed.');
    }

    serverCapabilities = await response.json();
    setTranscriptStatus(
      serverCapabilities.openAiConfigured
        ? 'Record or upload audio to transcribe and analyze it with GPT.'
        : 'OPENAI_API_KEY is not configured yet. Record or upload audio and type a summary manually, or add the key to enable audio transcription.'
    );
  } catch (error) {
    console.error(error);
    setTranscriptStatus('Could not load AI service status. You can still type a summary manually.');
  }

  refreshCapabilityWarnings();
}

async function startRecording() {
  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      elements.analysisStatus.textContent = 'This browser does not support microphone capture.';
      return;
    }

    if (!window.MediaRecorder) {
      elements.analysisStatus.textContent = 'This browser does not support audio recording.';
      return;
    }

    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioChunks = [];
    audioBlob = null;
    audioFileName = 'voice-note.webm';
    finalTranscript = '';
    hasRecordedTranscription = false;
    updateTimer(RECORDING_LIMIT_SECONDS);
    elements.transcriptInput.value = '';
    elements.transcribeButton.disabled = true;
    setTranscriptStatus(
      serverCapabilities.openAiConfigured
        ? 'Recording started. The saved audio will be uploaded for GPT transcription after you stop.'
        : 'Recording started. Add OPENAI_API_KEY to enable server-side transcription.'
    );

    const supportedMimeType = getSupportedMimeType();
    mediaRecorder = supportedMimeType
      ? new MediaRecorder(mediaStream, { mimeType: supportedMimeType })
      : new MediaRecorder(mediaStream);
    mediaRecorder.addEventListener('dataavailable', (event) => {
      if (event.data.size > 0) {
        audioChunks.push(event.data);
      }
    });
    mediaRecorder.addEventListener('stop', () => {
      finalizeRecording();
    });
    mediaRecorder.start();

    setupLevelMeter(mediaStream);
    startSpeechRecognition();

    elements.startButton.disabled = true;
    elements.stopButton.disabled = false;
    elements.analyzeButton.disabled = true;
    elements.analysisStatus.textContent = 'Recording in progress...';
    elements.meterCaption.textContent = 'Listening for energy and pacing cues';
    elements.playback.hidden = true;

    recordingTimer = window.setInterval(() => {
      secondsRemaining -= 1;
      updateTimer(secondsRemaining);
      if (secondsRemaining <= 0) {
        stopRecording();
      }
    }, 1000);
  } catch (error) {
    elements.analysisStatus.textContent = 'Microphone access is required to record a note.';
    console.error(error);
  }
}

function stopRecording() {
  if (!mediaRecorder || mediaRecorder.state === 'inactive') {
    return;
  }

  mediaRecorder.stop();
  stopSpeechRecognition();
  stopLevelMeter();
  clearInterval(recordingTimer);
  recordingTimer = null;
  elements.startButton.disabled = false;
  elements.stopButton.disabled = true;
  elements.meterCaption.textContent = 'Recording complete';
}

async function finalizeRecording(blobOverride, options = {}) {
  const { skipTranscription = false } = options;
  audioBlob = blobOverride || new Blob(audioChunks, { type: mediaRecorder?.mimeType || 'audio/webm' });
  audioFileName = blobOverride?.name || buildRecordingFilename(audioBlob);

  if (playbackUrl) {
    URL.revokeObjectURL(playbackUrl);
  }

  playbackUrl = URL.createObjectURL(audioBlob);
  elements.playback.src = playbackUrl;
  elements.playback.hidden = false;
  elements.analyzeButton.disabled = false;
  elements.transcribeButton.disabled = skipTranscription || !serverCapabilities.openAiConfigured;
  elements.analysisStatus.textContent = skipTranscription
    ? 'Audio ready. Add a transcript or summary to continue.'
    : serverCapabilities.openAiConfigured
      ? 'Audio ready. Preparing transcript...'
      : 'Audio ready. Add a transcript or summary to continue.';

  if (finalTranscript.trim() && !elements.transcriptInput.value.trim()) {
    elements.transcriptInput.value = finalTranscript.trim();
  }

  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }

  mediaRecorder = null;

  if (!skipTranscription && serverCapabilities.openAiConfigured) {
    await transcribeRecording();
  }
}

async function analyzeEntry() {
  const transcriptDraft = elements.transcriptInput.value.trim();

  // Talk mode: no audio blob, but transcript loaded from conversation
  if (!audioBlob && transcriptDraft) {
    await analyzeTextOnly(transcriptDraft);
    return;
  }

  if (!audioBlob) {
    elements.analysisStatus.textContent = 'Record or upload audio before analyzing.';
    return;
  }

  if (!serverCapabilities.openAiConfigured && !transcriptDraft) {
    elements.analysisStatus.textContent = 'Add a transcript or summary to continue.';
    return;
  }

  elements.analysisStatus.textContent = serverCapabilities.openAiConfigured
    ? 'Uploading audio for GPT transcription and analysis...'
    : 'Analyzing vocal cues with keyword matching...';
  elements.analyzeButton.disabled = true;

  try {
    let recordingAnalysis;
    if (serverCapabilities.openAiConfigured) {
      recordingAnalysis = await analyzeRecording(audioBlob, transcriptDraft, analysisMode);
    } else {
      const { analysis, emotionDetection } = await analyzeTranscript(transcriptDraft, analysisMode);
      recordingAnalysis = { transcript: transcriptDraft, textAnalysis: analysis, emotionDetection };
    }
    const transcript = normalizeTranscript(recordingAnalysis.transcript || transcriptDraft);

    if (!transcript) {
      throw new Error('Transcript unavailable for analysis.');
    }

    finalTranscript = transcript;
    hasRecordedTranscription = hasRecordedTranscription || Boolean(recordingAnalysis.transcript);
    elements.transcriptInput.value = transcript;

    const vocalMetrics = await extractAudioMetrics(audioBlob, transcript);
    const textAnalysis = recordingAnalysis.textAnalysis;
    const emotionDetection = recordingAnalysis.emotionDetection || null;
    const combined = combineSignals(vocalMetrics, textAnalysis);

    renderResult(combined, vocalMetrics, textAnalysis);
    renderEmotionDetection(emotionDetection);
    storeHistory(combined, vocalMetrics, textAnalysis, transcript, emotionDetection);
    renderHistory();
    elements.analysisStatus.textContent = 'Analysis complete.';
  } catch (error) {
    console.error(error);
    elements.analysisStatus.textContent = `Analysis failed: ${error.message || 'Check your server and API configuration.'}`;
  } finally {
    elements.analyzeButton.disabled = false;
  }
}

async function analyzeTextOnly(transcript) {
  elements.analysisStatus.textContent = 'Analyzing conversation transcript...';
  elements.analyzeButton.disabled = true;

  try {
    const { analysis: textAnalysis, emotionDetection } = await analyzeTranscript(transcript, analysisMode);
    const vocalMetrics = conversationVocalMetrics(transcript);
    const combined = combineSignals(vocalMetrics, textAnalysis);

    renderResult(combined, vocalMetrics, textAnalysis);
    renderEmotionDetection(emotionDetection);
    storeHistory(combined, vocalMetrics, textAnalysis, transcript, null);
    renderHistory();
    elements.analysisStatus.textContent = 'Analysis complete.';
  } catch (error) {
    console.error(error);
    elements.analysisStatus.textContent = `Analysis failed: ${error.message}`;
  } finally {
    elements.analyzeButton.disabled = false;
  }
}

function conversationVocalMetrics(transcript) {
  const words = transcript.trim().split(/\s+/).filter(Boolean).length;
  return {
    vocalScore: 0,
    wordsPerMinute: words,
    energyVariance: 'n/a',
    silenceRatio: 'n/a',
    vocalLabel: 'not measured',
    pacingLabel: 'not measured'
  };
}

async function analyzeTranscript(transcript, mode = ADVANCED_ANALYSIS) {
  const historyStr = JSON.stringify(readHistory().slice(0, 7));
  const response = await fetch('/api/analyze-text', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ transcript, mode, history: historyStr })
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.error || 'Transcript analysis failed.');
  }

  const payload = await response.json();
  return { analysis: normalizeAnalysisPayload(payload), emotionDetection: payload.emotionDetection || null };
}

async function analyzeRecording(blob, transcriptDraft, mode = ADVANCED_ANALYSIS) {
  const formData = new FormData();
  formData.append('audio', blob, audioFileName || buildRecordingFilename(blob));
  if (transcriptDraft) {
    formData.append('transcript', transcriptDraft);
  }
  formData.append('mode', mode);
  formData.append('history', JSON.stringify(readHistory().slice(0, 7)));

  const response = await fetch('/api/analyze-recording', {
    method: 'POST',
    body: formData
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.details || payload.error || 'Recording analysis failed.');
  }

  const payload = await response.json();
  return {
    ...payload,
    textAnalysis: normalizeAnalysisPayload(payload.textAnalysis || {})
  };
}

async function extractAudioMetrics(blob, transcript) {
  const buffer = await blob.arrayBuffer();
  const offlineContext = new AudioContext();

  try {
    const audioBuffer = await offlineContext.decodeAudioData(buffer);
    const channel = audioBuffer.getChannelData(0);
    const windowSize = 2048;
    const rmsValues = [];

    for (let index = 0; index < channel.length; index += windowSize) {
      let sum = 0;
      const sliceEnd = Math.min(index + windowSize, channel.length);
      for (let cursor = index; cursor < sliceEnd; cursor += 1) {
        sum += channel[cursor] * channel[cursor];
      }

      const rms = Math.sqrt(sum / (sliceEnd - index || 1));
      rmsValues.push(rms);
    }

    const averageEnergy = average(rmsValues);
    const energyVariance = standardDeviation(rmsValues, averageEnergy);
    const silenceRatio = rmsValues.filter((value) => value < 0.015).length / (rmsValues.length || 1);
    const wordsPerMinute = Math.round((countWords(transcript) / Math.max(audioBuffer.duration, 1)) * 60);
    const vocalScore = Number(
      (
        energyVariance * 8 +
        Math.max(0, 0.28 - silenceRatio) * 2 +
        paceDeviation(wordsPerMinute)
      ).toFixed(2)
    );

    return {
      durationSeconds: Number(audioBuffer.duration.toFixed(1)),
      averageEnergy: Number(averageEnergy.toFixed(3)),
      energyVariance: Number(energyVariance.toFixed(3)),
      silenceRatio: Number(silenceRatio.toFixed(2)),
      wordsPerMinute,
      vocalScore,
      pacingLabel: describePacing(wordsPerMinute),
      vocalLabel: describeVocalState(energyVariance, silenceRatio)
    };
  } finally {
    try {
      await offlineContext.close();
    } catch {
      // Ignore close errors caused by browser timing differences.
    }
  }
}

function combineSignals(vocalMetrics, textAnalysis) {
  const concernWeight = {
    low: 0,
    moderate: 1.3,
    high: 2.4
  }[textAnalysis.concernLevel] || 0;

  const textWeight = textAnalysis.sentimentScore < -0.4 ? 1.5 : textAnalysis.sentimentScore < -0.15 ? 0.8 : 0;
  const isVocalMeasured = vocalMetrics.vocalLabel !== 'not measured';
  const vocalWeight = isVocalMeasured ? (vocalMetrics.vocalScore > 2.3 ? 1.4 : vocalMetrics.vocalScore > 1.2 ? 0.8 : 0) : 0;
  const pacingWeight = isVocalMeasured ? (vocalMetrics.wordsPerMinute > 175 || vocalMetrics.wordsPerMinute < 85 ? 0.8 : 0) : 0;
  const combinedScore = Number((concernWeight + textWeight + vocalWeight + pacingWeight).toFixed(2));

  let level = 'stable';
  let headline = 'Your signals look balanced today.';

  if (combinedScore >= 3.3) {
    level = 'high';
    headline = 'It sounds like today is a really tough day for you.';
  } else if (combinedScore >= 1.6) {
    level = 'elevated';
    headline = 'It sounds like you\'re carrying a bit of weight today.';
  } else {
    headline = 'It sounds like you\'re doing okay today.';
  }

  const isVocalMeasuredSummary = vocalMetrics.vocalLabel !== 'not measured';
  const summary = [
    isVocalMeasuredSummary
      ? `Your voice sounded ${vocalMetrics.vocalLabel} with a ${vocalMetrics.pacingLabel} speaking pace.`
      : 'I listened to what you wrote instead of your voice today.',
    `Underneath it all, I'm hearing some ${textAnalysis.emotionalState} feelings.`
  ].join(' ');

  return {
    level,
    headline,
    summary,
    combinedScore
  };
}

function renderResult(combined, vocalMetrics, textAnalysis) {
  const normalizedAnalysis = normalizeAnalysisPayload(textAnalysis);
  elements.resultEmpty.hidden = true;
  elements.resultCard.hidden = false;
  elements.resultCard.dataset.level = combined.level;
  elements.resultBanner.textContent = combined.level === 'high'
    ? 'Needs some TLC'
    : combined.level === 'elevated'
      ? 'Checking in'
      : 'Doing alright';
  elements.resultHeadline.textContent = combined.headline;
  elements.resultSummary.textContent = combined.summary;
  elements.feedbackNarrative.textContent = normalizedAnalysis.feedback;
  renderTips(normalizedAnalysis.wellnessTips, vocalMetrics);
  elements.rationaleText.textContent = normalizedAnalysis.rationale;
  // Reset deep section to collapsed on each new analysis
  elements.deepSummarySection.hidden = true;
  elements.deepSummaryToggle.textContent = 'See deeper analysis ▾';
  elements.deepSummaryToggle.classList.remove('open');
}

function storeHistory(combined, vocalMetrics, textAnalysis, transcript, emotionDetection) {
  const entries = readHistory();
  const today = new Date().toISOString();
  const dominant = emotionDetection?.dominant?.label || null;
  const dk = todayDateKey();
  const nextEntries = [
    {
      id: today,
      dateKey: dk,
      date: formatDate(today),
      level: combined.level,
      headline: combined.headline,
      summary: combined.summary,
      emotionalState: textAnalysis.emotionalState,
      wordsPerMinute: vocalMetrics.wordsPerMinute,
      transcriptPreview: transcript.slice(0, 120),
      mlEmotion: dominant
    },
    ...entries.filter((e) => entryDateKey(e) !== dk)
  ].slice(0, 7);

  localStorage.setItem(HISTORY_KEY, JSON.stringify(nextEntries));
}

function logRestDay() {
  const entries = readHistory();
  const dk = todayDateKey();
  if (entries.some((e) => entryDateKey(e) === dk)) {
    return;
  }

  const today = new Date().toISOString();
  const nextEntries = [
    {
      id: today,
      dateKey: dk,
      date: formatDate(today),
      level: 'rest',
      isRest: true,
      headline: 'Rest day logged.',
      summary: 'You chose to rest today. Consistency matters more than perfection.',
      emotionalState: 'resting',
      wordsPerMinute: 0,
      transcriptPreview: ''
    },
    ...entries
  ].slice(0, 7);

  localStorage.setItem(HISTORY_KEY, JSON.stringify(nextEntries));
  renderHistory();
}

function resetTrendline() {
  if (!window.confirm('Clear all check-in history? This cannot be undone.')) {
    return;
  }
  localStorage.removeItem(HISTORY_KEY);
  renderHistory();
}

// ── Talk mode ─────────────────────────────────────────────────────────────────

async function enterTalkMode() {
  conversationHistory = [];
  pendingSessionComplete = false;
  elements.conversationList.innerHTML = '';
  if (elements.talkLiveTranscript) elements.talkLiveTranscript.textContent = '';
  elements.talkStatus.textContent = 'Starting your session…';
  elements.talkMicButton.disabled = true;
  elements.talkMicButton.textContent = 'Starting…';
  elements.talkMicButton.classList.remove('listening');
  elements.endTalkButton.disabled = false;
  updateTalkPhase(0);

  await sendUserGreeting();
}

function leaveTalkMode() {
  if (talkAbortController) {
    talkAbortController.abort();
    talkAbortController = null;
  }
  if (talkRecognition) {
    try { talkRecognition.abort(); } catch { /* ignore */ }
    talkRecognition = null;
  }
  if (talkAudio) {
    talkAudio.pause();
    talkAudio = null;
  }
  window.speechSynthesis.cancel();
  isBotSpeaking = false;
  isTalkListening = false;
  pendingSessionComplete = false;
}

function startUserListening() {
  if (isBotSpeaking || isTalkListening || captureMode !== TALK_MODE || pendingSessionComplete) {
    return;
  }

  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    elements.talkStatus.textContent = 'Speech recognition unavailable in this browser.';
    elements.talkMicButton.disabled = true;
    return;
  }

  talkRecognition = new SpeechRecognition();
  talkRecognition.lang = 'en-US';
  talkRecognition.continuous = true;     // don't stop on first pause
  talkRecognition.interimResults = true; // live partial results
  talkRecognition.maxAlternatives = 1;

  isTalkListening = true;
  elements.talkMicButton.classList.add('listening');
  elements.talkMicButton.textContent = 'Listening…';
  elements.talkMicButton.disabled = false;
  elements.talkStatus.textContent = 'Go ahead, I\'m listening…';
  if (elements.talkLiveTranscript) elements.talkLiveTranscript.textContent = '';

  let finalText = '';
  let silenceTimer = null;
  let hasSpoken = false;

  // Two-tier silence detection:
  //   — before speech begins: 8 s safety net (browser 'no-speech' usually fires first)
  //   — after speech detected: 2.5 s pause = user is done with their thought
  const INITIAL_WAIT    = 8000;
  const SPEECH_PAUSE    = 2500;
  // Give longer pause for longer utterances (the person is clearly sharing a lot)
  function pauseDelay() {
    const words = finalText.trim().split(/\s+/).length;
    return words > 20 ? 3200 : SPEECH_PAUSE;
  }

  function resetSilenceTimer() {
    clearTimeout(silenceTimer);
    silenceTimer = setTimeout(() => {
      if (talkRecognition) talkRecognition.stop();
    }, hasSpoken ? pauseDelay() : INITIAL_WAIT);
  }

  talkRecognition.onresult = (event) => {
    let interimText = '';
    finalText = '';

    for (let i = 0; i < event.results.length; i++) {
      const t = event.results[i][0].transcript;
      if (event.results[i].isFinal) {
        finalText += t + ' ';
      } else {
        interimText += t;
      }
    }

    finalText = finalText.trim();
    const displayText = (finalText + (interimText ? ' ' + interimText : '')).trim();

    if (displayText) {
      hasSpoken = true;
      // Show live words in the dedicated transcript preview
      if (elements.talkLiveTranscript) elements.talkLiveTranscript.textContent = displayText;
      elements.talkStatus.textContent = '';
    }

    // Reset the post-speech silence timer on every new word
    resetSilenceTimer();
  };

  talkRecognition.onend = () => {
    clearTimeout(silenceTimer);
    isTalkListening = false;
    elements.talkMicButton.classList.remove('listening');
    if (elements.talkLiveTranscript) elements.talkLiveTranscript.textContent = '';

    const text = finalText.trim();
    if (text) {
      elements.talkMicButton.textContent = 'Processing…';
      elements.talkMicButton.disabled = true;
      elements.talkStatus.textContent = '';
      sendUserTurn(text);
    } else if (captureMode === TALK_MODE && !isBotSpeaking && !pendingSessionComplete) {
      elements.talkStatus.textContent = 'Take your time — I\'m still here.';
      setTimeout(() => {
        if (captureMode === TALK_MODE && !isBotSpeaking && !pendingSessionComplete) {
          startUserListening();
        }
      }, 1200);
    }
  };

  talkRecognition.onerror = (event) => {
    clearTimeout(silenceTimer);
    isTalkListening = false;
    elements.talkMicButton.classList.remove('listening');
    if (elements.talkLiveTranscript) elements.talkLiveTranscript.textContent = '';

    if (event.error === 'not-allowed') {
      elements.talkStatus.textContent = 'Microphone access denied. Check browser permissions.';
      elements.talkMicButton.textContent = 'Mic blocked';
      elements.talkMicButton.disabled = true;
    } else if (event.error === 'no-speech') {
      elements.talkStatus.textContent = 'Take your time — I\'m still here.';
      setTimeout(() => {
        if (captureMode === TALK_MODE && !isBotSpeaking && !pendingSessionComplete) startUserListening();
      }, 1000);
    } else if (captureMode === TALK_MODE && !isBotSpeaking && !pendingSessionComplete) {
      setTimeout(() => startUserListening(), 1000);
    }
  };

  talkRecognition.start();
  // Safety net: if no speech arrives within INITIAL_WAIT, stop and retry
  resetSilenceTimer();
}

// Greeting — no user bubble, just stream the bot's opening line
async function sendUserGreeting() {
  updateTalkPhase(0);
  const botBubble = appendTypingBubble();
  let fullReply = '';
  let streamDone = false;
  let isPlayingAudio = false;
  let localSessionComplete = false;
  const audioQueue = [];

  function onAllAudioDone() {
    isBotSpeaking = false;
    talkAudio = null;
    if (localSessionComplete) {
      completeSession();
    } else if (captureMode === TALK_MODE) {
      elements.talkMicButton.textContent = 'Listening…';
      startUserListening();
    }
  }

  function playNext() {
    if (captureMode !== TALK_MODE) return;
    if (isPlayingAudio || audioQueue.length === 0) return;
    isPlayingAudio = true;
    const audio = audioQueue.shift();
    audio.onended = () => {
      isPlayingAudio = false;
      if (audioQueue.length > 0) {
        playNext();
      } else if (streamDone) {
        onAllAudioDone();
      }
    };
    audio.onerror = () => {
      isPlayingAudio = false;
      if (audioQueue.length > 0) {
        playNext();
      } else if (streamDone) {
        onAllAudioDone();
      }
    };
    talkAudio = audio;
    audio.play().catch(() => { isPlayingAudio = false; isBotSpeaking = false; });
  }

  talkAbortController = new AbortController();

  try {
    const res = await fetch('/api/chat-stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: [], turnCount: 0 }),
      signal: talkAbortController.signal
    });
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let sseBuffer = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      sseBuffer += decoder.decode(value, { stream: true });
      const lines = sseBuffer.split('\n');
      sseBuffer = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        let data; try { data = JSON.parse(line.slice(6)); } catch { continue; }
        if (data.sentence) {
          fullReply += (fullReply ? ' ' : '') + data.sentence;
          botBubble.classList.remove('typing');
          botBubble.textContent = fullReply;
          elements.conversationList.scrollTop = elements.conversationList.scrollHeight;
          const url = '/api/tts?text=' + encodeURIComponent(data.sentence);
          const audio = new Audio(url);
          audioQueue.push(audio);
          if (!isPlayingAudio) {
            isBotSpeaking = true;
            elements.talkMicButton.textContent = 'Speaking…';
            elements.talkStatus.textContent = '';
            playNext();
          }
        }
        if (data.done) {
          streamDone = true;
          localSessionComplete = Boolean(data.session_complete);
          if (localSessionComplete) pendingSessionComplete = true;
          conversationHistory.push({ role: 'assistant', content: fullReply });
          if (!isPlayingAudio && audioQueue.length === 0) onAllAudioDone();
        }
      }
    }
  } catch {
    botBubble.classList.remove('typing');
    botBubble.textContent = 'Hi, I\'m really glad you\'re here. How are you feeling today?';
    isBotSpeaking = false;
    if (captureMode === TALK_MODE) startUserListening();
  }
}

async function sendUserTurn(text) {
  appendChatBubble('user', text);
  conversationHistory.push({ role: 'user', content: text });

  // Turn count = number of user messages so far (we just pushed one)
  const turnCount = conversationHistory.filter((m) => m.role === 'user').length;
  updateTalkPhase(turnCount);

  elements.talkMicButton.disabled = true;
  elements.talkMicButton.classList.remove('listening');
  elements.talkMicButton.textContent = 'Thinking…';
  elements.talkStatus.textContent = '';
  if (elements.talkLiveTranscript) elements.talkLiveTranscript.textContent = '';

  const botBubble = appendTypingBubble();
  let fullReply = '';
  const audioQueue = [];
  let isPlayingAudio = false;
  let streamDone = false;
  let localSessionComplete = false;

  function onAllAudioDone() {
    isBotSpeaking = false;
    talkAudio = null;
    if (localSessionComplete) {
      completeSession();
    } else if (captureMode === TALK_MODE) {
      elements.talkMicButton.textContent = 'Listening…';
      startUserListening();
    }
  }

  function playNext() {
    if (captureMode !== TALK_MODE) return;
    if (isPlayingAudio || audioQueue.length === 0) return;
    isPlayingAudio = true;
    const audio = audioQueue.shift();

    audio.onended = () => {
      isPlayingAudio = false;
      if (audioQueue.length > 0) {
        playNext();
      } else if (streamDone) {
        onAllAudioDone();
      }
    };

    audio.onerror = () => {
      isPlayingAudio = false;
      if (audioQueue.length > 0) {
        playNext();
      } else if (streamDone) {
        onAllAudioDone();
      }
    };

    talkAudio = audio;
    audio.play().catch(() => { isPlayingAudio = false; isBotSpeaking = false; });
  }

  function queueSentence(sentence) {
    const url = '/api/tts?text=' + encodeURIComponent(sentence);
    const audio = new Audio(url);
    audioQueue.push(audio);
    if (!isPlayingAudio) {
      isBotSpeaking = true;
      elements.talkMicButton.textContent = 'Speaking…';
      elements.talkStatus.textContent = '';
      playNext();
    }
  }

  talkAbortController = new AbortController();

  try {
    const res = await fetch('/api/chat-stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: conversationHistory, turnCount }),
      signal: talkAbortController.signal
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let sseBuffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      sseBuffer += decoder.decode(value, { stream: true });
      const lines = sseBuffer.split('\n');
      sseBuffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        let data;
        try { data = JSON.parse(line.slice(6)); } catch { continue; }

        if (data.sentence) {
          fullReply += (fullReply ? ' ' : '') + data.sentence;
          botBubble.classList.remove('typing');
          botBubble.textContent = fullReply;
          elements.conversationList.scrollTop = elements.conversationList.scrollHeight;
          queueSentence(data.sentence);
        }
        if (data.done) {
          streamDone = true;
          localSessionComplete = Boolean(data.session_complete);
          if (localSessionComplete) pendingSessionComplete = true;
          conversationHistory.push({ role: 'assistant', content: fullReply });
          if (!isPlayingAudio && audioQueue.length === 0) onAllAudioDone();
        }
      }
    }
  } catch {
    botBubble.classList.remove('typing');
    botBubble.textContent = 'I\'m right here with you. Take your time.';
    streamDone = true;
    isBotSpeaking = false;
    if (captureMode === TALK_MODE && !pendingSessionComplete) startUserListening();
  }
}

async function speakBotReply(text, onDone) {
  // Stop anything currently playing
  if (talkAudio) {
    talkAudio.pause();
    talkAudio = null;
  }
  window.speechSynthesis.cancel();

  isBotSpeaking = true;
  elements.talkMicButton.disabled = true;
  elements.talkStatus.textContent = 'Speaking...';

  // Prefer OpenAI TTS (nova voice) — sounds human; fall back to browser TTS
  if (serverCapabilities.hasOpenAi) {
    try {
      // Use GET + audio.src so the browser streams and plays as chunks arrive
      const url = '/api/tts?text=' + encodeURIComponent(text);
      const audio = new Audio(url);
      talkAudio = audio;

      audio.onended = () => {
        talkAudio = null;
        isBotSpeaking = false;
        onDone?.();
      };

      audio.onerror = () => {
        talkAudio = null;
        isBotSpeaking = false;
        speakWithBrowserTTS(text, onDone);
      };

      await audio.play();
      return;
    } catch {
      // Fall through to browser TTS
    }
  }

  speakWithBrowserTTS(text, onDone);
}

function speakWithBrowserTTS(text, onDone) {
  if (!window.speechSynthesis) {
    isBotSpeaking = false;
    onDone?.();
    return;
  }

  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 0.9;
  utterance.pitch = 1.05;

  utterance.onend = () => {
    isBotSpeaking = false;
    onDone?.();
  };

  utterance.onerror = () => {
    isBotSpeaking = false;
    onDone?.();
  };

  window.speechSynthesis.speak(utterance);
}

function appendChatBubble(role, text) {
  const bubble = document.createElement('div');
  bubble.className = 'chat-bubble';
  bubble.dataset.role = role;
  bubble.textContent = text;
  elements.conversationList.appendChild(bubble);
  elements.conversationList.scrollTop = elements.conversationList.scrollHeight;
  return bubble;
}

function appendTypingBubble() {
  const bubble = document.createElement('div');
  bubble.className = 'chat-bubble typing';
  bubble.dataset.role = 'bot';
  bubble.innerHTML = '<span class="dot"></span><span class="dot"></span><span class="dot"></span>';
  elements.conversationList.appendChild(bubble);
  elements.conversationList.scrollTop = elements.conversationList.scrollHeight;
  return bubble;
}

function updateTalkPhase(turnCount) {
  if (!elements.talkPhasePill) return;

  let phase, label;
  if (turnCount <= 1) {
    phase = 'rapport'; label = 'Getting to know you';
  } else if (turnCount <= 4) {
    phase = 'exploring'; label = 'Exploring';
  } else if (turnCount <= 7) {
    phase = 'working'; label = 'Working through';
  } else {
    phase = 'wrapping'; label = 'Wrapping up';
  }

  elements.talkPhasePill.textContent = label;
  elements.talkPhasePill.dataset.phase = phase;

  if (elements.talkTurnCount) {
    elements.talkTurnCount.textContent = turnCount > 0 ? `Turn ${turnCount}` : '';
  }
}

function completeSession() {
  isBotSpeaking = false;
  talkAudio = null;
  elements.talkMicButton.disabled = true;
  elements.talkMicButton.textContent = 'Session complete';
  elements.talkMicButton.classList.remove('listening');

  if (elements.talkPhasePill) {
    elements.talkPhasePill.textContent = 'Complete';
    elements.talkPhasePill.dataset.phase = 'wrapping';
  }

  // Append a visual session-complete marker inside the conversation
  const pill = document.createElement('div');
  pill.className = 'talk-session-complete';
  pill.textContent = 'Session complete';
  elements.conversationList.appendChild(pill);
  elements.conversationList.scrollTop = elements.conversationList.scrollHeight;

  elements.talkStatus.textContent = 'Generating your emotional analysis…';
  elements.endTalkButton.disabled = false;

  // Auto-transition to analysis after a short pause
  setTimeout(() => {
    if (captureMode === TALK_MODE) endTalkSession();
  }, 2800);
}

function endTalkSession() {
  // Do NOT call leaveTalkMode() here — setCaptureMode handles it below
  if (conversationHistory.length === 0) {
    setCaptureMode(RECORDING_MODE);
    return;
  }

  const transcript = conversationHistory
    .map((m) => (m.role === 'user' ? 'Me: ' : 'Companion: ') + m.content)
    .join('\n\n');

  // Clear any prior audio so analyzeEntry routes to text-only path,
  // not to the old recording from a previous Record/Upload session
  audioBlob = null;
  if (playbackUrl) {
    URL.revokeObjectURL(playbackUrl);
    playbackUrl = null;
  }
  elements.playback.hidden = true;
  elements.playback.src = '';

  elements.transcriptInput.value = transcript;
  finalTranscript = transcript;
  setTranscriptStatus('Conversation transcript loaded. Run analysis below.');
  elements.analyzeButton.disabled = false;
  elements.analysisStatus.textContent = 'Conversation ready to analyze.';

  setCaptureMode(RECORDING_MODE); // calls leaveTalkMode internally

  document.querySelector('.transcript-panel')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── End talk mode ──────────────────────────────────────────────────────────────

function todayDateKey() {
  return new Date().toLocaleDateString('en-CA');
}

function entryDateKey(entry) {
  return entry.dateKey || (entry.id ? entry.id.slice(0, 10) : null);
}

function computeStreak(entries) {
  if (!entries.length) {
    return 0;
  }

  const keys = new Set(entries.map(entryDateKey).filter(Boolean));
  let streak = 0;
  const cursor = new Date();

  for (let i = 0; i < 365; i++) {
    const key = cursor.toLocaleDateString('en-CA');
    if (keys.has(key)) {
      streak++;
    } else if (i === 0) {
      // today has no entry yet — streak from yesterday still alive
    } else {
      break;
    }
    cursor.setDate(cursor.getDate() - 1);
  }

  return streak;
}

function renderStreak(entries) {
  const streak = computeStreak(entries);
  elements.streakCounter.hidden = streak === 0;
  elements.streakNumber.textContent = String(streak);

  const hasEntryToday = entries.some((e) => entryDateKey(e) === todayDateKey());
  elements.restDayButton.disabled = hasEntryToday;
  elements.restDayButton.textContent = hasEntryToday ? 'Rested today' : 'Log rest day';
}

function renderHistory() {
  const entries = readHistory();
  elements.historyList.innerHTML = '';
  renderSparkline(entries);
  renderStreak(entries);

  if (entries.length < 2) {
    elements.historyNudge.hidden = false;
    elements.historyNudge.textContent = 'Complete another check-in tomorrow to start spotting a trend.';
  } else {
    elements.historyNudge.hidden = true;
    elements.historyNudge.textContent = '';
  }

  if (!entries.length) {
    const empty = document.createElement('div');
    empty.className = 'empty-history';
    empty.textContent = 'No check-ins yet. Your first analysis will populate a 7-day trendline.';
    elements.historyList.append(empty);
    return;
  }

  entries.forEach((entry) => {
    const card = document.createElement('article');
    card.className = 'history-item';

    const header = document.createElement('header');
    const title = document.createElement('h3');
    title.textContent = String(entry.date || 'Recent check-in');
    const tag = document.createElement('span');
    tag.className = 'history-tag';
    tag.dataset.level = String(entry.level || 'stable');
    tag.textContent = entry.isRest ? 'rest day' : String(entry.level || 'stable');
    header.append(title, tag);

    const headline = document.createElement('p');
    headline.textContent = String(entry.headline || 'No headline available.');

    const transcript = document.createElement('p');
    transcript.textContent = String(entry.transcriptPreview || '(no transcript)');

    const metrics = document.createElement('p');
    metrics.textContent = `${Number(entry.wordsPerMinute || 0)} wpm · ${String(entry.emotionalState || 'steady')}${entry.mlEmotion ? ` · ML: ${entry.mlEmotion}` : ''}`;

    card.append(header, headline, transcript, metrics);
    elements.historyList.append(card);
  });
}

function readHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
  } catch {
    return [];
  }
}

function setupLevelMeter(stream) {
  audioContext = new AudioContext();
  analyser = audioContext.createAnalyser();
  analyser.fftSize = 256;
  sourceNode = audioContext.createMediaStreamSource(stream);
  sourceNode.connect(analyser);

  const data = new Uint8Array(analyser.frequencyBinCount);

  const update = () => {
    analyser.getByteFrequencyData(data);
    const averageLevel = data.reduce((sum, value) => sum + value, 0) / (data.length || 1);
    const width = Math.min(100, Math.round((averageLevel / 160) * 100));
    elements.meterFill.style.width = `${width}%`;
    meterLoop = requestAnimationFrame(update);
  };

  update();
}

function stopLevelMeter() {
  if (meterLoop) {
    cancelAnimationFrame(meterLoop);
    meterLoop = null;
  }

  elements.meterFill.style.width = '0%';

  if (sourceNode) {
    sourceNode.disconnect();
    sourceNode = null;
  }

  if (audioContext) {
    audioContext.close().catch(() => undefined);
    audioContext = null;
  }
}

function startSpeechRecognition() {
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!Recognition) {
    elements.speechWarning.hidden = false;
    elements.speechWarning.textContent = SPEECH_WARNING_MESSAGE;
    if (serverCapabilities.openAiConfigured) {
      setTranscriptStatus('Live browser transcript is unavailable here. The saved recording will be uploaded for GPT transcription after capture.');
    }
    return;
  }

  recognition = new Recognition();
  recognition.lang = 'en-US';
  recognition.continuous = true;
  recognition.interimResults = true;

  recognition.onresult = (event) => {
    const transcript = Array.from(event.results)
      .map((result) => result[0].transcript)
      .join(' ')
      .trim();

    finalTranscript = transcript;
    elements.transcriptInput.value = transcript;
    setTranscriptStatus('Live transcript updated. The uploaded recording will be used as the final analysis source.');
  };

  recognition.onerror = () => {
    elements.speechWarning.hidden = false;
    elements.speechWarning.textContent = 'Speech recognition ran into an issue. You can continue by typing the transcript manually.';
    if (serverCapabilities.openAiConfigured) {
      setTranscriptStatus('Live transcript capture had an issue. The saved recording will still be uploaded for GPT transcription.');
    }
  };

  recognition.start();
}

function stopSpeechRecognition() {
  if (recognition) {
    recognition.stop();
    recognition = null;
  }
}

function updateTimer(remaining = secondsRemaining) {
  secondsRemaining = remaining;
  const minutes = String(Math.floor(secondsRemaining / 60)).padStart(2, '0');
  const seconds = String(secondsRemaining % 60).padStart(2, '0');
  elements.timerPill.textContent = `${minutes}:${seconds}`;
}

function getSupportedMimeType() {
  const candidates = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4'];
  return candidates.find((candidate) => MediaRecorder.isTypeSupported(candidate));
}

function renderSparkline(entries) {
  const svg = elements.historySparkline;
  svg.replaceChildren();

  if (!entries.length) {
    elements.sparklineWrap.hidden = true;
    return;
  }

  elements.sparklineWrap.hidden = false;
  const width = 320;
  const height = 80;
  const padding = 10;
  const maxIndex = Math.max(entries.length - 1, 1);

  const points = entries
    .slice()
    .reverse()
    .map((entry, index) => {
      const x = padding + ((width - padding * 2) * index) / maxIndex;
      const y = levelToY(entry.level, height, padding);
      return {
        x,
        y,
        level: entry.level || 'stable'
      };
    });

  const path = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
  path.setAttribute('points', points.map((point) => `${point.x},${point.y}`).join(' '));
  path.setAttribute('fill', 'none');
  path.setAttribute('stroke', 'rgba(23, 36, 47, 0.45)');
  path.setAttribute('stroke-width', '2.5');
  path.setAttribute('stroke-linecap', 'round');
  path.setAttribute('stroke-linejoin', 'round');
  svg.append(path);

  points.forEach((point) => {
    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    circle.setAttribute('cx', String(point.x));
    circle.setAttribute('cy', String(point.y));
    circle.setAttribute('r', '4.5');
    circle.setAttribute('fill', levelColor(point.level));
    circle.setAttribute('stroke', 'rgba(255, 255, 255, 0.9)');
    circle.setAttribute('stroke-width', '1.5');
    svg.append(circle);
  });
}

function levelToY(level, height, padding) {
  const yByLevel = {
    stable: height - padding,
    elevated: height / 2,
    high: padding,
    rest: height - padding
  };

  return yByLevel[level] || yByLevel.stable;
}

function levelColor(level) {
  const colorByLevel = {
    stable: '#5f8b7e',
    elevated: '#d97a20',
    high: '#ab3f45',
    rest: '#b5aca0'
  };

  return colorByLevel[level] || colorByLevel.stable;
}

function average(values) {
  return values.reduce((sum, value) => sum + value, 0) / (values.length || 1);
}

function standardDeviation(values, mean) {
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (values.length || 1);
  return Math.sqrt(variance);
}

function countWords(text) {
  return text.trim().split(/\s+/).filter(Boolean).length;
}

function paceDeviation(wordsPerMinute) {
  if (wordsPerMinute < 85) {
    return 0.9;
  }
  if (wordsPerMinute > 175) {
    return 0.8;
  }
  return 0.2;
}

function describePacing(wordsPerMinute) {
  if (wordsPerMinute < 85) {
    return 'slower';
  }
  if (wordsPerMinute > 175) {
    return 'faster';
  }
  return 'steady';
}

function describeVocalState(energyVariance, silenceRatio) {
  if (energyVariance > 0.15 && silenceRatio < 0.18) {
    return 'strained';
  }
  if (silenceRatio > 0.45) {
    return 'hesitant';
  }
  return 'grounded';
}

function formatDate(isoDate) {
  return new Intl.DateTimeFormat(undefined, {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit'
  }).format(new Date(isoDate));
}

async function transcribeRecording({ forceRefresh = false } = {}) {
  if (!audioBlob || isTranscribing) {
    return;
  }

  if (!serverCapabilities.openAiConfigured) {
    setTranscriptStatus('OPENAI_API_KEY is not configured. Type a short summary manually or add the key to enable audio transcription.');
    return;
  }

  if (!forceRefresh && hasRecordedTranscription) {
    setTranscriptStatus('Transcript captured. Edit it if needed before analysis.');
    elements.analysisStatus.textContent = 'Ready to analyze.';
    return;
  }

  isTranscribing = true;
  elements.transcribeButton.disabled = true;
  setTranscriptStatus('Uploading audio for GPT transcription...');

  try {
    const transcript = await requestRecordingTranscription(audioBlob);
    if (!transcript) {
      throw new Error('No transcript returned from the API.');
    }

    finalTranscript = transcript;
    hasRecordedTranscription = true;
    elements.transcriptInput.value = transcript;
    setTranscriptStatus('Saved recording transcribed successfully through GPT audio.');
    elements.analysisStatus.textContent = 'Ready to analyze.';
  } catch (error) {
    console.error(error);
    if (forceRefresh) {
      hasRecordedTranscription = false;
    }
    if (elements.transcriptInput.value.trim()) {
      setTranscriptStatus('Automatic transcription had an issue. You can keep editing the live transcript draft.');
      elements.analysisStatus.textContent = 'Ready to analyze.';
    } else {
      setTranscriptStatus('Automatic transcription failed. Type a short summary manually, or try transcribing again.');
      elements.analysisStatus.textContent = 'Transcript required before analysis.';
    }
  } finally {
    isTranscribing = false;
    elements.transcribeButton.disabled = !audioBlob || !serverCapabilities.openAiConfigured;
  }
}

async function requestRecordingTranscription(blob) {
  const formData = new FormData();
  formData.append('audio', blob, audioFileName || buildRecordingFilename(blob));

  const response = await fetch('/api/transcribe-recording', {
    method: 'POST',
    body: formData
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.error || 'Recording transcription failed.');
  }

  const payload = await response.json();
  return normalizeTranscript(payload.transcript || '');
}

function normalizeTranscript(text) {
  return text.replace(/\s+/g, ' ').trim();
}

function normalizeAnalysisPayload(textAnalysis) {
  const fallbackTips = [
    'Pause for one slow minute of breathing.',
    'Shrink the next task to one manageable step.',
    'Check in again later if the strain increases.'
  ];

  const normalizedTips = Array.isArray(textAnalysis?.wellnessTips)
    ? textAnalysis.wellnessTips.map((tip) => String(tip || '').trim()).filter(Boolean).slice(0, 3)
    : [];

  return {
    emotionalState: String(textAnalysis?.emotionalState || 'steady'),
    concernLevel: String(textAnalysis?.concernLevel || 'low'),
    sentimentScore: Number.isFinite(Number(textAnalysis?.sentimentScore)) ? Number(textAnalysis.sentimentScore) : 0,
    rationale: String(textAnalysis?.rationale || 'The transcript was analyzed for emotional cues.'),
    feedback: String(textAnalysis?.feedback || textAnalysis?.supportiveMessage || 'Take one supportive step and check back in later today.'),
    wellnessTips: normalizedTips.length ? normalizedTips : fallbackTips,
    supportiveMessage: String(textAnalysis?.supportiveMessage || 'Check in again tomorrow to keep building a trend line.')
  };
}

function renderTips(tips, vocalMetrics) {
  const mergedTips = [...normalizeTipList(tips), buildVocalTip(vocalMetrics)].filter(Boolean).slice(0, 4);
  elements.tipsList.innerHTML = '';

  mergedTips.forEach((tip) => {
    const item = document.createElement('li');
    item.textContent = tip;
    elements.tipsList.append(item);
  });
}

function normalizeTipList(tips) {
  if (!Array.isArray(tips)) {
    return [];
  }

  return tips.map((tip) => String(tip || '').trim()).filter(Boolean);
}

function buildVocalTip(vocalMetrics) {
  if (vocalMetrics.vocalLabel === 'not measured') {
    return null;
  }
  if (vocalMetrics.wordsPerMinute > 175) {
    return 'Your pace ran fast. Try one slower breath before your next task.';
  }
  if (vocalMetrics.wordsPerMinute < 85) {
    return 'Your pace was slower today. Keep the next task very small and concrete.';
  }
  if (vocalMetrics.silenceRatio > 0.45) {
    return 'There were longer pauses in the recording. Give yourself a short reset before pushing ahead.';
  }

  return 'Your vocal pace was fairly steady. Protect one short recovery break later today.';
}

function setTranscriptStatus(message) {
  elements.transcriptStatus.textContent = message;
}

function buildRecordingFilename(blob) {
  const extension = blob.type.includes('mp4') ? 'm4a' : 'webm';
  return `voice-note.${extension}`;
}

function renderEmotionDetection(emotionDetection) {
  if (!elements.emotionCard) {
    return;
  }

  if (!emotionDetection || !emotionDetection.emotions) {
    elements.emotionCard.hidden = true;
    return;
  }

  elements.emotionCard.hidden = false;
  elements.emotionModelLabel.textContent = emotionDetection.model
    ? `Model: ${emotionDetection.model}`
    : 'ML emotion classifier';

  elements.emotionBars.innerHTML = '';

  const sortedEmotions = Object.entries(emotionDetection.emotions)
    .sort((a, b) => b[1] - a[1]);

  for (const [emotion, score] of sortedEmotions) {
    const row = document.createElement('div');
    row.className = 'emotion-bar-row';

    const label = document.createElement('span');
    label.className = 'emotion-bar-label';
    label.textContent = emotion;

    const track = document.createElement('div');
    track.className = 'emotion-bar-track';

    const fill = document.createElement('div');
    fill.className = 'emotion-bar-fill';
    fill.dataset.emotion = emotion;
    fill.style.width = '0%';
    track.append(fill);

    const value = document.createElement('span');
    value.className = 'emotion-bar-value';
    value.textContent = `${(score * 100).toFixed(1)}%`;

    row.append(label, track, value);
    elements.emotionBars.append(row);

    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        fill.style.width = `${(score * 100).toFixed(1)}%`;
      });
    });
  }

  if (emotionDetection.dominant) {
    elements.emotionDominant.innerHTML =
      `Dominant emotion: <strong>${emotionDetection.dominant.label}</strong> (${(emotionDetection.dominant.score * 100).toFixed(1)}% confidence)`;
  } else {
    elements.emotionDominant.textContent = '';
  }
}
