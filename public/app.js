const RECORDING_LIMIT_SECONDS = 30;
const HISTORY_KEY = 'ellipsis-health-history';
const SPEECH_WARNING_MESSAGE = 'Browser speech recognition is unavailable. Recordings still work, but type your transcript manually.';

const RECORDING_MODE = 'record';
const UPLOAD_MODE = 'upload';

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
  vocalScore: document.querySelector('#vocalScore'),
  textScore: document.querySelector('#textScore'),
  paceScore: document.querySelector('#paceScore'),
  vocalNarrative: document.querySelector('#vocalNarrative'),
  textNarrative: document.querySelector('#textNarrative'),
  feedbackNarrative: document.querySelector('#feedbackNarrative'),
  tipsList: document.querySelector('#tipsList'),
  historyList: document.querySelector('#historyList'),
  sparklineWrap: document.querySelector('#sparklineWrap'),
  historySparkline: document.querySelector('#historySparkline'),
  historyNudge: document.querySelector('#historyNudge')
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
let serverCapabilities = {
  openAiConfigured: false,
  anthropicConfigured: false,
  analysisModel: 'local-heuristic',
  transcriptionModel: null
};

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
  elements.chooseFileButton.addEventListener('click', () => elements.audioUploadInput.click());
  elements.audioUploadInput.addEventListener('change', (event) => {
    const file = event.target.files?.[0];
    if (file) {
      handleUploadedFile(file);
    }
  });
  elements.analyzeButton.addEventListener('click', analyzeEntry);
  elements.clearTranscriptButton.addEventListener('click', () => {
    finalTranscript = '';
    elements.transcriptInput.value = '';
    setTranscriptStatus('Transcript cleared. Re-transcribe or type a summary.');
  });

  setupDropZone();
  refreshCapabilityWarnings();
}

function setCaptureMode(mode) {
  if (captureMode === mode) {
    return;
  }

  captureMode = mode;
  const usingRecordMode = captureMode === RECORDING_MODE;

  elements.modeRecordButton.classList.toggle('active', usingRecordMode);
  elements.modeRecordButton.setAttribute('aria-selected', String(usingRecordMode));
  elements.modeUploadButton.classList.toggle('active', !usingRecordMode);
  elements.modeUploadButton.setAttribute('aria-selected', String(!usingRecordMode));

  elements.recordingPanel.hidden = !usingRecordMode;
  elements.recordingButtons.hidden = !usingRecordMode;
  elements.uploadPanel.hidden = usingRecordMode;

  if (!usingRecordMode) {
    stopRecording();
    updateTimer(RECORDING_LIMIT_SECONDS);
    elements.meterCaption.textContent = 'Upload mode ready';
  } else if (!audioBlob) {
    elements.meterCaption.textContent = 'Microphone idle';
  }

  refreshCapabilityWarnings();
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
  const looksAudio = file.type.startsWith('audio/') || /\.(wav|mp3|m4a|ogg|webm|aac|flac)$/i.test(file.name);
  if (!looksAudio) {
    elements.analysisStatus.textContent = 'Please choose a valid audio file.';
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
  if (!audioBlob) {
    elements.analysisStatus.textContent = 'Record or upload audio before analyzing.';
    return;
  }

  const transcriptDraft = elements.transcriptInput.value.trim();
  if (!serverCapabilities.openAiConfigured && !transcriptDraft) {
    elements.analysisStatus.textContent = 'Add a transcript or summary to continue.';
    return;
  }

  elements.analysisStatus.textContent = serverCapabilities.openAiConfigured
    ? 'Uploading audio for GPT transcription and analysis...'
    : 'Analyzing vocal cues and transcript sentiment...';
  elements.analyzeButton.disabled = true;

  try {
    const recordingAnalysis = serverCapabilities.openAiConfigured
      ? await analyzeRecording(audioBlob, transcriptDraft)
      : { transcript: transcriptDraft, textAnalysis: await analyzeTranscript(transcriptDraft) };
    const transcript = normalizeTranscript(recordingAnalysis.transcript || transcriptDraft);

    if (!transcript) {
      throw new Error('Transcript unavailable for analysis.');
    }

    finalTranscript = transcript;
    hasRecordedTranscription = hasRecordedTranscription || Boolean(recordingAnalysis.transcript);
    elements.transcriptInput.value = transcript;

    const vocalMetrics = await extractAudioMetrics(audioBlob, transcript);
    const textAnalysis = recordingAnalysis.textAnalysis;
    const combined = combineSignals(vocalMetrics, textAnalysis);

    renderResult(combined, vocalMetrics, textAnalysis);
    storeHistory(combined, vocalMetrics, textAnalysis, transcript);
    renderHistory();
    elements.analysisStatus.textContent = 'Analysis complete.';
  } catch (error) {
    console.error(error);
    elements.analysisStatus.textContent = 'Analysis failed. Check your server and API configuration.';
  } finally {
    elements.analyzeButton.disabled = false;
  }
}

async function analyzeTranscript(transcript) {
  const response = await fetch('/api/analyze-text', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ transcript })
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.error || 'Transcript analysis failed.');
  }

  return response.json();
}

async function analyzeRecording(blob, transcriptDraft) {
  const formData = new FormData();
  formData.append('audio', blob, audioFileName || buildRecordingFilename(blob));
  if (transcriptDraft) {
    formData.append('transcript', transcriptDraft);
  }

  const response = await fetch('/api/analyze-recording', {
    method: 'POST',
    body: formData
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.error || 'Recording analysis failed.');
  }

  return response.json();
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
  const vocalWeight = vocalMetrics.vocalScore > 2.3 ? 1.4 : vocalMetrics.vocalScore > 1.2 ? 0.8 : 0;
  const pacingWeight = vocalMetrics.wordsPerMinute > 175 || vocalMetrics.wordsPerMinute < 85 ? 0.8 : 0;
  const combinedScore = Number((concernWeight + textWeight + vocalWeight + pacingWeight).toFixed(2));

  let level = 'stable';
  let headline = 'Your signals look balanced today.';

  if (combinedScore >= 3.3) {
    level = 'high';
    headline = 'Today shows multiple elevated emotional cues.';
  } else if (combinedScore >= 1.6) {
    level = 'elevated';
    headline = 'There are some emotional signals worth tracking.';
  }

  const summary = [
    `Vocal delivery sounded ${vocalMetrics.vocalLabel} with ${vocalMetrics.pacingLabel} pacing.`,
    `Transcript analysis suggested ${textAnalysis.emotionalState} cues at a ${textAnalysis.concernLevel} concern level.`
  ].join(' ');

  return {
    level,
    headline,
    summary,
    combinedScore
  };
}

function renderResult(combined, vocalMetrics, textAnalysis) {
  elements.resultEmpty.hidden = true;
  elements.resultCard.hidden = false;
  elements.resultCard.dataset.level = combined.level;
  elements.resultBanner.textContent = combined.level === 'high'
    ? 'High attention'
    : combined.level === 'elevated'
      ? 'Elevated'
      : 'Stable';
  elements.resultHeadline.textContent = combined.headline;
  elements.resultSummary.textContent = combined.summary;
  elements.vocalScore.textContent = vocalMetrics.vocalScore.toFixed(2);
  elements.textScore.textContent = textAnalysis.sentimentScore.toFixed(2);
  elements.paceScore.textContent = `${vocalMetrics.wordsPerMinute} wpm`;
  elements.vocalNarrative.textContent = [
    `Energy variance: ${vocalMetrics.energyVariance}.`,
    `Silence ratio: ${vocalMetrics.silenceRatio}.`,
    `Interpretation: ${vocalMetrics.vocalLabel}.`
  ].join(' ');
  elements.textNarrative.textContent = `${textAnalysis.rationale} ${textAnalysis.supportiveMessage}`;
  elements.feedbackNarrative.textContent = textAnalysis.feedback || 'Take one supportive step and check back in later today.';
  renderTips(textAnalysis.wellnessTips, vocalMetrics);
}

function storeHistory(combined, vocalMetrics, textAnalysis, transcript) {
  const entries = readHistory();
  const today = new Date().toISOString();
  const nextEntries = [
    {
      id: today,
      date: formatDate(today),
      level: combined.level,
      headline: combined.headline,
      summary: combined.summary,
      emotionalState: textAnalysis.emotionalState,
      wordsPerMinute: vocalMetrics.wordsPerMinute,
      transcriptPreview: transcript.slice(0, 120)
    },
    ...entries
  ].slice(0, 7);

  localStorage.setItem(HISTORY_KEY, JSON.stringify(nextEntries));
}

function renderHistory() {
  const entries = readHistory();
  elements.historyList.innerHTML = '';
  renderSparkline(entries);

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
    tag.textContent = String(entry.level || 'stable');
    header.append(title, tag);

    const headline = document.createElement('p');
    headline.textContent = String(entry.headline || 'No headline available.');

    const transcript = document.createElement('p');
    transcript.textContent = String(entry.transcriptPreview || '(no transcript)');

    const metrics = document.createElement('p');
    metrics.textContent = `${Number(entry.wordsPerMinute || 0)} wpm · ${String(entry.emotionalState || 'steady')}`;

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
    high: padding
  };

  return yByLevel[level] || yByLevel.stable;
}

function levelColor(level) {
  const colorByLevel = {
    stable: '#5f8b7e',
    elevated: '#d97a20',
    high: '#ab3f45'
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
