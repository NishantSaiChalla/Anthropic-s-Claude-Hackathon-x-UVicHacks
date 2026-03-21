import { env, pipeline } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';

env.allowLocalModels = false;
env.useBrowserCache = true;

const RECORDING_LIMIT_SECONDS = 30;
const HISTORY_KEY = 'ellipsis-health-history';

const elements = {
  startButton: document.querySelector('#startButton'),
  stopButton: document.querySelector('#stopButton'),
  transcribeButton: document.querySelector('#transcribeButton'),
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
  historyList: document.querySelector('#historyList')
};

let mediaRecorder = null;
let mediaStream = null;
let audioChunks = [];
let audioBlob = null;
let recordingTimer = null;
let meterLoop = null;
let secondsRemaining = RECORDING_LIMIT_SECONDS;
let recognition = null;
let finalTranscript = '';
let audioContext = null;
let analyser = null;
let sourceNode = null;
let transcriberPromise = null;
let isTranscribing = false;
let hasRecordedTranscription = false;

renderHistory();
wireEvents();

function wireEvents() {
  elements.startButton.addEventListener('click', startRecording);
  elements.stopButton.addEventListener('click', stopRecording);
  elements.transcribeButton.addEventListener('click', () => {
    transcribeRecording({ forceRefresh: true });
  });
  elements.analyzeButton.addEventListener('click', analyzeEntry);
  elements.clearTranscriptButton.addEventListener('click', () => {
    finalTranscript = '';
    elements.transcriptInput.value = '';
    setTranscriptStatus('Transcript cleared. Re-transcribe or type a summary.');
  });
}

async function startRecording() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioChunks = [];
    audioBlob = null;
    finalTranscript = '';
    hasRecordedTranscription = false;
    secondsRemaining = RECORDING_LIMIT_SECONDS;
    updateTimer();
    elements.transcriptInput.value = '';
    elements.transcribeButton.disabled = true;
    setTranscriptStatus('Recording started. Transcript will generate after you stop.');

    mediaRecorder = new MediaRecorder(mediaStream, { mimeType: getSupportedMimeType() });
    mediaRecorder.addEventListener('dataavailable', (event) => {
      if (event.data.size > 0) {
        audioChunks.push(event.data);
      }
    });
    mediaRecorder.addEventListener('stop', finalizeRecording);
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
      updateTimer();
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

async function finalizeRecording() {
  audioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
  elements.playback.src = URL.createObjectURL(audioBlob);
  elements.playback.hidden = false;
  elements.analyzeButton.disabled = false;
  elements.transcribeButton.disabled = false;
  elements.analysisStatus.textContent = 'Audio ready. Preparing transcript...';

  if (finalTranscript.trim() && !elements.transcriptInput.value.trim()) {
    elements.transcriptInput.value = finalTranscript.trim();
  }

  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }

  await transcribeRecording();
}

async function analyzeEntry() {
  if (!audioBlob) {
    elements.analysisStatus.textContent = 'Record a note before analyzing.';
    return;
  }

  const transcript = elements.transcriptInput.value.trim();
  if (!transcript) {
    elements.analysisStatus.textContent = 'Add a transcript or summary to continue.';
    return;
  }

  elements.analysisStatus.textContent = 'Analyzing vocal cues and transcript sentiment...';
  elements.analyzeButton.disabled = true;

  try {
    const vocalMetrics = await extractAudioMetrics(audioBlob, transcript);
    const textAnalysis = await analyzeTranscript(transcript);
    const combined = combineSignals(vocalMetrics, textAnalysis);

    renderResult(combined, vocalMetrics, textAnalysis);
    storeHistory(combined, vocalMetrics, textAnalysis, transcript);
    renderHistory();
    elements.analysisStatus.textContent = 'Analysis complete.';
  } catch (error) {
    console.error(error);
    elements.analysisStatus.textContent = 'Analysis failed. Check your server or Claude key.';
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
    offlineContext.close();
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
    card.innerHTML = `
      <header>
        <h3>${entry.date}</h3>
        <span class="history-tag">${entry.level}</span>
      </header>
      <p>${entry.headline}</p>
      <p>${entry.transcriptPreview}</p>
      <p>${entry.wordsPerMinute} wpm · ${entry.emotionalState}</p>
    `;
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
    audioContext.close();
    audioContext = null;
  }
}

function startSpeechRecognition() {
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!Recognition) {
    setTranscriptStatus('Live browser transcript is unavailable here. The saved recording will still be transcribed after capture.');
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
    setTranscriptStatus('Live transcript updated. Final recording transcription will refine it after capture.');
  };

  recognition.onerror = () => {
    setTranscriptStatus('Live transcript capture had an issue. The saved recording will still be transcribed after capture.');
  };

  recognition.start();
}

function stopSpeechRecognition() {
  if (recognition) {
    recognition.stop();
    recognition = null;
  }
}

function updateTimer() {
  const minutes = String(Math.floor(secondsRemaining / 60)).padStart(2, '0');
  const seconds = String(secondsRemaining % 60).padStart(2, '0');
  elements.timerPill.textContent = `${minutes}:${seconds}`;
}

function getSupportedMimeType() {
  const candidates = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4'];
  return candidates.find((candidate) => MediaRecorder.isTypeSupported(candidate));
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

  if (!forceRefresh && hasRecordedTranscription) {
    setTranscriptStatus('Transcript captured. Edit it if needed before analysis.');
    elements.analysisStatus.textContent = 'Ready to analyze.';
    return;
  }

  isTranscribing = true;
  elements.transcribeButton.disabled = true;
  setTranscriptStatus('Transcribing saved recording...');

  try {
    const audio = await decodeAudioForTranscription(audioBlob);
    const transcriber = await getTranscriber();
    const result = await transcriber(audio, {
      chunk_length_s: 30,
      stride_length_s: 5,
      return_timestamps: false,
      language: 'english',
      task: 'transcribe'
    });

    const transcript = normalizeTranscript(result?.text || '');
    if (!transcript) {
      throw new Error('No transcript returned from speech model.');
    }

    finalTranscript = transcript;
    hasRecordedTranscription = true;
    elements.transcriptInput.value = transcript;
    setTranscriptStatus('Saved recording transcribed successfully.');
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
    elements.transcribeButton.disabled = !audioBlob;
  }
}

async function getTranscriber() {
  if (!transcriberPromise) {
    setTranscriptStatus('Loading speech model for first-time transcription...');
    transcriberPromise = pipeline('automatic-speech-recognition', 'Xenova/whisper-tiny.en').catch((error) => {
      transcriberPromise = null;
      throw error;
    });
  }

  return transcriberPromise;
}

async function decodeAudioForTranscription(blob) {
  const arrayBuffer = await blob.arrayBuffer();
  const decodeContext = new AudioContext();

  try {
    const decoded = await decodeContext.decodeAudioData(arrayBuffer);
    const mono = downmixToMono(decoded);
    return resampleAudio(mono, decoded.sampleRate, 16000);
  } finally {
    await decodeContext.close();
  }
}

function downmixToMono(audioBuffer) {
  if (audioBuffer.numberOfChannels === 1) {
    return audioBuffer.getChannelData(0);
  }

  const mono = new Float32Array(audioBuffer.length);
  for (let channelIndex = 0; channelIndex < audioBuffer.numberOfChannels; channelIndex += 1) {
    const channelData = audioBuffer.getChannelData(channelIndex);
    for (let sampleIndex = 0; sampleIndex < audioBuffer.length; sampleIndex += 1) {
      mono[sampleIndex] += channelData[sampleIndex] / audioBuffer.numberOfChannels;
    }
  }

  return mono;
}

function resampleAudio(input, inputSampleRate, targetSampleRate) {
  if (inputSampleRate === targetSampleRate) {
    return input;
  }

  const ratio = inputSampleRate / targetSampleRate;
  const newLength = Math.round(input.length / ratio);
  const output = new Float32Array(newLength);

  for (let index = 0; index < newLength; index += 1) {
    const start = Math.floor(index * ratio);
    const end = Math.min(input.length, Math.floor((index + 1) * ratio));
    let sum = 0;
    let count = 0;

    for (let sampleIndex = start; sampleIndex < end; sampleIndex += 1) {
      sum += input[sampleIndex];
      count += 1;
    }

    output[index] = count ? sum / count : input[start] || 0;
  }

  return output;
}

function normalizeTranscript(text) {
  return text.replace(/\s+/g, ' ').trim();
}

function setTranscriptStatus(message) {
  elements.transcriptStatus.textContent = message;
}
