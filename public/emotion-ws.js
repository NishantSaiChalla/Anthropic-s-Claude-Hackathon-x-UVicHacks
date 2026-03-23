// ── Multimodal Emotion Engine — WebSocket Client ──────────────────────────────
// Self-contained module. Connects to FastAPI backend at /ws, streams audio+video,
// and updates the live emotion panel in the DOM.

const EMOTION_WS_URL = (() => {
  const meta = document.querySelector('meta[name="emotion-ws-url"]');
  return meta?.content || 'ws://localhost:8001/ws';
})();
const AUDIO_CHUNK_MS = 2000;      // 2-second chunks matching backend window
const VIDEO_FPS = 5;              // 5 face frames per second
const RECONNECT_DELAY_MS = 3000;
const JPEG_QUALITY = 0.6;

let emotionSocket = null;
let emotionAudioStream = null;
let emotionVideoStream = null;
let audioScriptNode = null;
let audioCtx = null;
let audioSourceNode = null;
let pcmBuffer = [];
let audioChunkTimer = null;
let videoCanvas = null;
let videoCanvasCtx = null;
let videoElement = null;
let videoFrameTimer = null;
let shouldReconnect = true;

// ── Public API ────────────────────────────────────────────────────────────────

export function startEmotionEngine() {
  shouldReconnect = true;
  connectWebSocket();
}

export function stopEmotionEngine() {
  shouldReconnect = false;
  teardownAudio();
  teardownVideo();
  if (emotionSocket) {
    emotionSocket.close();
    emotionSocket = null;
  }
}

export function isEmotionEngineConnected() {
  return emotionSocket && emotionSocket.readyState === WebSocket.OPEN;
}

// ── WebSocket Lifecycle ───────────────────────────────────────────────────────

function connectWebSocket() {
  if (emotionSocket && emotionSocket.readyState <= WebSocket.OPEN) return;

  try {
    emotionSocket = new WebSocket(EMOTION_WS_URL);
  } catch {
    scheduleReconnect();
    return;
  }

  emotionSocket.onopen = () => {
    updateConnectionBadge(true);
    startAudioCapture();
    startVideoCapture();
  };

  emotionSocket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      renderEmotionState(data);
    } catch { /* ignore malformed messages */ }
  };

  emotionSocket.onclose = () => {
    updateConnectionBadge(false);
    teardownAudio();
    teardownVideo();
    scheduleReconnect();
  };

  emotionSocket.onerror = () => {
    // onclose will fire right after
  };
}

function scheduleReconnect() {
  if (!shouldReconnect) return;
  setTimeout(() => {
    if (shouldReconnect) connectWebSocket();
  }, RECONNECT_DELAY_MS);
}

// ── Audio Capture → PCM16 Base64 ─────────────────────────────────────────────

async function startAudioCapture() {
  try {
    emotionAudioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch {
    return; // mic denied — continue with video only
  }

  audioCtx = new AudioContext({ sampleRate: 16000 });
  audioSourceNode = audioCtx.createMediaStreamSource(emotionAudioStream);

  // ScriptProcessor gives us raw PCM samples (deprecated but universally supported)
  audioScriptNode = audioCtx.createScriptProcessor(4096, 1, 1);
  pcmBuffer = [];

  audioScriptNode.onaudioprocess = (e) => {
    const samples = e.inputBuffer.getChannelData(0);
    // Convert float32 [-1,1] → int16 [-32768,32767]
    const int16 = new Int16Array(samples.length);
    for (let i = 0; i < samples.length; i++) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    pcmBuffer.push(int16);
  };

  audioSourceNode.connect(audioScriptNode);
  audioScriptNode.connect(audioCtx.destination); // required for processing to fire

  // Flush accumulated PCM every AUDIO_CHUNK_MS
  audioChunkTimer = setInterval(() => {
    if (!pcmBuffer.length || !isEmotionEngineConnected()) return;

    const totalLen = pcmBuffer.reduce((n, a) => n + a.length, 0);
    const merged = new Int16Array(totalLen);
    let offset = 0;
    for (const chunk of pcmBuffer) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }
    pcmBuffer = [];

    const b64 = arrayBufferToBase64(merged.buffer);
    emotionSocket.send(JSON.stringify({
      type: 'audio',
      payload_base64: b64,
      timestamp_ms: Date.now()
    }));
  }, AUDIO_CHUNK_MS);
}

function teardownAudio() {
  clearInterval(audioChunkTimer);
  audioChunkTimer = null;
  pcmBuffer = [];
  if (audioScriptNode) { audioScriptNode.disconnect(); audioScriptNode = null; }
  if (audioSourceNode) { audioSourceNode.disconnect(); audioSourceNode = null; }
  if (audioCtx) { audioCtx.close().catch(() => {}); audioCtx = null; }
  if (emotionAudioStream) {
    emotionAudioStream.getTracks().forEach(t => t.stop());
    emotionAudioStream = null;
  }
}

// ── Video Capture → JPEG + Quality Hints ─────────────────────────────────────

async function startVideoCapture() {
  try {
    emotionVideoStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 320, height: 240, facingMode: 'user' }
    });
  } catch {
    return; // camera denied — continue with audio only
  }

  videoElement = document.createElement('video');
  videoElement.srcObject = emotionVideoStream;
  videoElement.muted = true;
  videoElement.playsInline = true;
  videoElement.play();

  videoCanvas = document.createElement('canvas');
  videoCanvas.width = 320;
  videoCanvas.height = 240;
  videoCanvasCtx = videoCanvas.getContext('2d');

  // Show preview in the live panel if element exists
  const preview = document.getElementById('emotionCameraPreview');
  if (preview) {
    preview.srcObject = emotionVideoStream;
    preview.style.display = 'block';
    preview.play().catch(() => {});
  }

  videoFrameTimer = setInterval(() => {
    if (!isEmotionEngineConnected() || !videoElement.videoWidth) return;

    videoCanvasCtx.drawImage(videoElement, 0, 0, 320, 240);

    // Compute blur heuristic (variance of Laplacian approximation)
    const gray = videoCanvasCtx.getImageData(0, 0, 320, 240);
    const blur = estimateBlur(gray);

    videoCanvas.toBlob((blob) => {
      if (!blob) return;
      const reader = new FileReader();
      reader.onloadend = () => {
        if (!isEmotionEngineConnected()) return;
        const b64 = reader.result.split(',')[1];
        emotionSocket.send(JSON.stringify({
          type: 'video',
          payload_b64_jpg: b64,
          timestamp_ms: Date.now(),
          quality_hints: { yaw: 0, pitch: 0, blur }
        }));
      };
      reader.readAsDataURL(blob);
    }, 'image/jpeg', JPEG_QUALITY);
  }, 1000 / VIDEO_FPS);
}

function teardownVideo() {
  clearInterval(videoFrameTimer);
  videoFrameTimer = null;
  if (emotionVideoStream) {
    emotionVideoStream.getTracks().forEach(t => t.stop());
    emotionVideoStream = null;
  }
  videoElement = null;
  videoCanvas = null;
  videoCanvasCtx = null;

  const preview = document.getElementById('emotionCameraPreview');
  if (preview) {
    preview.srcObject = null;
    preview.style.display = 'none';
  }
}

function estimateBlur(imageData) {
  // Simple Laplacian variance on grayscale
  const { data, width, height } = imageData;
  let sum = 0, sumSq = 0, count = 0;
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const idx = (y * width + x) * 4;
      const c = data[idx]; // red channel as proxy for gray
      const l = data[idx - 4];
      const r = data[idx + 4];
      const t = data[(idx - width * 4)];
      const b = data[(idx + width * 4)];
      const laplacian = -4 * c + l + r + t + b;
      sum += laplacian;
      sumSq += laplacian * laplacian;
      count++;
    }
  }
  const mean = sum / count;
  return (sumSq / count) - (mean * mean); // variance
}

// ── DOM Rendering ─────────────────────────────────────────────────────────────

function renderEmotionState(data) {
  const el = (id) => document.getElementById(id);

  const badge = el('emotionStateBadge');
  if (badge) {
    const cat = data.fused_emotion?.mapped_categorical || '—';
    badge.textContent = cat;
    badge.dataset.state = data.system_state || 'UNKNOWN';
  }

  const valEl = el('emotionValence');
  if (valEl) valEl.textContent = (data.fused_emotion?.valence ?? 0).toFixed(2);

  const aroEl = el('emotionArousal');
  if (aroEl) aroEl.textContent = (data.fused_emotion?.arousal ?? 0).toFixed(2);

  const sysEl = el('emotionSystemState');
  if (sysEl) sysEl.textContent = `${data.system_state || '—'} · ${data.signal_status || '—'}`;

  const wEl = el('emotionWeights');
  if (wEl) {
    const wa = (data.modality_weights?.audio ?? 0).toFixed(2);
    const wv = (data.modality_weights?.video ?? 0).toFixed(2);
    wEl.textContent = `Weights - A: ${wa} V: ${wv}`;
    wEl.style.color = wa > wv ? '#60a5fa' : (wv > wa ? '#f472b6' : 'inherit');
  }

  const cEl = el('emotionConf');
  if (cEl) {
    const conf = data.fused_emotion?.label_confidence ?? 0;
    cEl.textContent = `Confidence: ${(conf * 100).toFixed(0)}%`;
  }

  const dAudio = el('debugAudio');
  if (dAudio && data.debug) {
    dAudio.textContent = `Audio: ${data.debug.raw_audio_valence.toFixed(2)} / ${data.debug.raw_audio_arousal.toFixed(2)}`;
  }
  const dVideo = el('debugVideo');
  if (dVideo && data.debug) {
    const cls = data.debug.raw_video_class;
    const cNames = ['Neu', 'Hap', 'Sur', 'Sad', 'Ang', 'Dis', 'Fea', 'Con'];
    const name = cNames[cls] || '-';
    dVideo.textContent = `Video: ${data.debug.raw_video_valence.toFixed(2)} / ${data.debug.raw_video_arousal.toFixed(2)} (${name})`;
  }
  const dFused = el('debugFused');
  if (dFused && data.debug) {
    dFused.textContent = `Fused: ${data.debug.fused_valence.toFixed(2)} / ${data.debug.fused_arousal.toFixed(2)} (Conf: ${(data.debug.label_confidence * 100).toFixed(0)}%)`;
  }

  const qEl = el('emotionQuality');
  if (qEl) {
    const flags = data.quality_flags || {};
    const parts = [];
    if (flags.audio_vad_active) parts.push('🎙️ voice');
    if (flags.video_face_detected) parts.push('📷 face');
    if (flags.video_blur_warning) parts.push('⚠️ blur');
    if (flags.video_pose_warning) parts.push('⚠️ pose');
    if (data.is_stale) parts.push('⏳ stale');
    qEl.textContent = parts.join('  ') || '—';
  }
}

function updateConnectionBadge(connected) {
  const dot = document.getElementById('emotionConnDot');
  if (dot) {
    dot.dataset.connected = String(connected);
    dot.title = connected ? 'Emotion engine connected' : 'Emotion engine disconnected';
  }
}

// ── Utilities ─────────────────────────────────────────────────────────────────

function arrayBufferToBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  let binary = '';
  for (let i = 0; i < bytes.byteLength; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}
