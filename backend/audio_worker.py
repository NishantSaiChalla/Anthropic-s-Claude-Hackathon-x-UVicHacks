import torch
import torch.nn as nn
import numpy as np
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model,
)


# ── Audeering model-card architecture (verbatim) ─────────────────────


class RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class EmotionModel(Wav2Vec2PreTrainedModel):
    _tied_weights_keys = []
    _keys_to_ignore_on_load_missing = [r"classifier"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)
        return hidden_states, logits


# ── Worker ────────────────────────────────────────────────────────────


class AudioWorker:
    def __init__(self):
        model_name = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = EmotionModel.from_pretrained(model_name)
        self.model.eval()

        self.silero, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=True,
        )

    def infer(self, req_data: dict) -> dict:
        waveform = req_data["array"]  # float32 ndarray, 16 kHz mono
        duration_sec = len(waveform) / 16000.0

        # Minimum receptive-field guard
        if duration_sec < 0.5:
            return {
                "val": 0.0,
                "aro": 0.0,
                "conf": 0.0,
                "vad_prob": 0.0,
                "valid": False,
            }

        with torch.no_grad():
            wav_tensor = torch.from_numpy(waveform)
            # Silero VAD v4 expects specific frame sizes (512 for 16kHz)
            # We iterate in chunks and take the maximum probability
            window_size = 512
            vad_probs = []
            for i in range(0, len(wav_tensor), window_size):
                chunk = wav_tensor[i : i + window_size]
                if len(chunk) < window_size:
                    break
                prob = self.silero(chunk, 16000).item()
                vad_probs.append(prob)
            
            vad_prob = max(vad_probs) if vad_probs else 0.0

            if vad_prob < 0.4:  # Slightly lower threshold for voice detection
                return {
                    "val": 0.0,
                    "aro": 0.0,
                    "conf": 0.0,
                    "vad_prob": vad_prob,
                    "valid": True,
                }

            # Exact model-card preprocessing flow
            inputs = self.processor(waveform, sampling_rate=16000)
            x = inputs["input_values"][0]
            x = x.reshape(1, -1)
            x_tensor = torch.from_numpy(x)

            _, logits = self.model(x_tensor)
            preds = logits[0].detach().cpu().numpy()

            # Output order per model card: Arousal[0], Dominance[1], Valence[2]
            # Native range: ~[0.0, 1.0]
            arousal_raw = float(preds[0])
            _dominance_raw = float(preds[1])
            valence_raw = float(preds[2])

        # Keep short but expressive speech from being discounted too aggressively.
        dur_penalty = min(1.0, max(0.6, duration_sec / 2.0))
        base_conf = max(0.35, vad_prob)
        confidence = float(min(1.0, base_conf * dur_penalty))

        # Rescale [0,1] -> [-1,1] to align with visual model V/A space
        return {
            "val": (valence_raw - 0.5) * 2.0,
            "aro": (arousal_raw - 0.5) * 2.0,
            "conf": confidence,
            "vad_prob": float(vad_prob),
            "valid": True,
        }
