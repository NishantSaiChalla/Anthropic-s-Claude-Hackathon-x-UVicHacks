import os
import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification


FER_VA_MAP = {
    0: (0.00, 0.00),   # neutral
    1: (0.75, 0.45),   # happiness
    2: (0.45, 0.75),   # surprise
    3: (-0.75, -0.45), # sadness
    4: (-0.80, 0.80),  # anger
    5: (-0.70, 0.25),  # disgust
    6: (-0.65, 0.90),  # fear
    7: (-0.45, 0.10),  # contempt
}

CANONICAL_LABELS = [
    "neutral",
    "happy",
    "surprise",
    "sad",
    "anger",
    "disgust",
    "fear",
    "contempt",
]

LABEL_TO_INDEX = {label: idx for idx, label in enumerate(CANONICAL_LABELS)}
LABEL_TO_VA = {label: FER_VA_MAP[idx] for idx, label in enumerate(CANONICAL_LABELS)}
LABEL_ALIASES = {
    "neutral": "neutral",
    "happy": "happy",
    "happiness": "happy",
    "surprise": "surprise",
    "surprised": "surprise",
    "sad": "sad",
    "sadness": "sad",
    "anger": "anger",
    "angry": "anger",
    "disgust": "disgust",
    "disgusted": "disgust",
    "fear": "fear",
    "fearful": "fear",
    "contempt": "contempt",
}


class VideoWorker:
    def __init__(self, model_path: str = "emotion-ferplus-8.onnx"):
        self.preferred_backend = os.getenv("VISION_MODEL_BACKEND", "hybrid").strip().lower()
        self.transformer_model_name = os.getenv(
            "VISION_MODEL_NAME",
            "HardlyHumans/Facial-expression-detection",
        ).strip()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.transformer_processor = None
        self.transformer_model = None
        self.fer_session = None
        self.fer_input_name = None
        self.fer_model_name = None

        resolved_model_path = model_path
        if not os.path.isabs(resolved_model_path):
            resolved_model_path = os.path.join(os.path.dirname(__file__), resolved_model_path)

        if self.preferred_backend in {"hybrid", "transformers"}:
            self._try_load_transformer()

        if self.preferred_backend in {"hybrid", "onnx"} or self.transformer_model is None:
            self._try_load_ferplus(resolved_model_path)

        if self.transformer_model is None and self.fer_session is None:
            raise RuntimeError(
                "No visual emotion model could be initialized. "
                "Check internet access for the transformer model or restore the FER+ ONNX file."
            )

    def _try_load_transformer(self):
        try:
            self.transformer_processor = AutoImageProcessor.from_pretrained(self.transformer_model_name)
            self.transformer_model = AutoModelForImageClassification.from_pretrained(self.transformer_model_name)
            self.transformer_model.eval()
            self.transformer_model.to(self.device)
        except Exception as exc:
            print(f"[video_worker] transformer model unavailable, falling back to FER+: {exc}")
            self.transformer_processor = None
            self.transformer_model = None

    def _try_load_ferplus(self, resolved_model_path: str):
        if not os.path.exists(resolved_model_path):
            print(f"[video_worker] FER+ ONNX model missing at {resolved_model_path}")
            return

        self.fer_session = ort.InferenceSession(
            resolved_model_path,
            providers=["CPUExecutionProvider"],
        )
        self.fer_input_name = self.fer_session.get_inputs()[0].name
        self.fer_model_name = os.path.basename(resolved_model_path)

    def infer(self, req_data: dict) -> dict:
        hints = dict(req_data.get("quality_hints", {}))
        rgb_image = req_data.get("rgb_image")
        tensor = req_data["tensor"]

        transformer_result = None
        if self.transformer_model is not None and rgb_image is not None:
            transformer_result = self._infer_transformer(rgb_image)

        fer_result = None
        if self.fer_session is not None:
            fer_result = self._infer_ferplus(tensor)

        result = self._select_visual_result(transformer_result, fer_result)

        face_detected = bool(hints.get("face_detected", False))
        face_area_ratio = float(hints.get("face_area_ratio", 0.0))
        face_center_offset = float(hints.get("face_center_offset", 0.0))

        face_quality_mult = 1.0 if face_detected else 0.3
        if face_area_ratio < 0.05:
            face_quality_mult *= 0.45
        elif face_area_ratio < 0.10:
            face_quality_mult *= 0.72

        if face_center_offset > 0.35:
            face_quality_mult *= 0.85

        adjusted_conf = float(max(0.0, min(1.0, result["confidence"] * face_quality_mult)))

        return {
            "val": result["val"],
            "aro": result["aro"],
            "conf": adjusted_conf,
            "raw_class": result["raw_class"],
            "raw_label": result["raw_label"],
            "raw_conf": result["confidence"],
            "vision_model": result["provider"],
            "vision_model_name": result["model_name"],
            "face_detected": face_detected,
            "face_area_ratio": face_area_ratio,
            "face_center_offset": face_center_offset,
            "yaw": hints.get("yaw", 0.0),
            "pitch": hints.get("pitch", 0.0),
            "blur": hints.get("blur", 100.0),
        }

    def _select_visual_result(self, transformer_result: dict | None, fer_result: dict | None) -> dict:
        if transformer_result is None and fer_result is None:
            return {
                "provider": "fallback",
                "model_name": "neutral-fallback",
                "raw_class": 0,
                "raw_label": "neutral",
                "confidence": 0.0,
                "val": 0.0,
                "aro": 0.0,
            }

        if transformer_result is None:
            return fer_result

        if fer_result is None:
            return transformer_result

        t_label = transformer_result["raw_label"]
        t_conf = transformer_result["confidence"]
        f_label = fer_result["raw_label"]
        f_conf = fer_result["confidence"]

        negative_labels = {"sad", "disgust", "fear", "contempt", "anger"}
        stable_fer_labels = {"neutral", "happy", "surprise", "anger"}

        if t_label == f_label:
            winner = transformer_result if t_conf >= f_conf else fer_result
            return {
                **winner,
                "provider": "hybrid-consensus",
                "model_name": f"{transformer_result['model_name']} + {fer_result['model_name']}",
                "confidence": max(t_conf, f_conf),
            }

        if f_label == "neutral" and t_label in negative_labels and t_conf >= 0.45:
            return {
                **transformer_result,
                "provider": "hybrid-transformer-negative-override",
            }

        if f_label in stable_fer_labels and f_conf >= 0.70:
            return {
                **fer_result,
                "provider": "hybrid-fer-anchor",
            }

        if t_label in negative_labels and t_conf >= (f_conf + 0.08):
            return {
                **transformer_result,
                "provider": "hybrid-transformer-override",
            }

        if f_conf >= t_conf:
            return {
                **fer_result,
                "provider": "hybrid-fer-preferred",
            }

        return {
            **transformer_result,
            "provider": "hybrid-transformer-preferred",
        }

    def _infer_transformer(self, rgb_image: np.ndarray) -> dict:
        with torch.inference_mode():
            inputs = self.transformer_processor(images=rgb_image, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            logits = self.transformer_model(**inputs).logits[0]
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

        class_idx = int(np.argmax(probs))
        confidence = float(probs[class_idx])
        raw_label = str(self.transformer_model.config.id2label[class_idx])
        canonical_label = canonicalize_label(raw_label)
        mapped_val, mapped_aro = LABEL_TO_VA.get(canonical_label, (0.0, 0.0))

        return {
            "provider": "transformers",
            "model_name": self.transformer_model_name,
            "raw_class": LABEL_TO_INDEX.get(canonical_label, 0),
            "raw_label": canonical_label,
            "confidence": confidence,
            "val": mapped_val,
            "aro": mapped_aro,
        }

    def _infer_ferplus(self, tensor: np.ndarray) -> dict:
        if self.fer_session is None or self.fer_input_name is None:
            return {
                "provider": "fallback",
                "model_name": "neutral-fallback",
                "raw_class": 0,
                "raw_label": "neutral",
                "confidence": 0.0,
                "val": 0.0,
                "aro": 0.0,
            }

        logits = self.fer_session.run(None, {self.fer_input_name: tensor})[0][0]
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        class_idx = int(np.argmax(probs))
        confidence = float(probs[class_idx])
        mapped_val, mapped_aro = FER_VA_MAP.get(class_idx, (0.0, 0.0))

        return {
            "provider": "ferplus-onnx",
            "model_name": self.fer_model_name or "emotion-ferplus-8.onnx",
            "raw_class": class_idx,
            "raw_label": CANONICAL_LABELS[class_idx],
            "confidence": confidence,
            "val": mapped_val,
            "aro": mapped_aro,
        }


def canonicalize_label(label: str) -> str:
    normalized = str(label or "neutral").strip().lower().replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())
    return LABEL_ALIASES.get(normalized, "neutral")
