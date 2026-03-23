import os
import numpy as np
import onnxruntime as ort

# FER+ class index -> (valence, arousal) in [-1, 1]
FER_VA_MAP = {
    0: (0.0, 0.0),    # neutral
    1: (0.8, 0.4),    # happiness
    2: (0.6, 0.7),    # surprise
    3: (-0.7, -0.3),  # sadness
    4: (-0.6, 0.8),   # anger
    5: (-0.5, 0.1),   # disgust
    6: (-0.4, 0.9),   # fear
    7: (-0.3, 0.1),   # contempt
}


class VideoWorker:
    def __init__(self, model_path: str = "emotion-ferplus-8.onnx"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"ONNX model missing at '{model_path}'. "
                "Download from https://github.com/onnx/models/tree/main/validated/vision/body_analysis/emotion_ferplus "
                "before starting the server."
            )

        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def infer(self, req_data: dict) -> dict:
        tensor = req_data["tensor"]  # float32 [1, 1, 64, 64]

        logits = self.session.run(None, {self.input_name: tensor})[0][0]

        # Stabilized softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        class_idx = int(np.argmax(probs))
        confidence = float(probs[class_idx])

        mapped_val, mapped_aro = FER_VA_MAP.get(class_idx, (0.0, 0.0))

        hints = req_data.get("quality_hints", {})
        return {
            "val": mapped_val,
            "aro": mapped_aro,
            "conf": confidence,
            "raw_class": class_idx,
            "raw_conf": confidence,
            "yaw": hints.get("yaw", 0.0),
            "pitch": hints.get("pitch", 0.0),
            "blur": hints.get("blur", 100.0),
        }
