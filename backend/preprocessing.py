import base64
import numpy as np
import cv2


def decode_audio_b64(payload_b64: str) -> np.ndarray:
    """
    Decodes base64 payload: raw PCM16 little-endian, mono, 16kHz.
    Returns float32 numpy array normalized to [-1.0, 1.0].
    """
    raw_bytes = base64.b64decode(payload_b64)

    if len(raw_bytes) == 0:
        raise ValueError("Malformed audio payload: empty buffer.")
    if len(raw_bytes) % 2 != 0:
        raise ValueError(
            "Malformed audio payload: byte length not divisible by 2 (invalid PCM16)."
        )

    audio_16k = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return audio_16k


def decode_image_b64_ferplus(payload_b64: str) -> np.ndarray:
    """Decodes base64 JPEG into FER+ float32 [1, 1, 64, 64] grayscale tensor."""
    raw_bytes = base64.b64decode(payload_b64)
    np_arr = np.frombuffer(raw_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Malformed image payload: unable to decode JPEG.")

    img_resized = cv2.resize(img, (64, 64))
    tensor = np.expand_dims(np.expand_dims(img_resized, axis=0), axis=0).astype(
        np.float32
    )
    return tensor
