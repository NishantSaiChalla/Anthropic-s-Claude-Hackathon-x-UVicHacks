import base64
import cv2
import numpy as np


_FACE_CASCADE_PATHS = [
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
    cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml",
]
_FACE_CASCADES = [
    cascade
    for cascade in (cv2.CascadeClassifier(path) for path in _FACE_CASCADE_PATHS)
    if not cascade.empty()
]


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


def decode_image_b64_ferplus(payload_b64: str) -> dict:
    """Decode webcam JPEG and return a face-cropped FER+ tensor plus richer quality hints."""
    raw_bytes = base64.b64decode(payload_b64)
    np_arr = np.frombuffer(raw_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise ValueError("Malformed image payload: unable to decode JPEG.")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    frame_h, frame_w = gray.shape[:2]
    blur = estimate_blur(gray)

    face = detect_largest_face(gray)
    face_detected = face is not None

    if face_detected:
        x1, y1, x2, y2 = expand_face_box(face, frame_w, frame_h, scale=1.4)
    else:
        x1, y1, x2, y2 = center_square_crop(frame_w, frame_h)

    face_gray = gray[y1:y2, x1:x2]
    face_bgr = img_bgr[y1:y2, x1:x2]

    if face_gray.size == 0 or face_bgr.size == 0:
        face_gray = gray
        face_bgr = img_bgr
        x1, y1, x2, y2 = 0, 0, frame_w, frame_h
        face_detected = False

    face_area_ratio = ((x2 - x1) * (y2 - y1)) / max(1.0, float(frame_w * frame_h))
    face_center_offset = compute_center_offset(x1, y1, x2, y2, frame_w, frame_h)

    return {
        "tensor": prepare_ferplus_tensor(face_gray),
        "rgb_image": prepare_transformer_face(face_bgr),
        "quality_hints": {
            "face_detected": face_detected,
            "face_area_ratio": round(float(face_area_ratio), 4),
            "face_center_offset": round(float(face_center_offset), 4),
            "blur": round(float(blur), 2),
        },
    }


def detect_largest_face(gray: np.ndarray):
    if not _FACE_CASCADES:
        return None

    min_face = max(36, min(gray.shape[:2]) // 6)
    for cascade in _FACE_CASCADES:
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(min_face, min_face),
        )
        if len(faces):
            return max(faces, key=lambda item: item[2] * item[3])

    return None


def expand_face_box(face, frame_w: int, frame_h: int, scale: float = 1.35):
    x, y, w, h = [int(v) for v in face]
    side = int(round(max(w, h) * scale))
    cx = x + (w / 2.0)
    cy = y + (h / 2.0)

    x1 = int(round(cx - (side / 2.0)))
    y1 = int(round(cy - (side / 2.0)))
    x2 = x1 + side
    y2 = y1 + side

    if x1 < 0:
        x2 += -x1
        x1 = 0
    if y1 < 0:
        y2 += -y1
        y1 = 0
    if x2 > frame_w:
        shift = x2 - frame_w
        x1 = max(0, x1 - shift)
        x2 = frame_w
    if y2 > frame_h:
        shift = y2 - frame_h
        y1 = max(0, y1 - shift)
        y2 = frame_h

    return x1, y1, x2, y2


def center_square_crop(frame_w: int, frame_h: int):
    side = min(frame_w, frame_h)
    x1 = (frame_w - side) // 2
    y1 = (frame_h - side) // 2
    return x1, y1, x1 + side, y1 + side


def compute_center_offset(x1: int, y1: int, x2: int, y2: int, frame_w: int, frame_h: int) -> float:
    face_cx = (x1 + x2) / 2.0
    face_cy = (y1 + y2) / 2.0
    frame_cx = frame_w / 2.0
    frame_cy = frame_h / 2.0
    norm_dx = (face_cx - frame_cx) / max(1.0, frame_w / 2.0)
    norm_dy = (face_cy - frame_cy) / max(1.0, frame_h / 2.0)
    return float((norm_dx ** 2 + norm_dy ** 2) ** 0.5)


def estimate_blur(gray: np.ndarray) -> float:
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def prepare_ferplus_tensor(face_gray: np.ndarray) -> np.ndarray:
    face_resized = cv2.resize(face_gray, (64, 64), interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    equalized = clahe.apply(face_resized)
    normalized = cv2.normalize(equalized, None, 0, 255, cv2.NORM_MINMAX)
    return np.expand_dims(np.expand_dims(normalized, axis=0), axis=0).astype(np.float32)


def prepare_transformer_face(face_bgr: np.ndarray) -> np.ndarray:
    resized = cv2.resize(face_bgr, (224, 224), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
