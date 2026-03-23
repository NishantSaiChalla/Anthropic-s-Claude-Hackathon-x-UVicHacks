from temporal import SessionTemporalState
import os
from datetime import datetime

DEBUG_FILE = os.path.join(os.path.dirname(__file__), "fusion_debug.txt")

def log_debug(msg: str):
    with open(DEBUG_FILE, "a") as f:
        f.write(f"{datetime.now().isoformat()} - {msg}\n")


def map_categorical(v: float, a: float, confidence: float = 1.0) -> str:
    if confidence < 0.18:
        return "Uncertain"

    if a >= 0.72:
        if v >= 0.10:
            return "Surprised"
        if v <= -0.45:
            return "Angry"
        return "Fearful"

    if a >= 0.38:
        if v >= 0.38:
            return "Happy"
        if v <= -0.45:
            return "Angry"
        if v < -0.12:
            return "Stressed"

    if a <= -0.30:
        if v >= 0.28:
            return "Calm"
        if v <= -0.32:
            return "Sad"

    if v >= 0.48:
        return "Content"
    if v <= -0.58:
        return "Disgusted"

    if abs(v) <= 0.12 and abs(a) <= 0.12:
        return "Neutral"

    if v < -0.18 and a > 0.18:
        return "Stressed"
    if v < -0.22 and a < -0.08:
        return "Sad"
    if v > 0.18 and a > 0.18:
        return "Happy"
    if v > 0.18 and a < 0.00:
        return "Content"
    if v < -0.18:
        return "Disgusted"

    return "Neutral"


def compute_fusion(
    a_state: dict | None,
    v_state: dict | None,
    temporal: SessionTemporalState,
    now_ms: float,
) -> dict:
    a_stale_ms = (now_ms - a_state["ts"]) if a_state else 9999
    v_stale_ms = (now_ms - v_state["ts"]) if v_state else 9999

    a_valid = (a_stale_ms < 1500) and (a_state is not None) and a_state.get("valid", False)
    v_valid = (v_stale_ms < 500) and (v_state is not None)

    # ── audio weight ──
    w_a, val_a, aro_a = 0.0, 0.0, 0.0
    audio_vad_active = False
    if a_valid:
        val_a, aro_a = a_state["val"], a_state["aro"]
        audio_vad_active = a_state["vad_prob"] >= 0.5
        w_a = a_state["conf"] * (1.0 if audio_vad_active else 0.0)

    # ── video weight ──
    w_v, val_v, aro_v = 0.0, 0.0, 0.0
    pose_good = True
    blur_good = True
    face_detected = True
    face_area_ratio = 0.0
    if v_valid:
        pose_good = abs(v_state["yaw"]) <= 30 and abs(v_state["pitch"]) <= 30
        blur_good = v_state["blur"] >= 100
        face_detected = bool(v_state.get("face_detected", True))
        face_area_ratio = float(v_state.get("face_area_ratio", 0.0))
        pose_mult = 1.0 if pose_good else 0.55
        blur_mult = 1.0 if blur_good else 0.6
        face_mult = 1.0 if face_detected else 0.25
        if face_area_ratio < 0.05:
            face_mult *= 0.4
        elif face_area_ratio < 0.10:
            face_mult *= 0.7
        w_v = v_state["conf"] * pose_mult * blur_mult * face_mult
        val_v, aro_v = v_state["val"], v_state["aro"]

    total_w = w_a + w_v
    label_conf = round(float(total_w), 4)

    if total_w >= 0.15:
        signal_status = (
            "FULL_AV"
            if (w_a >= 0.15 and w_v >= 0.15)
            else ("AUDIO_ONLY" if w_a >= 0.15 else "VIDEO_ONLY")
        )
        raw_v = ((w_a * val_a) + (w_v * val_v)) / total_w
        raw_a = ((w_a * aro_a) + (w_v * aro_v)) / total_w
    else:
        signal_status = "INSUFFICIENT"
        raw_v, raw_a = 0.0, 0.0

    final_v, final_a, sys_state, is_stale, hold_reason = temporal.update_and_decay(
        raw_v, raw_a, signal_status, now_ms, confidence=total_w
    )

    res = {
        "timestamp_ms": int(now_ms),
        "system_state": sys_state,
        "signal_status": signal_status,
        "is_stale": is_stale,
        "hold_reason": hold_reason,
        "fused_emotion": {
            "valence": round(float(final_v), 4),
            "arousal": round(float(final_a), 4),
            "mapped_categorical": map_categorical(final_v, final_a, confidence=total_w),
            "label_confidence": label_conf,
        },
        "debug": {
            "raw_audio_valence": round(float(val_a), 4) if a_valid else 0.0,
            "raw_audio_arousal": round(float(aro_a), 4) if a_valid else 0.0,
            "raw_video_valence": round(float(val_v), 4) if v_valid else 0.0,
            "raw_video_arousal": round(float(aro_v), 4) if v_valid else 0.0,
            "raw_video_class": v_state.get("raw_class", -1) if (v_valid and v_state) else -1,
            "raw_video_label": v_state.get("raw_label", "unknown") if (v_valid and v_state) else "unknown",
            "raw_video_model": v_state.get("vision_model", "unknown") if (v_valid and v_state) else "unknown",
            "raw_video_confidence": round(float(v_state.get("raw_conf", 0.0)), 4) if (v_valid and v_state) else 0.0,
            "raw_video_face_area_ratio": round(float(face_area_ratio), 4) if v_valid else 0.0,
            "fused_valence": round(float(raw_v), 4),
            "fused_arousal": round(float(raw_a), 4),
            "label_confidence": label_conf,
        },
        "modality_weights": {
            "audio": round(float(w_a), 4),
            "video": round(float(w_v), 4),
        },
        "quality_flags": {
            "audio_vad_active": audio_vad_active,
            "video_face_detected": v_valid and face_detected,
            "video_face_small_warning": v_valid and face_detected and face_area_ratio < 0.10,
            "video_blur_warning": v_valid and not blur_good,
            "video_pose_warning": v_valid and not pose_good,
        },
        "stale_flags": {
            "audio_stale_ms": int(a_stale_ms),
            "video_stale_ms": int(v_stale_ms),
        },
    }
    
    # Debug logging
    try:
        log_debug(f"Fusion: V={float(final_v):.3f}, A={float(final_a):.3f}, Label={res['fused_emotion']['mapped_categorical']}, Signals: A_conf={(a_state['conf'] if a_valid else 0):.2f}, V_conf={(v_state['conf'] if v_valid else 0):.2f}")
    except:
        pass
        
    return res
