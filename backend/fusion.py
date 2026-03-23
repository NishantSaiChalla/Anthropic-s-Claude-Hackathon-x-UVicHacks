from temporal import SessionTemporalState
import os
from datetime import datetime

DEBUG_FILE = os.path.join(os.path.dirname(__file__), "fusion_debug.txt")

def log_debug(msg: str):
    with open(DEBUG_FILE, "a") as f:
        f.write(f"{datetime.now().isoformat()} - {msg}\n")


def map_categorical(v: float, a: float, confidence: float = 1.0) -> str:
    """Nuanced mapping of Valence/Arousal to categorical labels."""
    if confidence < 0.15:
        return "Uncertain"
    
    # ── High Arousal (Surprised, Fearful, Angry, Excited) ─────────
    if a > 0.3: # WAS 0.4
        if v > 0.2: # WAS 0.4
            return "Surprised" if a > 0.55 else "Excited"
        if v < -0.15: # WAS -0.3
            return "Angry"
        return "Stressed" if v < 0 else "Alert"
    
    # ── High Valence (Happy, Content, Calm) ───────────────────────
    if v > 0.2: # WAS 0.35
        if a > 0.0: return "Happy" # WAS 0.1
        if a < -0.2: return "Calm"
        return "Content"
    
    # ── Low Valence (Sad, Disgusted, Frustrated) ──────────────────
    if v < -0.2: # WAS -0.35
        if a < 0.1: return "Sad" # WAS -0.1
        return "Disgusted" if a > 0.2 else "Frustrated"
    
    # ── Low Arousal (Bored, Tired) ───────────────────────────────
    if a < -0.25: # WAS -0.35
        return "Calm" if v > 0 else "Bored"
    
    # ── Center Region (Neutral) ──────────────────────────────────
    if abs(v) < 0.12 and abs(a) < 0.12: # WAS 0.2
        return "Neutral"
    
    # Fallback for anything else
    if v > 0:
        return "Positive"
    else:
        return "Negative"


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
    if v_valid:
        pose_good = abs(v_state["yaw"]) <= 30 and abs(v_state["pitch"]) <= 30
        blur_good = v_state["blur"] >= 100
        w_v = v_state["conf"] * (1.0 if (pose_good and blur_good) else 0.2)
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
            "raw_video_confidence": round(float(v_state.get("raw_conf", 0.0)), 4) if (v_valid and v_state) else 0.0,
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
            "video_face_detected": v_valid,
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
