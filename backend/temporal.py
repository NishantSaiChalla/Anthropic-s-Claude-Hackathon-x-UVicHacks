class SessionTemporalState:
    """Per-session temporal smoothing with EMA and decay-on-signal-loss."""

    def __init__(self, ema_alpha: float = 0.3):
        self.ema_alpha = ema_alpha
        self.last_v = 0.0
        self.last_a = 0.0
        self.time_stale_ms = 0.0
        self.last_eval_time = None

    def update_and_decay(
        self,
        raw_v: float,
        raw_a: float,
        signal_status: str,
        now_ms: float,
        confidence: float = 0.5,
    ):
        delta_ms = (now_ms - self.last_eval_time) if self.last_eval_time else 200
        self.last_eval_time = now_ms

        sys_state = "STABLE"
        hold_reason = None
        is_stale = False

        if signal_status in ("FULL_AV", "AUDIO_ONLY", "VIDEO_ONLY"):
            effective_alpha = max(self.ema_alpha, min(0.92, 0.25 + confidence))
            
            self.last_v = (effective_alpha * raw_v) + (
                (1 - effective_alpha) * self.last_v
            )
            self.last_a = (effective_alpha * raw_a) + (
                (1 - effective_alpha) * self.last_a
            )
            self.time_stale_ms = 0.0
            sys_state = "STABLE" if signal_status == "FULL_AV" else "DEGRADED"
        else:
            is_stale = True
            self.time_stale_ms += delta_ms

            if self.time_stale_ms >= 5000:
                self.last_v, self.last_a = 0.0, 0.0
                sys_state = "UNKNOWN"
                hold_reason = "BOTH_MODALITIES_LOST"
            else:
                sys_state = "HELD_STALE"
                hold_reason = "TEMPORARY_LOSS"
                if self.time_stale_ms >= 2000:
                    decay = (5000 - self.time_stale_ms) / 3000.0
                    self.last_v *= decay
                    self.last_a *= decay

        return self.last_v, self.last_a, sys_state, is_stale, hold_reason
