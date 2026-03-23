import asyncio
import time


class SessionStateManager:
    """Instantiated per websocket connection. No shared state across sessions."""

    def __init__(self):
        self.lock = asyncio.Lock()
        self.audio_state = None
        self.video_state = None

    async def update_audio(self, result: dict):
        async with self.lock:
            result["ts"] = time.time() * 1000
            self.audio_state = result

    async def update_video(self, result: dict):
        async with self.lock:
            result["ts"] = time.time() * 1000
            self.video_state = result

    async def get_states(self):
        async with self.lock:
            a_copy = dict(self.audio_state) if self.audio_state else None
            v_copy = dict(self.video_state) if self.video_state else None
            return a_copy, v_copy
