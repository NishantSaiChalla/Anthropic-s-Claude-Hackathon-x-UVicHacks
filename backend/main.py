import asyncio
import time
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from state_manager import SessionStateManager
from temporal import SessionTemporalState
from audio_worker import AudioWorker
from video_worker import VideoWorker
from preprocessing import decode_audio_b64, decode_image_b64_ferplus
from fusion import compute_fusion

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("emotion_engine")

app = FastAPI()

# Shared globally: stateless, no per-request mutation
global_audio_worker = AudioWorker()
global_video_worker = VideoWorker()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected")

    # Per-connection state
    session_state = SessionStateManager()
    session_temporal = SessionTemporalState()

    async def fusion_loop():
        while True:
            await asyncio.sleep(0.2)  # 5 Hz
            now = time.time() * 1000
            a_state, v_state = await session_state.get_states()
            result = compute_fusion(a_state, v_state, session_temporal, now)
            await websocket.send_json(result)

    loop_task = asyncio.create_task(fusion_loop())

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "audio":
                try:
                    payload = decode_audio_b64(data["payload_base64"])
                except (ValueError, KeyError) as e:
                    logger.warning("Bad audio payload: %s", e)
                    continue
                asyncio.create_task(
                    _process_audio({"array": payload}, session_state)
                )

            elif msg_type == "video":
                try:
                    payload = decode_image_b64_ferplus(data["payload_b64_jpg"])
                except (ValueError, KeyError) as e:
                    logger.warning("Bad video payload: %s", e)
                    continue
                req = {
                    "tensor": payload,
                    "quality_hints": data.get("quality_hints", {}),
                }
                asyncio.create_task(
                    _process_video(req, session_state)
                )
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error("WS error: %s", e)
    finally:
        loop_task.cancel()


async def _process_audio(req: dict, state: SessionStateManager):
    res = await asyncio.to_thread(global_audio_worker.infer, req)
    await state.update_audio(res)


async def _process_video(req: dict, state: SessionStateManager):
    res = await asyncio.to_thread(global_video_worker.infer, req)
    await state.update_video(res)
