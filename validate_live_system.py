import asyncio
import base64
import json
import os
import cv2
import numpy as np
import websockets
import time

BACKEND_URL = "ws://localhost:8001/ws"

IMAGES = {
    "neutral": r"C:\Users\raviv\.gemini\antigravity\brain\a6126d66-4eb9-4bc0-ac63-2c06f42d60be\neutral_face_test_1774141285070.png",
    "smiling": r"C:\Users\raviv\.gemini\antigravity\brain\a6126d66-4eb9-4bc0-ac63-2c06f42d60be\smiling_face_test_1774141374796.png",
    "sad": r"C:\Users\raviv\.gemini\antigravity\brain\a6126d66-4eb9-4bc0-ac63-2c06f42d60be\sad_face_test_1774141385255.png",
    "angry": r"C:\Users\raviv\.gemini\antigravity\brain\a6126d66-4eb9-4bc0-ac63-2c06f42d60be\angry_face_test_1774141409665.png",
    "surprised": r"C:\Users\raviv\.gemini\antigravity\brain\a6126d66-4eb9-4bc0-ac63-2c06f42d60be\surprised_face_test_1774141450533.png",
}

def get_image_b64(path: str) -> str:
    img = cv2.imread(path)
    if img is None:
        return ""
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

def get_blank_audio_b64(duration_sec: float = 2.0) -> str:
    num_samples = int(duration_sec * 16000)
    samples = np.zeros(num_samples, dtype=np.int16)
    return base64.b64encode(samples.tobytes()).decode('utf-8')

current_img_b64 = None
current_audio_b64 = None

async def sender_loop(ws):
    while True:
        try:
            if current_img_b64:
                await ws.send(json.dumps({
                    "type": "video",
                    "payload_b64_jpg": current_img_b64,
                    "quality_hints": {"blur": 150, "yaw": 0, "pitch": 0}
                }))
            if current_audio_b64:
                await ws.send(json.dumps({
                    "type": "audio",
                    "payload_base64": current_audio_b64
                }))
            await asyncio.sleep(0.2)
        except:
            break

async def main():
    global current_img_b64, current_audio_b64
    try:
        async with websockets.connect(BACKEND_URL) as ws:
            sender_task = asyncio.create_task(sender_loop(ws))
            
            async def get_latest():
                res = None
                try:
                    while True:
                        msg = await asyncio.wait_for(ws.recv(), timeout=0.1)
                        res = json.loads(msg)
                except asyncio.TimeoutError:
                    pass
                return res

            cases = [
                ("1. Neutral", IMAGES["neutral"], get_blank_audio_b64()),
                ("2. Smile", IMAGES["smiling"], None),
                ("3. Sad", IMAGES["sad"], None),
                ("4. Angry", IMAGES["angry"], None),
                ("5. Surprised", IMAGES["surprised"], None),
                ("6. Audio only", None, get_blank_audio_b64()),
                ("7. Video only", IMAGES["neutral"], None),
            ]

            results = []
            for name, img_path, audio_b64 in cases:
                current_img_b64 = get_image_b64(img_path) if img_path else None
                current_audio_b64 = audio_b64
                await asyncio.sleep(2.5)
                res = await get_latest()
                results.append((name, res))

            await asyncio.sleep(3)
            sender_task.cancel()

            print("FINAL_VALIDATION_START")
            for name, res in results:
                print(f"CASE: {name}")
                if res:
                    fe = res['fused_emotion']
                    d = res.get('debug', {})
                    print(f"  label: {fe['mapped_categorical']}")
                    print(f"  val: {fe['valence']}")
                    print(f"  aro: {fe['arousal']}")
                    print(f"  conf: {fe['label_confidence']}")
                    print(f"  weights: {res['modality_weights']}")
                    print(f"  state: {res['system_state']}")
                    print(f"  status: {res['signal_status']}")
                    print(f"  raw_v_class: {d.get('raw_video_class')}")
                    print(f"  raw_v_val: {d.get('raw_video_valence')}")
            print("FINAL_VALIDATION_END")

    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(main())
