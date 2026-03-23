import asyncio
import websockets
import json
import base64

async def test_ws():
    uri = "ws://localhost:8001/ws"
    async with websockets.connect(uri) as websocket:
        print("Connected to WS!")
        
        # Send a dummy audio b64 (short pcm16)
        payload = base64.b64encode(bytes([0]*1000)).decode('utf-8')
        print("Sending dummy audio...")
        await websocket.send(json.dumps({
            "type": "audio",
            "payload_base64": payload,
            "timestamp_ms": 12345
        }))
        
        # Expect at least one message back
        print("Waiting for response...")
        response = await websocket.recv()
        print(f"Received: {response[:100]}...")
        data = json.loads(response)
        if "fused_emotion" in data:
            print("SUCCESS: Fused emotion received.")
        else:
            print("FAILURE: Fused emotion NOT in response.")

if __name__ == "__main__":
    asyncio.run(test_ws())
