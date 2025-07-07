import asyncio
import json
from typing import Dict, List, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn

import traceback

# Start FastAPI app
app = FastAPI()

# maps client id to websocket
connected_clients: Dict[str, WebSocket] = {}

# map scene to client id
scene_clients: Dict[str, Set[str]] = {}

client_events: Dict[str, asyncio.Event] = {}

# maps model name to a queue of fragments to be processed
model_inputs: Dict[str, asyncio.Queue] = {} 

# maps scene name to a dictionary of model outputs, e.g. {'test_scene': {'neucon': b'...', 'slam3r': b'...'}}
model_outputs: Dict[str, Dict[str, bytes]] = {}

# Client WebSocket Endpoint
@app.websocket("/ws/client")
async def websocket_client_endpoint(websocket: WebSocket):
    await websocket.accept()

    client_id = str(id(websocket))
    connected_clients[client_id] = websocket
    client_events.setdefault(client_id, asyncio.Event())
    print(f"Client connected: {client_id}")

    async def send_task():
        try:
            while True:
                await client_events[client_id].wait()  # wait for notification
                client_events[client_id].clear()       # reset the event

                for scene_name, models in model_outputs.items():
                    for model_name, result in list(models.items()):
                        if client_id in scene_clients.get(scene_name, set()):
                            await websocket.send_bytes(result)
                            model_outputs[scene_name].pop(model_name, None)
        except Exception as e:
            print(f"Send task error for client {client_id}: {e}")
    send_loop = asyncio.create_task(send_task())

    try:
        while True:
            data = await websocket.receive_bytes()
            
            # 0:4 : magic, 4:8 : version, 8:12 : window size
            # TODO: Check magic and version
            mode_name_length = int.from_bytes(data[12:16], 'little')
            model_name = data[16:16 + mode_name_length].decode('utf-8')
            scene_name = 'test_scene'  # Placeholder for scene name, can be extracted from data later if needed
            
            if model_name in model_inputs:
                print(f"Forwarding {len(data)} bytes to model: {model_name}")
                scene_clients.setdefault(scene_name, set()).add(client_id)
                await model_inputs[model_name].put(data)
            else:
                print(f"Model {model_name} not found, sending error to client")
                await websocket.send_text(json.dumps({"error": f"No model with name '{model_name}' found"}))
    except WebSocketDisconnect:
        print(f"Client disconnected: {client_id}")
        del connected_clients[client_id]
    finally:
        client_events.pop(client_id, None)
        send_loop.cancel()

# Model WebSocket Endpoint
@app.websocket("/ws/model/{model_name}")
async def websocket_model_endpoint(websocket: WebSocket, model_name: str):
    await websocket.accept()

    queue = asyncio.Queue(maxsize=100)
    model_inputs[model_name] = queue
    print(f"Model [{model_name}] connected")

    # async def send_task():
    #     try:
    #         while True:
    #             fragment = await queue.get()
    #             await websocket.send_bytes(fragment)
    #             print(f"Sent {len(fragment)} bytes to model: {model_name}")
    #     except Exception as e:
    #         print(f"Send task error for model {model_name}: {e}")

    # send_loop = asyncio.create_task(send_task())

    try:
        while True:
            fragment = await queue.get()
            await websocket.send_bytes(fragment)
            result = await websocket.receive_bytes()  

            # Extract scene_name from result if possible
            scene_name = 'test_scene'  # Replace with actual extraction
            model_outputs.setdefault(scene_name, {})[model_name] = result

            # Notify all interested clients
            for client_id in scene_clients.get(scene_name, set()):
                if client_id in client_events:
                    client_events[client_id].set()

    except WebSocketDisconnect as e:
        print(f"Model {model_name} disconnected, because: {e}")
        traceback.print_exc()
    finally:
        # send_loop.cancel()
        del model_inputs[model_name]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", ws_ping_timeout=300, log_level='debug', ws_max_size=None, port=5000)
