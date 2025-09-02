import asyncio
import json
from common_utils import myutils
from db.base_db_handler import DBHandler
from db.sqlite_db_handler import SQLiteDBHandler
from typing import Any, Dict, List, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn

# =========================
# FastAPI WebSocket Backend
# =========================

# Start FastAPI app
app: FastAPI = FastAPI()

# --- Global State ---
# Maps client id to websocket connection
connected_clients: Dict[str, WebSocket] = {}
# Maps scene name to a set of client ids interested in that scene
scene_clients: Dict[str, Set[str]] = {}
# Maps client id to an asyncio.Event for notifying when model outputs are ready
client_events: Dict[str, asyncio.Event] = {}
# Maps model name to a queue of fragments to be processed
model_inputs: Dict[str, asyncio.Queue] = {}
# Maps scene name to a dictionary of model outputs, e.g. {'scene1': {'neucon': b'...', 'slam3r': b'...'}}
model_outputs: Dict[str, Dict[str, myutils.ModelResult]] = {}

# =========================
# Client WebSocket Endpoint
# =========================
@app.websocket("/ws/client")
async def websocket_client_endpoint(websocket: WebSocket):
    """
    Handles client WebSocket connections.
    - Registers the client and its scene of interest.
    - Receives data fragments from the client and forwards them to the appropriate model queue.
    - Notifies the client when model results are available.
    """
    await websocket.accept()

    client_id = str(id(websocket))
    client_role = None
    client_scene = None

    # --- Handshake: Expect 'role' and 'scene' in the first message ---
    try:
        handshake = await websocket.receive_json()
        assert 'role' in handshake, "Handshake must contain 'role'"
        client_role = handshake['role']
        assert 'scene' in handshake, "Handshake must contain 'scene'"
        client_scene = handshake['scene']
    except Exception as e:
        print(f"Error receiving handshake: {e}")
        await websocket.close(code=1008)
        return
    
    available_models: List[str] = list(model_inputs.keys())
    await websocket.send_json(available_models)

    # Register client in global state
    connected_clients[client_id] = websocket
    client_events.setdefault(client_id, asyncio.Event())
    scene_clients.setdefault(client_scene, set()).add(client_id)
    print(f"Client connected: {client_id}, with role: {client_role}, scene: {client_scene}")

    # If model output is already available for this scene, send it immediately
    if client_scene in model_outputs:
        print(f"Sending model outputs for scene: {client_scene} to client: {client_id}")
        for model_name, result in model_outputs[client_scene].items():
            await websocket.send_bytes(result.Serialize())
    else:
        print(f"No model outputs available for scene: {client_scene}")

    # --- Background task to send model results to the client when available ---
    async def send_task():
        try:
            while True:
                await client_events[client_id].wait()  # Wait for notification
                client_events[client_id].clear()       # Reset the event

                # Send all available model outputs for the client's scene
                if client_id in scene_clients.get(client_scene, set()):
                    for model_name, model_result in model_outputs.get(client_scene, {}).items():
                        print(f"Sending model output for scene: {client_scene}, model: {model_name} to client: {client_id}")
                        await websocket.send_bytes(model_result.Serialize())
        except Exception as e:
            print(f"Send task error for client {client_id}: {e}")

    send_loop = asyncio.create_task(send_task())

    # --- Main receive loop: handle incoming data from the client ---
    try:
        while True:
            # 0:4 : magic, 4:8 : version, 8:12 : window size
            data = await websocket.receive_bytes()
            (magic, version) = data[0:4], int.from_bytes(data[4:8], 'little')

            if(magic != b'LEON'):
                print(f"Received invalid data from client {client_id}, expected LEON magic bytes")
                continue
        
            if(version == 1): # Fragment request
                model_name_length = int.from_bytes(data[12:12 + 4], 'little')
                model_name = data[16:16 + model_name_length].decode('utf-8')
                scene_name_length = int.from_bytes(data[16 + model_name_length:16 + model_name_length + 4], 'little')
                scene_name = data[20 + model_name_length:20 + model_name_length + scene_name_length].decode('utf-8')
                print(f"Received data for model: {model_name}, scene: {scene_name} from client: {client_id}")

                if model_name in model_inputs:
                    await model_inputs[model_name].put(data)
                else:
                    print(f"Model {model_name} not found, sending error to client")
                    await websocket.send_text(json.dumps({"error": f"No model with name '{model_name}' found"}))
            
            elif(version == 2): # Translate request
                transform_request: Dict[str, Any] = myutils.DeserializeTransformFragment(data)
                print(f"Received Translation data: {transform_request} from client: {client_id}")

                scene_name = transform_request['name']
                if(scene_name not in model_outputs):
                    print(f"No scene named '{scene_name}' found in model outputs. Ignoring transform request.")
                    continue

                for _, result in model_outputs[scene_name].items():
                    result.SetTranslation(*transform_request['translation'])
                    result.SetRotationFromQuaternion(*transform_request['rotation'])
                    result.SetScale(*transform_request['scale'])

                # Notify all clients interested in this scene
                for cid in scene_clients.get(scene_name, set()):
                    if cid in client_events and cid != client_id:
                        client_events[cid].set()
    except WebSocketDisconnect:
        print(f"Client disconnected: {client_id}")
        del connected_clients[client_id]
    finally:
        client_events.pop(client_id, None)
        send_loop.cancel()

# =========================
# Model WebSocket Endpoint
# =========================
@app.websocket("/ws/model/{model_name}")
async def websocket_model_endpoint(websocket: WebSocket, model_name: str):
    """
    Handles model WebSocket connections.
    - Receives data fragments from the queue and sends them to the model.
    - Receives results from the model and stores them in model_outputs.
    - Notifies all interested clients when a new result is available.
    """
    await websocket.accept()

    queue = asyncio.Queue()
    model_inputs[model_name] = queue
    print(f"Model [{model_name}] connected")

    try:
        with SQLiteDBHandler("db/results.db") as dbhandler:
            while True:
                # Wait for a fragment from any client
                fragment: bytes = await queue.get()
                await websocket.send_bytes(fragment)
                print(f"Sent fragment from queue to model [{model_name}]")

                # Wait for the model's result
                result_bytes: bytes = await websocket.receive_bytes()

                # 0 for failure, 1 for successful but unimportant result
                if(result_bytes == b'0' or result_bytes == b'1'):
                    print(f"Model [{model_name}] sent no result. Continuing to next fragment.")
                    continue

                result: myutils.ModelResult = myutils.DeserializeResult(result_bytes)
                print(f"Received result for scene: {result.scene_name} from model: {model_name}, PointCloud: {result.is_pointcloud}")

                # Store the result by scene and model
                scene = model_outputs.setdefault(result.scene_name, {})
                if model_name not in scene:
                    scene[model_name] = result
                else:
                    scene[model_name].SetGLB(result.output)
                
                # Store the result in the database
                dbhandler.insert_result(model_name, result)
                print(f"Inserted result for scene: {result.scene_name}, model: {model_name} into database")

                # Notify all clients interested in this scene
                for client_id in scene_clients.get(result.scene_name, set()):
                    if client_id in client_events:
                        client_events[client_id].set()
    except WebSocketDisconnect as e:
        print(f"Model {model_name} disconnected, because: {e}")
    finally:
        del model_inputs[model_name]

# =========================
# Run the FastAPI app
# =========================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
