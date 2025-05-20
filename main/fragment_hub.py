import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import Dict
import uvicorn

app = FastAPI()

connected_clients: Dict[str, WebSocket] = {}
model_queues: Dict[str, asyncio.Queue] = {}  # name -> queue of fragments

@app.websocket("/ws/client")
async def websocket_client_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_id = f"client_{id(websocket)}"
    connected_clients[client_id] = websocket
    print(f"Client connected: {client_id}")

    try:
        while True:
            data = await websocket.receive_text()
            fragment = json.loads(data)
            model_name = fragment.get("name")
            fragment["client_id"] = client_id

            print(f"Forwarding fragment to model: {model_name}")
            if model_name in model_queues:
                await model_queues[model_name].put(fragment)
            else:
                await websocket.send_text(json.dumps({"error": "Model not available"}))
    except WebSocketDisconnect:
        print(f"Client disconnected: {client_id}")
        del connected_clients[client_id]

@app.websocket("/ws/model/{model_name}")
async def websocket_model_endpoint(websocket: WebSocket, model_name: str):
    await websocket.accept()
    queue = asyncio.Queue()
    model_queues[model_name] = queue
    print(f"Model {model_name} connected")

    async def send_task():
        while True:
            fragment = await queue.get()
            await websocket.send_text(json.dumps(fragment))

    send_loop = asyncio.create_task(send_task())

    try:
        while True:
            result = await websocket.receive_text()
            data = json.loads(result)
            client_id = data.get("client_id")
            if client_id in connected_clients:
                await connected_clients[client_id].send_text(json.dumps(data))
    except WebSocketDisconnect:
        print(f"Model {model_name} disconnected")
        send_loop.cancel()
        del model_queues[model_name]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
