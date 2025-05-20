import asyncio
import json
import websockets
from abc import ABC, abstractmethod


class BaseReconstructionModel(ABC):
    def __init__(self, model_name: str, server_url: str = "ws://localhost:5000/ws/model"):
        self.model_name = model_name
        self.server_url = f"{server_url}/{model_name}"
        self.ws = None

    async def connect(self):
        async with websockets.connect(self.server_url) as websocket:
            self.ws = websocket
            print(f"[{self.model_name}] Connected to server")
            await asyncio.gather(self.listen(), self.process_loop())

    async def listen(self):
        async for message in self.ws:
            fragment = json.loads(message)
            print(f"[{self.model_name}] Received fragment from {fragment.get('client_id')}")
            asyncio.create_task(self.handle_fragment(fragment))

    async def send_result(self, result: dict):
        await self.ws.send(json.dumps(result))

    async def process_loop(self):
        while True:
            await asyncio.sleep(1)  # Optional: health check, heartbeat, etc.

    @abstractmethod
    async def handle_fragment(self, fragment: dict):
        """
        This method must be implemented by subclasses.
        It should process the fragment and send the result via `send_result()`.
        """
        pass