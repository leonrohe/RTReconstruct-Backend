import asyncio
import json
import websockets
from abc import ABC, abstractmethod
from scripts import myutils

import logging

logger = logging.getLogger("websockets")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

class BaseReconstructionModel(ABC):
    def __init__(self, model_name: str, server_url: str = "ws://localhost:5000/ws/model"):
        self.model_name = model_name
        self.server_url = f"{server_url}/{model_name}"
        self.ws = None
        self.result_idx = 0

        # Useful debug info
        self.bytes_send = 0
        self.fragments_received = 0

    async def connect(self):
        async with websockets.connect(self.server_url, max_size=None, ping_timeout=300) as websocket:
            self.ws = websocket
            print(f"[{self.model_name}] Connected to server")
            await asyncio.gather(self.listen(), self.process_loop())

    async def listen(self):
        async for message in self.ws:
            fragment = myutils.DeserializeFragment(message)
            self.fragments_received += 1
            print(f"[{self.model_name}] received {self.fragments_received} fragments so far")
            asyncio.create_task(self.handle_fragment(fragment))

    async def send_result(self, result: bytes):
        try:
            print("Sending result to server...")
            await self.ws.send(result)
            self.bytes_send += len(result)
        except (websockets.ConnectionClosed, BrokenPipeError) as e:
            print(f"[{self.model_name}] Connection lost while sending result: {e}")
        finally:
            self.result_idx += 1
            print(f"[{self.model_name}] Sent {len(result)} bytes, total sent: {self.bytes_send} bytes")

    async def process_loop(self):
        while True:
            await asyncio.sleep(1)  # Implement a delay to avoid busy-waiting
            

    @abstractmethod
    async def handle_fragment(self, fragment: dict):
        """
        This method must be implemented by subclasses.
        It should process the fragment and send the result via `send_result()`.
        """
        pass