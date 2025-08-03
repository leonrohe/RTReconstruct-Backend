import asyncio
import json
import websockets
from abc import ABC, abstractmethod
from scripts import myutils

import logging

# =========================
# Logging Setup
# =========================
# logger = logging.getLogger("websockets")
# logger.setLevel(logging.DEBUG)
# logger.addHandler(logging.StreamHandler())

# =========================
# Base Reconstruction Model
# =========================
class BaseReconstructionModel(ABC):
    """
    Abstract base class for reconstruction models that communicate with the backend server via WebSocket.
    Handles connection management, receiving fragments, and sending results.
    Subclasses must implement the handle_fragment() method.
    """

    def __init__(self, model_name: str, server_url: str = "ws://localhost:5000/ws/model"):
        """
        Initialize the model with a name and server URL.

        Args:
            model_name (str): The name of the model (used for routing).
            server_url (str): The base URL of the backend server.
        """
        self.model_name = model_name
        self.server_url = f"{server_url}/{model_name}"
        self.ws = None
        self.result_idx = 0

    async def connect(self):
        """
        Establishes a WebSocket connection to the backend server and starts
        listening for fragments and processing them.
        """
        async with websockets.connect(self.server_url, max_size=None, ping_timeout=300) as websocket:
            self.ws = websocket
            print(f"[{self.model_name}] Connected to server")
            # Run listen() and process_loop() concurrently
            await asyncio.gather(self.listen(), self.process_loop())

    async def listen(self):
        """
        Listens for incoming fragments from the server.
        For each received fragment, spawns a task to handle it.
        """
        async for message in self.ws:
            print(f"[{self.model_name}] Received fragment of size: {len(message)}...")
            fragment = myutils.DeserializeFragment(message)
            asyncio.create_task(self.handle_fragment(fragment))

    async def send_result(self, result: myutils.ModelResult):
        """
        Sends the processed result back to the server.

        Args:
            result (myutils.ModelResult): The result to send back to the server.
        """
        try:
            print("Sending result to server...")
            if(result is None):
                print("No result to send. Sending empty result.")
                await self.ws.send(b'')
                return

            result_bytes = result.Serialize()
            await self.ws.send(result_bytes)
            print(f"[{self.model_name}] Sent {len(result_bytes)} bytes")
        except (websockets.ConnectionClosed, BrokenPipeError) as e:
            print(f"[{self.model_name}] Connection lost while sending result: {e}")

    async def process_loop(self):
        """
        Main processing loop for the model.
        Subclasses can override this to implement custom background tasks.
        By default, it just sleeps to avoid busy-waiting.
        """
        while True:
            await asyncio.sleep(1)  # Implement a delay to avoid busy-waiting

    @abstractmethod
    async def handle_fragment(self, fragment: dict):
        """
        Abstract method to process a received fragment.
        Must be implemented by subclasses.
        Should process the fragment and send the result via `send_result()`.

        Args:
            fragment (dict): The fragment data received from the server.
        """
        pass