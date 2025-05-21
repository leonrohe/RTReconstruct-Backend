import asyncio
import json
import os
from models.base_model import BaseReconstructionModel

SERVER_URL = os.getenv("SERVER_URL", "ws://router:5000/ws/model")

class NeuConReconstructionModel(BaseReconstructionModel):
   """
   NeuConReconstructionModel is a subclass of BaseReconstructionModel.
   It is designed to handle the reconstruction of fragments from a WebSocket connection.
   """
   def __init__(self, model_name: str, server_url: str = SERVER_URL):
       super().__init__(model_name, server_url)
  
   async def handle_fragment(self, fragment: dict):
        await asyncio.sleep(1)  # Simulate processing time

        await self.send_result({"result":"yippie"})


if __name__ == "__main__":
   model = NeuConReconstructionModel("neural_recon")
   asyncio.run(model.connect())