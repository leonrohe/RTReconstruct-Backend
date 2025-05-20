import asyncio
from base_model import BaseReconstructionModel

class Neucon(BaseReconstructionModel):
    async def handle_fragment(self, fragment: dict):
        # Simulate async reconstruction
        await asyncio.sleep(1)

        # Example dummy mesh result
        result = {
            "client_id": fragment["client_id"],
            "mesh_data": f"Dummy mesh from {self.model_name}"
        }
        await self.send_result(result)
