import argparse
import asyncio
import io
import json
import pickle
from PIL import Image
import traceback
import os

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
from models.NeuralRecon.datasets import transforms
from models.NeuralRecon.models.neuralrecon import NeuralRecon
from models.base_model import BaseReconstructionModel
from models.NeuralRecon.config import cfg, update_config

SERVER_URL = os.getenv("SERVER_URL", "ws://router:5000/ws/model")
TRANSFORMS = transforms.Compose([
    transforms.ResizeImage((640, 480)),
    transforms.ToTensor(),
    transforms.RandomTransformSpace(
        cfg.MODEL.N_VOX, cfg.MODEL.VOXEL_SIZE, random_rotation=False, random_translation=False,
        paddingXY=0, paddingZ=0, max_epoch=cfg.TRAIN.EPOCHS),
    transforms.IntrinsicsPoseToProjection(cfg.TEST.N_VIEWS, 4)
])

class NeuConReconstructionModel(BaseReconstructionModel):
    """
    NeuConReconstructionModel is a subclass of BaseReconstructionModel.
    It is designed to handle the reconstruction of fragments from a WebSocket connection.
    """
    def __init__(self, model_name: str, server_url: str = SERVER_URL):
        super().__init__(model_name, server_url)

    async def handle_fragment(self, fragment: dict):
        global MODEL, TRANSFORMS

        try:
            neucon_fragment = self.base_fragment_to_neucon_fragment(fragment)
            item = {
                'imgs': neucon_fragment["images"],
                'intrinsics': np.stack(neucon_fragment['intrinsics']),
                'extrinsics': np.stack(neucon_fragment['extrinsics']),
                'scene': "",
                'fragment': "",
                'epoch': [None],
                'vol_origin': np.array([0, 0, 0])
            }
            item = TRANSFORMS(item)
            item = default_collate([item])
        except Exception as e:
            await self.send_result({"error": f"Failed to load and process fragment.pkl.\nError: {e}"})
            return

        with torch.no_grad():
            try:
                outputs, _ = MODEL(item, True)
                print("Inference complete.")
                scene_tsdf = outputs['scene_tsdf'][0].data.cpu().numpy()
            except Exception as e:
                # throw stacktrace
                await self.send_result({"error": f"Failed to run inference.\nError: {e}\nTraceback: {traceback.format_exc()}"})
                return

        print(scene_tsdf)
        await self.send_result(pickle.dumps(scene_tsdf))
    
    def base_fragment_to_neucon_fragment(self, fragment: dict) -> dict:
        from transforms3d.quaternions import quat2mat

        def transform_image(image: bytes):
            image = Image.open(io.BytesIO(image)).convert("RGB")
            return image
        
        def transform_pose(pose: dict, use_homogenous=True) -> np.ndarray:
            def rotx(theta):
                """Rotation matrix around X-axis."""
                return np.array([
                    [1, 0, 0],
                    [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta),  np.cos(theta)]
                ], dtype=np.float32)

            trans = pose['camera_position']
            quat = pose['camera_rotation']

            # ARKit quaternions are [x, y, z, w] → convert to [w, x, y, z]
            quat_wxyz = np.append(quat[3], quat[:3])
            rot_mat = quat2mat(quat_wxyz.tolist())

            # ARKit coordinate adjustment
            rot_mat = rot_mat.dot(np.array([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1]
            ]))
            rot_mat = rotx(np.pi / 2) @ rot_mat
            trans = rotx(np.pi / 2) @ trans

            # Compose 3x4 pose matrix
            trans_mat = np.zeros((3, 4), dtype=np.float32)
            trans_mat[:, :3] = rot_mat
            trans_mat[:, 3] = trans

            # Make it 4x4 if needed
            if use_homogenous:
                trans_mat = np.vstack((trans_mat, [0, 0, 0, 1]))

            return trans_mat
        
        def transform_intrinsics(intrinsics: dict) -> np.ndarray:
            fx, fy = intrinsics['focal_length']
            cx, cy = intrinsics['principal_point']
            
            K = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0,  0,  1]
            ], dtype=np.float32)
            
            return K

        return {
            "images": [transform_image(image) for image in fragment["images"]],
            "extrinsics": [transform_pose(pose) for pose in fragment["extrinsics"]],
            "intrinsics": [transform_intrinsics(intrinsic) for intrinsic in fragment["intrinsics"]]
        }


if __name__ == "__main__":
    # ------------------------ CONFIG SETUP ---------------------
    parser = argparse.ArgumentParser(description='NeuralRecon Server')
    parser.add_argument(
        '--cfg',
        help='The config file to use for the server',
        required=True,
        type=str)
    parser.add_argument(
        'opts',
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    update_config(cfg, args)

    # --------------------- MODEL & TRANSFORMS INIT ---------------------
    print("Initializing the model on GPU...")
    MODEL = NeuralRecon(cfg).cuda().eval()
    MODEL = torch.nn.DataParallel(MODEL, device_ids=[0])

    SAVED_MODELS = [fn for fn in os.listdir(cfg.LOGDIR) if fn.endswith(".ckpt")]
    SAVED_MODELS = sorted(SAVED_MODELS, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    chkpt = os.path.join(cfg.LOGDIR, SAVED_MODELS[-1])

    print("Resuming from " + str(chkpt))
    STATE_DICT = torch.load(chkpt)
    EPOCH_IDX = STATE_DICT['epoch']

    MODEL.load_state_dict(STATE_DICT['model'], strict=False)

    # --------------------- WEBSOCKET CONNECTION ---------------------
    model = NeuConReconstructionModel("neural_recon")
    asyncio.run(model.connect())