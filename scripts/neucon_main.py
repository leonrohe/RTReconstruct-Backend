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
from myutils import ModelResult
from models.NeuralRecon.datasets import transforms
from models.NeuralRecon.models.neuralrecon import NeuralRecon
from models.base_model import BaseReconstructionModel
from models.NeuralRecon.config import cfg, update_config
from models.NeuralRecon.utils import SaveScene

MODEL_NAME = os.getenv("MODEL_NAME", "neucon")
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
        self.fragmentIndex = 0

    async def handle_fragment(self, fragment: dict):
        global MODEL, TRANSFORMS

        # Fragment processing
        try:
            neucon_fragment = self.base_fragment_to_neucon_fragment(fragment)

            scene_name = fragment['scene_name']
            fragment_name = f"{scene_name}_{self.fragmentIndex}"
            item = {
                'imgs': neucon_fragment["images"],
                'intrinsics': np.stack(neucon_fragment['intrinsics']),
                'extrinsics': np.stack(neucon_fragment['extrinsics']),
                'scene': scene_name,
                'fragment': fragment_name,
                'epoch': [None],
                'vol_origin': np.array([0, 0, 0])
            }

            item = TRANSFORMS(item)
            item = default_collate([item])
        except Exception as e:
            print("Error during fragment processing:", e)
            print(traceback.format_exc())
            return

        # Inference
        try:
            with torch.no_grad():
                outputs, _ = MODEL(item, True)
                self.fragmentIndex += 1
                print("Inference complete.")

                if outputs == {}:
                    print("No output from the model. Resetting ...")
                    MODEL = NeuralRecon(cfg).cuda().eval()
                    MODEL = torch.nn.DataParallel(MODEL, device_ids=[0])
                    MODEL.load_state_dict(STATE_DICT['model'], strict=False)
                    print("Model reset complete.")
                    await self.send_result(None)

                tsdf = outputs['scene_tsdf'][0].data.cpu().numpy()
                origin = outputs['origin'][0].data.cpu().numpy()
                origin[2] -= 1.5

                mesh = SaveScene.tsdf2mesh(cfg.MODEL.VOXEL_SIZE, origin, tsdf)
                glb_bytes = mesh.export(file_type='glb')
                result: ModelResult = ModelResult(scene_name, glb_bytes, False)
                # result.SetTranslation(0, -2, 0)
                result.SetRotation(90, 0, 180, degrees=True)

                await self.send_result(result)
        except Exception as e:
            print("Error during inference:", e)
            print(traceback.format_exc())
            return
    
    def base_fragment_to_neucon_fragment(self, fragment: dict) -> dict:
        from transforms3d.quaternions import quat2mat

        def transform_image(image: bytes):
            image = Image.open(io.BytesIO(image)).convert("RGB")
            return image
        
        def transform_pose(pose: dict, use_homogenous=True):
            from transforms3d.quaternions import quat2mat

            def rotx(t):
                ''' 3D Rotation about the x-axis. '''
                c = np.cos(t)
                s = np.sin(t)
                return np.array([[1, 0, 0],
                                [0, c, -s],
                                [0, s, c]])

            trans = pose['camera_position']
            quat = pose['camera_rotation']

            trans[-1] = -trans[-1]
            quat[0] = -quat[0]
            quat[1] = -quat[1]


            rot_mat = quat2mat(np.append(quat[-1], quat[:3]).tolist())
            rot_mat = rot_mat.dot(np.array([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1]
            ]))
            rot_mat = rotx(np.pi / 2) @ rot_mat
            trans = rotx(np.pi / 2) @ trans

            trans_mat = np.zeros([3, 4])
            trans_mat[:3, :3] = rot_mat
            trans_mat[:3, 3] = trans
            if use_homogenous:
                trans_mat = np.vstack((trans_mat, [0, 0, 0, 1]))
            
            # Idek why this is needed, but it is in the original code
            # Supposedly to match the training data
            # lets just hardcode this value, yayyy
            # Fun Fact: This took me the whole day to figure out :))
            trans_mat[2, 3] += 1.5 

            return trans_mat

        # may need to scale them based on original image size late on
        def transform_intrinsics(intrinsics: dict):
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
    model = NeuConReconstructionModel(MODEL_NAME)
    asyncio.run(model.connect())