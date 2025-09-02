import argparse
import asyncio
import os
import traceback
from PIL import Image
import io
import torch
from torch.utils.data._utils.collate import default_collate
import numpy as np

from utils import SaveScene
from datasets import transforms
from models import VisFusion
from config import cfg, update_config
from ops.comm import *
from datasets import transforms

from common_utils.myutils import ModelResult
from recon_models.base_model import BaseReconstructionModel

class VisFusionReconstructionModel(BaseReconstructionModel):
    def __init__(self, model_name, server_url = "ws://localhost:5000/ws/model"):
        super().__init__(model_name, server_url)
        self.fragmentIdx = 0

    async def handle_fragment(self, fragment: dict):
        global MODEL, TRANSFORMS

        # Fragment processing
        try:
            visfusion_fragment = self.transform_base_fragment(fragment)

            scene_name = fragment['scene_name']
            fragment_name = f"{scene_name}_{self.fragmentIdx}"
            item = {
                'imgs': visfusion_fragment["images"],
                'intrinsics': np.stack(visfusion_fragment['intrinsics']),
                'poses': np.stack(visfusion_fragment['extrinsics']),
                'scene': scene_name,
                'fragment': fragment_name,
                'epoch': [None],
                'vol_origin': np.array([0, 0, 0])
            }
            item = TRANSFORMS(item)
            item = default_collate([item])
        except Exception as e:
            print(f"[ERROR] Fragment processing failed, because {e}.")
            print(traceback.format_exc())
            return

        # Inference
        try:
            with torch.no_grad():
                MODEL.eval()
                outputs, loss_dict  = MODEL(item, True)
                print("[INFO] Inference complete.")

                if outputs == {}:
                    print("[ERROR] No output from the model. Resetting ...")
                    MODEL = VisFusion(cfg).cuda().eval()
                    MODEL = torch.nn.DataParallel(MODEL, device_ids=[0])
                    MODEL.load_state_dict(STATE_DICT['model'], strict=False)
                    await self.send_result(None, False)
                    return

                tsdf = outputs['scene_tsdf'][0].data.cpu().numpy()
                origin = outputs['origin'][0].data.cpu().numpy()
                origin[2] -= 1.5

                mesh = SaveScene.tsdf2mesh(cfg.MODEL.VOXEL_SIZE, origin, tsdf, cfg.MODEL.PASS_LAYERS, cfg.MODEL.SINGLE_LAYER_MESH)
                glb_bytes = mesh.export(file_type='glb')
                result: ModelResult = ModelResult(scene_name, glb_bytes, False)
                result.SetRotationFromEuler(90, 0, 180, degrees=True)

                await self.send_result(result)

                self.fragmentIdx += 1
                

        except Exception as e:
            print(f"[ERROR] Inference failed, because {e}.")
            print(traceback.format_exc())

    def transform_base_fragment(self, fragment: dict) -> dict:
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

# Augmentation (Test only)
n_views = cfg.TEST.N_VIEWS
random_rotation = False
random_translation = False
paddingXY = 0
paddingZ = 0

TRANSFORMS = [transforms.ResizeImage((640, 480)),
             transforms.ToTensor(),
             transforms.RandomTransformSpace(
                  cfg.MODEL.N_VOX, cfg.MODEL.VOXEL_SIZE, random_rotation, random_translation,
                  paddingXY, paddingZ, max_epoch=cfg.TRAIN.EPOCHS),
             transforms.IntrinsicsPoseToProjection(n_views, 4)]
TRANSFORMS = transforms.Compose(TRANSFORMS)

def args():
    parser = argparse.ArgumentParser(description='A PyTorch Implementation of VisFusion')

    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    
    args = parser.parse_args()
    return args

MODEL_NAME = os.getenv("MODEL_NAME", "visfusion")
SERVER_URL = os.getenv("SERVER_URL", "ws://router:5000/ws/model")
if __name__ == "__main__":
    # --- cfg setup ---
    args = args()
    update_config(cfg, args)
    cfg.freeze()

    # --- torch seed ---
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed(cfg.SEED)

    # --- (non dist) model setup ---
    print("[INFO] Initializing the model on the GPU...")
    MODEL = VisFusion(cfg)
    MODEL = torch.nn.DataParallel(MODEL, device_ids=[0])
    MODEL.cuda()

    # --- load model checkpoint ---
    if cfg.LOADCKPT != "":
        saved_models = [cfg.LOADCKPT]
    else:
        saved_models = [fn for fn in os.listdir(cfg.LOGDIR) if fn.endswith(".ckpt")]
        saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    chkpt = os.path.join(cfg.LOGDIR, saved_models[-1])

    print(f"[INFO] Resuming from {chkpt}")
    STATE_DICT = torch.load(chkpt)
    MODEL.load_state_dict(STATE_DICT['model'], strict=False)

    # --- start recon model ---
    recon_model = VisFusionReconstructionModel(MODEL_NAME, SERVER_URL)
    asyncio.run(recon_model.connect())

