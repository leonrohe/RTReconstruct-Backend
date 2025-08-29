import argparse
import asyncio
import os
import time
import datetime
import traceback
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from tensorboardX import SummaryWriter
from loguru import logger
import numpy as np

from models.VisFusion.utils import tensor2float, save_scalars, DictAverageMeter, SaveScene, make_nograd_func
from models.VisFusion.datasets import transforms, find_dataset_def
from models.VisFusion.models import VisFusion
from models.VisFusion.config import cfg, update_config
from models.VisFusion.datasets.sampler import DistributedSampler
from models.VisFusion.ops.comm import *

from models.base_model import BaseReconstructionModel

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

torch.manual_seed(cfg.SEED)
torch.cuda.manual_seed(cfg.SEED)

# Augmentation
n_views = cfg.TEST.N_VIEWS
random_rotation = False
random_translation = False
paddingXY = 0
paddingZ = 0

transform = [transforms.ResizeImage((640, 480)),
             transforms.ToTensor(),
             transforms.RandomTransformSpace(
                  cfg.MODEL.N_VOX, cfg.MODEL.VOXEL_SIZE, random_rotation, random_translation,
                  paddingXY, paddingZ, max_epoch=cfg.TRAIN.EPOCHS),
             transforms.IntrinsicsPoseToProjection(n_views, 4)]
transforms = transforms.Compose(transform)

Dataset = find_dataset_def(cfg.DATASET)
test_dataset = Dataset(cfg.TEST.PATH, "test", transforms, cfg.TEST.N_VIEWS, len(cfg.MODEL.THRESHOLDS) - 1, cfg.SCENE)
TestImgLoader = DataLoader(test_dataset, cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.TEST.N_WORKERS, drop_last=False)


def test(from_latest=False):
    ckpt_list = []

    if cfg.LOADCKPT != '':
        saved_models = [cfg.LOADCKPT]
    else:
        saved_models = [fn for fn in os.listdir(cfg.LOGDIR) if fn.endswith(".ckpt")]
        saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        if from_latest:
            saved_models = saved_models[-1:]

    for ckpt in saved_models:
        if ckpt not in ckpt_list:
            # use the latest checkpoint file
            loadckpt = os.path.join(cfg.LOGDIR, ckpt)
            logger.info("resuming " + str(loadckpt))
            state_dict = torch.load(loadckpt)
            model.load_state_dict(state_dict['model'], strict=False)
            epoch_idx = state_dict['epoch']

            TestImgLoader.dataset.tsdf_cashe = {}

            avg_test_scalars = DictAverageMeter()
            save_mesh_scene = SaveScene(cfg)
            batch_len = len(TestImgLoader)
            for batch_idx, sample in enumerate(TestImgLoader):
                for n in sample['fragment']:
                    logger.info(n)
                # save mesh if SAVE_SCENE_MESH and is the last fragment
                save_scene = cfg.SAVE_SCENE_MESH and batch_idx == batch_len - 1

                start_time = time.time()
                loss, scalar_outputs, outputs = test_sample(sample, save_scene)
                logger.info('Epoch {}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(epoch_idx, batch_idx,
                                                            len(TestImgLoader), loss, time.time() - start_time))
                avg_test_scalars.update(scalar_outputs)
                del scalar_outputs

                if batch_idx % 100 == 0:
                    logger.info("Iter {}/{}, test results = {}".format(batch_idx, len(TestImgLoader),
                                                                       avg_test_scalars.mean()))

                # save mesh
                if cfg.SAVE_SCENE_MESH:
                    save_mesh_scene(outputs, sample, epoch_idx)
            ckpt_list.append(ckpt)

@make_nograd_func
def test_sample(sample, save_scene=False):
    model.eval()
    outputs, loss_dict = model(sample, save_scene)
    loss = loss_dict['total_loss']

    return tensor2float(loss), tensor2float(loss_dict), outputs

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
                'extrinsics': np.stack(visfusion_fragment['extrinsics']),
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
                

        except Exception as e:
            print(f"[ERROR] Inference failed, because {e}.")
            print(traceback.format_exc())

    def transform_base_fragment(self, fragment: dict):
        pass

MODEL_NAME = os.getenv("MODEL_NAME", "neucon")
SERVER_URL = os.getenv("SERVER_URL", "ws://router:5000/ws/model")
if __name__ == "__main__":
    args = args()
    update_config(cfg, args)

    cfg.defrost()
    cfg.LOCAL_RANK = args.local_rank
    cfg.freeze()

    print("[INFO] Initializing the model on the GPU...")
    MODEL = VisFusion(cfg)
    MODEL = torch.nn.DataParallel(MODEL, device_ids=[0])
    MODEL.cuda()

    saved_models = [fn for fn in os.listdir(cfg.LOGDIR) if fn.endswith(".ckpt")]
    saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    chkpt = os.path.join(cfg.LOGDIR, saved_models[-1])

    print(f"[INFO] Resuming from {chkpt}")
    state_dict = torch.load(chkpt)
    MODEL.load_state_dict(state_dict['model'], strict=False)

    recon_model = VisFusionReconstructionModel(MODEL_NAME, SERVER_URL)
    asyncio.run(recon_model.connect())

