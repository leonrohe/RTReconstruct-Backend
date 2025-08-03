import asyncio
import os
from pathlib import Path
import torch
from utils.myutils import ModelResult
import numpy as np

from models.base_model import BaseReconstructionModel
from models.SLAM3R.recon import get_img_tokens, initialize_scene, i2p_inference_batch, l2w_inference, normalize_views, scene_frame_retrieve
from models.SLAM3R.slam3r.datasets.wild_seq import Seq_Data
from models.SLAM3R.slam3r.models import Local2WorldModel, Image2PointsModel
from models.SLAM3R.slam3r.utils.recon_utils import *

class SLAM3RScene:
    batch_size = 5

    def __init__(self):
        self.batch_idx = 0
        
        self.input_views = []
        self.registered_confs_mean = []
        self.buffering_set_ids = []

        self.per_frame_res = dict(i2p_pcds=[], i2p_confs=[], l2w_pcds=[], l2w_confs=[], rgb_imgs=[])
    
    def get_global_idx(self, local_idx):
        return self.batch_idx * self.batch_size + local_idx

class SLAM3RReconstructModel(BaseReconstructionModel):
    """
    SLAM3RReconstructModel is a subclass of BaseReconstructionModel.
    It is designed to handle the reconstruction of fragments from a WebSocket connection.
    """


    def __init__(self,
                 model_name: str,
                 server_url: str,
                 i2p_model: Image2PointsModel,
                 l2w_model: Local2WorldModel):
        super().__init__(model_name, server_url)
        self.i2p_model = i2p_model
        self.l2w_model = l2w_model

        self.scenes = {}

    async def handle_fragment(self, fragment: dict):
        scene: SLAM3RScene = self.scenes.setdefault(fragment['scene_name'], SLAM3RScene())

        tmp_img_dir = Path("/tmp/tmp_images")
        tmp_img_dir.mkdir(parents=True, exist_ok=True)

        # Clear the directory first
        for f in tmp_img_dir.glob("*.jpg"):
            f.unlink()

        # Save images
        for idx, img_bytes in enumerate(fragment['images']):
            img_path = tmp_img_dir / f"image_{idx:04d}.jpg"
            with open(img_path, 'wb') as f:
                f.write(img_bytes)

        # Call your processing function
        (glb_bytes, _) = self.recon_scene_batched(
                            scene,
                            self.i2p_model,
                            self.l2w_model,
                            str(tmp_img_dir))
                            
        result: ModelResult = ModelResult(fragment['scene_name'], glb_bytes, True) 
        result.SetTranslation(0.09, 0.111, -0.463)
        result.SetRotation(35.685, -4.701, -2.377, degrees=True)
        result.SetScale(2, 2, 2)

        await self.send_result(result)

    def recon_scene_batched(self,
                            scene: SLAM3RScene,
                            i2p_model: Image2PointsModel,
                            l2w_model: Local2WorldModel,
                            imgs: list,

                            device: str = 'cuda',
                            win_r: int = 5,
                            conf_thres_i2p: float = 1.5,
                            num_scene_frame: int = 10,
                            update_buffer_intv: int = 1,
                            buffer_size: int = 100,
                            conf_thres_l2w: int = 12):
        np.random.seed(42)

        dataset = Seq_Data(imgs, to_tensor=True)
        data_views = dataset[0][:] # copy the first group of views, in this case all images

        assert len(data_views) == scene.batch_size, \
            f"the number of views in the batch should be {scene.batch_size}, but got {len(data_views)}"

        # Pre-save the RGB images along with their corresponding masks 
        # in preparation for visualization at last.
        rgb_imgs = []
        for i in range(scene.batch_size):
            if data_views[i]['img'].shape[0] == 1:
                data_views[i]['img'] = data_views[i]['img'][0]        
            rgb_imgs.append(transform_img(dict(img=data_views[i]['img'][None]))[...,::-1])

        for view_id in data_views:
            view_id['img'] = torch.tensor(view_id['img'][None]) # add a batch dimension
            view_id['true_shape'] = torch.tensor(view_id['true_shape'][None]) # add a batch dimension

            # remove unnecessary keys to save memory
            for key in ['valid_mask', 'pts3d_cam', 'pts3d']:
                if key in view_id:
                    del view_id[key]

            # move data to the device
            to_device(view_id, device=device)
        
        # pre-extract img tokens by encoder, which can be reused 
        # in the following inference by both i2p and l2w models
        res_shapes, res_feats, res_poses = get_img_tokens(data_views, i2p_model)
        print('finish pre-extracting img tokens')

        # re-organize input views for the following inference.
        # Keep necessary attributes only.
        for i in range(scene.batch_size):
            scene.input_views.append(dict(label=data_views[i]['label'],
                                    img_tokens=res_feats[i], 
                                    true_shape=data_views[i]['true_shape'], 
                                    img_pos=res_poses[i]))
        for key in scene.per_frame_res:
            scene.per_frame_res[key].extend([None for _ in range(scene.batch_size)])
        scene.registered_confs_mean.extend([None for _ in range(scene.batch_size)])
        
        if scene.batch_idx == 0:
            # initialize the scene with the first several frames
            initial_pcds, initial_confs, init_ref_id = initialize_scene(scene.input_views[:scene.batch_size], 
                                                        i2p_model, 
                                                        winsize=scene.batch_size,
                                                        return_ref_id=True)
            
            for i in range(scene.batch_size):
                scene.per_frame_res['l2w_confs'][i] = initial_confs[i][0].to(device)  # 224,224
                scene.registered_confs_mean[i] = scene.per_frame_res['l2w_confs'][i].mean().cpu()
            
            assert buffer_size <= 0 or buffer_size >= scene.batch_size
            scene.buffering_set_ids = [i for i in range(scene.batch_size)]

            for i in range(scene.batch_size):
                scene.input_views[i]['pts3d_world'] = initial_pcds[i]
            
            initial_valid_masks = [conf > conf_thres_i2p for conf in initial_confs] # 1,224,224
            normed_pts = normalize_views([view['pts3d_world'] for view in scene.input_views[:scene.batch_size]],
                                                        initial_valid_masks)
            
            for i in range(scene.batch_size):
                scene.input_views[i]['pts3d_world'] = normed_pts[i]
                # filter out points with low confidence
                scene.input_views[i]['pts3d_world'][~initial_valid_masks[i]] = 0       
                scene.per_frame_res['l2w_pcds'][i] = normed_pts[i]  # 224,224,3
        
        # recover the pointmap of each view in their local coordinates with the I2P model
        local_confs_mean = []
        # TODO: range(scene.batch_size) -> range(num_views) falls fehler
        for i in range(scene.batch_size):
            view_id = scene.get_global_idx(i)

            # skip the views in the initial window
            if view_id in scene.buffering_set_ids:
                # trick to mark the keyframe in the initial window
                if view_id == init_ref_id:
                    scene.per_frame_res['i2p_pcds'][view_id] = scene.per_frame_res['l2w_pcds'][view_id].cpu()
                else:
                    scene.per_frame_res['i2p_pcds'][view_id] = torch.zeros_like(scene.per_frame_res['l2w_pcds'][view_id], device="cpu")
                scene.per_frame_res['i2p_confs'][view_id] = scene.per_frame_res['l2w_confs'][view_id].cpu()
                continue

            sel_ids = [view_id]
            for j in range(1, win_r+1):
                if view_id-j >= 0:
                    sel_ids.append(view_id-j)
                if view_id+j < scene.batch_size:
                    sel_ids.append(view_id+j)
            local_views = [scene.input_views[id] for id in sel_ids]
            ref_id = 0
            # recover points in the local window, and save the keyframe points and confs
            output = i2p_inference_batch([local_views], i2p_model, ref_id=ref_id, 
                                        tocpu=False, unsqueeze=False)['preds']
            
            #save results of the i2p model
            scene.per_frame_res['i2p_pcds'][view_id] = output[ref_id]['pts3d'].cpu() # 1,224,224,3
            scene.per_frame_res['i2p_confs'][view_id] = output[ref_id]['conf'][0].cpu() # 224,224

            # construct the input for L2W model
            scene.input_views[view_id]['pts3d_cam'] = output[ref_id]['pts3d'] # 1,224,224,3
            valid_mask = output[ref_id]['conf'] > conf_thres_i2p # 1,224,224
            scene.input_views[view_id]['pts3d_cam'] = normalize_views([scene.input_views[view_id]['pts3d_cam']],
                                                        [valid_mask])[0]
            scene.input_views[view_id]['pts3d_cam'][~valid_mask] = 0

        local_confs_mean = [conf.mean() for conf in scene.per_frame_res['i2p_confs']] # 224,224
        print(f'finish recovering pcds of {len(local_confs_mean)} frames in their local coordinates, with a mean confidence of {torch.stack(local_confs_mean).mean():.2f}')

        next_register_id = max(scene.batch_size, scene.get_global_idx(0))
        milestone = next_register_id # All frames before milestone have undergone the selection process for entry into the buffering set.
        num_register = 2   # how many frames to register in each round
        max_buffer_size = buffer_size

        pbar = tqdm(total=len(scene.input_views), desc="registering")
        pbar.update(next_register_id-1)

        del i
        while next_register_id < len(scene.input_views):
            ni = next_register_id
            max_id = min(ni+num_register, len(scene.input_views))-1  # the last frame to be registered in this round

            cand_ref_ids = scene.buffering_set_ids
            ref_views, sel_pool_ids = scene_frame_retrieve(
                [scene.input_views[i] for i in cand_ref_ids], 
                scene.input_views[ni:ni+num_register:2], 
                i2p_model, sel_num=num_scene_frame, 
                # cand_recon_confs=[per_frame_res['l2w_confs'][i] for i in cand_ref_ids],
                depth=2)
            
            # register the source frames in the local coordinates to the world coordinates with L2W model
            l2w_input_views = ref_views + scene.input_views[ni:max_id+1]
            input_view_num = len(ref_views) + max_id - ni + 1
            assert input_view_num == len(l2w_input_views)

            output = l2w_inference(l2w_input_views, l2w_model, 
                            ref_ids=list(range(len(ref_views))), 
                            device=device,
                            normalize=False)
            
            # process the output of L2W model
            src_ids_local = [len(ref_views)+id for id in range(max_id-ni+1)]  # the ids of src views in the local window
            src_ids_global = [id for id in range(ni, max_id+1)]    #the ids of src views in the whole dataset
            succ_num = 0
            for id in range(len(src_ids_global)):
                output_id = src_ids_local[id] # the id of the output in the output list
                view_id = src_ids_global[id]    # the id of the view in all views
                conf_map = output[output_id]['conf'] # 1,224,224
                scene.input_views[view_id]['pts3d_world'] = output[output_id]['pts3d_in_other_view'] # 1,224,224,3
                scene.per_frame_res['l2w_confs'][view_id] = conf_map[0]
                scene.registered_confs_mean[view_id] = conf_map[0].mean().cpu()
                scene.per_frame_res['l2w_pcds'][view_id] = scene.input_views[view_id]['pts3d_world']
                succ_num += 1

            next_register_id += succ_num
            pbar.update(succ_num) 

            if next_register_id - milestone >= update_buffer_intv:
                while(next_register_id - milestone >= 1):
                    full_flag = max_buffer_size > 0 and len(scene.buffering_set_ids) >= max_buffer_size

                    # Use offest to ensure the selected view is not too close to the last selected view
                    # If the last selected view is 0, 
                    # the next selected view should be at least kf_stride*3//4 frames away
                    start_ids_offset = max(0, scene.buffering_set_ids[-1]+3//4 - milestone)
                        
                    # get the mean confidence of the candidate views
                    mean_cand_recon_confs = torch.stack([scene.registered_confs_mean[i]
                                            for i in range(milestone+start_ids_offset, milestone+1)])
                    mean_cand_local_confs = torch.stack([local_confs_mean[i]
                                            for i in range(milestone+start_ids_offset, milestone+1)])
                    # normalize the confidence to [0,1], to avoid overconfidence
                    mean_cand_recon_confs = (mean_cand_recon_confs - 1)/mean_cand_recon_confs # transform to sigmoid
                    mean_cand_local_confs = (mean_cand_local_confs - 1)/mean_cand_local_confs
                    # the final confidence is the product of the two kinds of confidences
                    mean_cand_confs = mean_cand_recon_confs*mean_cand_local_confs
                    
                    most_conf_id = mean_cand_confs.argmax().item()
                    most_conf_id += start_ids_offset
                    id_to_buffer = milestone + most_conf_id
                    scene.buffering_set_ids.append(id_to_buffer)
                
                    # since we have inserted a new frame, overflow must happen when full_flag is True
                    if full_flag:
                        scene.buffering_set_ids.pop(0)
                    
                    milestone += 1 
            # transfer the data to cpu if it is not in the buffering set, to save gpu memory
            for i in range(next_register_id):
                to_device(scene.input_views[i], device=device if i in scene.buffering_set_ids else 'cpu')

        pbar.close()

        fail_view = {}
        for i,conf in enumerate(scene.registered_confs_mean):
            if conf < 10:
                fail_view[i] = conf.item()
        print(f'mean confidence for whole scene reconstruction: {torch.tensor(scene.registered_confs_mean).mean().item():.2f}')
        print(f"{len(fail_view)} views with low confidence: ", {key:round(fail_view[key],2) for key in fail_view.keys()})

        for i in range(len(rgb_imgs)):
            idx = scene.get_global_idx(i)
            scene.per_frame_res['rgb_imgs'][idx] = rgb_imgs[i]

        glb_bytes = self.get_model_from_scene(per_frame_res=scene.per_frame_res, 
                                              save_dir="/tmp/tmp_results", 
                                              num_points_save=500000, 
                                              conf_thres_res=conf_thres_l2w)

        scene.batch_idx += 1

        return glb_bytes, scene.per_frame_res

    def get_model_from_scene(self, 
                             per_frame_res, save_dir, 
                             num_points_save=200000, 
                             conf_thres_res=3, 
                             valid_masks=None):  
        # collect the registered point clouds and rgb colors
        pcds = []
        rgbs = []
        pred_frame_num = len(per_frame_res['l2w_pcds'])
        registered_confs = per_frame_res['l2w_confs']   
        registered_pcds = per_frame_res['l2w_pcds']
        rgb_imgs = per_frame_res['rgb_imgs']
        for i in range(pred_frame_num):
            registered_pcd = to_numpy(registered_pcds[i])
            if registered_pcd.shape[0] == 3:
                registered_pcd = registered_pcd.transpose(1,2,0)
            registered_pcd = registered_pcd.reshape(-1,3)
            rgb = rgb_imgs[i].reshape(-1,3)
            pcds.append(registered_pcd)
            rgbs.append(rgb)
            
        res_pcds = np.concatenate(pcds, axis=0)
        res_rgbs = np.concatenate(rgbs, axis=0)
        
        pts_count = len(res_pcds)
        valid_ids = np.arange(pts_count)
        
        # filter out points with gt valid masks
        if valid_masks is not None:
            valid_masks = np.stack(valid_masks, axis=0).reshape(-1)
            # print('filter out ratio of points by gt valid masks:', 1.-valid_masks.astype(float).mean())
        else:
            valid_masks = np.ones(pts_count, dtype=bool)
        
        # filter out points with low confidence
        if registered_confs is not None:
            conf_masks = []
            for i in range(len(registered_confs)):
                conf = registered_confs[i]
                conf_mask = (conf > conf_thres_res).reshape(-1).cpu() 
                conf_masks.append(conf_mask)
            conf_masks = np.array(torch.cat(conf_masks))
            valid_ids = valid_ids[conf_masks&valid_masks]
            print('ratio of points filered out: {:.2f}%'.format((1.-len(valid_ids)/pts_count)*100))
        
        # sample from the resulting pcd consisting of all frames
        n_samples = min(num_points_save, len(valid_ids))
        print(f"resampling {n_samples} points from {len(valid_ids)} points")
        sampled_idx = np.random.choice(valid_ids, n_samples, replace=False)
        sampled_pts = res_pcds[sampled_idx]
        sampled_rgbs = res_rgbs[sampled_idx]
        sampled_pts[:, :2] *= -1 # flip the x,y axis for better visualization
        
        Path(save_dir).mkdir(exist_ok=True)

        save_name = f"recon.glb"
        scene = trimesh.Scene()
        scene.add_geometry(trimesh.PointCloud(vertices=sampled_pts, colors=sampled_rgbs/255.))
        save_path = join(save_dir, save_name)
        scene.export(save_path)

        glb_bytes = scene.export(file_type='glb')

        return glb_bytes

MODEL_NAME = os.getenv("MODEL_NAME", "slam3r")
SERVER_URL = os.getenv("SERVER_URL", "ws://router:5000/ws/model")

if __name__ == "__main__":

    print("Loading SLAM3R I2P Model ...")
    i2p_model = Image2PointsModel.from_pretrained('siyan824/slam3r_i2p')
    i2p_model.to('cuda')
    i2p_model.eval()
    print("Loading SLAM3R L2W Model ...")
    l2w_model = Local2WorldModel.from_pretrained('siyan824/slam3r_l2w')
    l2w_model.to('cuda')
    l2w_model.eval()

    
    model = SLAM3RReconstructModel(MODEL_NAME, SERVER_URL, i2p_model, l2w_model)
    asyncio.run(model.connect())