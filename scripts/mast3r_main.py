import argparse
import asyncio
import os
import pathlib
import sys
import time
import trimesh

import cv2
import lietorch
import numpy as np
import torch
import torch.multiprocessing as mp
import tqdm
import yaml

import models.MAST3R.mast3r_slam.evaluate as eval
from utils.myutils import ModelResult
from models.MAST3R.mast3r_slam.config import config, load_config, set_global_config
from models.MAST3R.mast3r_slam.dataloader import Intrinsics, load_dataset
from models.MAST3R.mast3r_slam.frame import Mode, SharedKeyframes, SharedStates, create_frame
from models.MAST3R.mast3r_slam.global_opt import FactorGraph
from models.MAST3R.mast3r_slam.mast3r_utils import (
    load_mast3r,
    load_retriever,
    mast3r_inference_mono,
)
from models.MAST3R.mast3r_slam.multiprocess_utils import new_queue, try_get_msg
from models.MAST3R.mast3r_slam.tracker import FrameTracker
from models.MAST3R.mast3r_slam.visualization import WindowMsg, run_visualization
from models.base_model import BaseReconstructionModel

class MAST3RReconstructionModel(BaseReconstructionModel):
    """
    MAST3RReconstructionModel is a subclass  of BaseRecnstructionModel
    It is designed to handle reconstuction of fragments from a WebSocket connection.
    """

    def __init__(self, model_name, server_url, mast3r_model, device: str = "cuda:0") -> None:
        super().__init__(model_name, server_url)
        
        self.frame_idx = 0
        self.device = device
        self.mast3r_model = mast3r_model
        self.manager = mp.Manager()
        self.keyframes = SharedKeyframes(self.manager, -1, -1) #TODO Change hardcoded h, w
        self.states = SharedStates(self.manager, -1, -1) #TODO Change hardcoded h, w
        self.tracker = FrameTracker(self.mast3r_model, self.keyframes, self.device)
        self.backend = mp.Process(
            target=run_backend,
            args=(config, self.mast3r_model, self.states, self.keyframes, None)
        )
        
        self.backend.start()

    async def handle_fragment(self, fragment: dict) -> None:
        tmp_img_dir = pathlib.Path("/tmp/tmp_images")
        tmp_img_dir.mkdir(parents=True, exist_ok=True)

        [f.unlink() for f in tmp_img_dir.glob("*.jpg")]
        for idx, img_bytes in enumerate(fragment['images']):
            img_path = tmp_img_dir / f"image_{idx:04d}.jpg"
            with open(img_path, 'wb') as f:
                f.write(img_bytes)
        
        glb_bytes: bytes = self.recon_scene_batched(str(tmp_img_dir))

        result: ModelResult = ModelResult(fragment['scene_name'], glb_bytes, True)
        await self.send_result(result)

    def recon_scene_batched(self, img_dir: str) -> bytes:
        dataset = load_dataset(img_dir)
        
        for _ in range(len(dataset)):
            mode = self.states.get_mode()

            _, img = dataset[self.frame_idx]
            T_WC = (
                lietorch.Sim3.Identity(1, device=self.device)
                if self.frame_idx == 0
                else self.states.get_frame().T_WC
            )
            frame = create_frame(self.frame_idx, img, T_WC, img_size=dataset.img_size, device=self.device)

            if mode == Mode.INIT:
                # Initialize via mono inference, and encoded features neeed for database
                X_init, C_init = mast3r_inference_mono(model, frame)
                frame.update_pointmap(X_init, C_init)
                self.keyframes.append(frame)
                self.states.queue_global_optimization(len(self.keyframes) - 1)
                self.states.set_mode(Mode.TRACKING)
                self.states.set_frame(frame)
                self.frame_idx += 1
                continue
            if mode == Mode.TRACKING:
                add_new_kf, match_info, try_reloc = self.tracker.track(frame)
                if try_reloc:
                    self.states.set_mode(Mode.RELOC)
                self.states.set_frame(frame)
            elif mode == Mode.RELOC:
                X, C = mast3r_inference_mono(model, frame)
                frame.update_pointmap(X, C)
                self.states.set_frame(frame)
                self.states.queue_reloc()
                # In single threaded mode, make sure relocalization happen for every frame
                while config["single_thread"]:
                    with self.states.lock:
                        if self.states.reloc_sem.value == 0:
                            break
                    time.sleep(0.01)
            else:
                raise Exception("Invalid mode")
            
            if add_new_kf:
                self.keyframes.append(frame)
                self.states.queue_global_optimization(len(self.keyframes) - 1)
                # In single threaded mode, wait for the backend to finish
                while config["single_thread"]:
                    with self.states.lock:
                        if len(self.states.global_optimizer_tasks) == 0:
                            break
                    time.sleep(0.01)         
            self.frame_idx += 1

        glb_bytes: bytes = self.construct_glb(self.keyframes)
        return glb_bytes

    def construct_glb(self, keyframes, conf_threshold: float) -> bytes:
        # Get Pcds and Colors from keyframe
        pcd_list, color_list = [], []
        for i in range(len(keyframes)):
            keyframe = keyframes[i]
            pW = keyframe.T_WC.act(keyframe.X_canon).cpu().numpy().reshape(-1, 3)
            color = (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8).reshape(-1, 3)
            valid = (
                keyframe.get_average_conf().cpu().numpy().astype(np.float32).reshape(-1)
                > conf_threshold
            )
            pcd_list.append(pW[valid])
            color_list.append(color[valid])
        pointclouds = np.concatenate(pcd_list, axis=0)
        colors = np.concatenate(color_list, axis=0)

        # Construct glb from Pcds and Colors
        scene = trimesh.Scene()
        scene.add_geometry(trimesh.PointCloud(vertices=pointclouds, colors=colors/255.))

        return scene.export(file_type='glb')    


def relocalization(frame, keyframes, factor_graph, retrieval_database):
    # we are adding and then removing from the keyframe, so we need to be careful.
    # The lock slows viz down but safer this way...
    with keyframes.lock:
        kf_idx = []
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=False,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds
        successful_loop_closure = False
        if kf_idx:
            keyframes.append(frame)
            n_kf = len(keyframes)
            kf_idx = list(kf_idx)  # convert to list
            frame_idx = [n_kf - 1] * len(kf_idx)
            print("RELOCALIZING against kf ", n_kf - 1, " and ", kf_idx)
            if factor_graph.add_factors(
                frame_idx,
                kf_idx,
                config["reloc"]["min_match_frac"],
                is_reloc=config["reloc"]["strict"],
            ):
                retrieval_database.update(
                    frame,
                    add_after_query=True,
                    k=config["retrieval"]["k"],
                    min_thresh=config["retrieval"]["min_thresh"],
                )
                print("Success! Relocalized")
                successful_loop_closure = True
                keyframes.T_WC[n_kf - 1] = keyframes.T_WC[kf_idx[0]].clone()
            else:
                keyframes.pop_last()
                print("Failed to relocalize")

        if successful_loop_closure:
            if config["use_calib"]:
                factor_graph.solve_GN_calib()
            else:
                factor_graph.solve_GN_rays()
        return successful_loop_closure

def run_backend(cfg, model, states, keyframes, K):
    set_global_config(cfg)

    device = keyframes.device
    factor_graph = FactorGraph(model, keyframes, K, device)
    retrieval_database = load_retriever(model)

    mode = states.get_mode()
    while mode is not Mode.TERMINATED:
        mode = states.get_mode()
        if mode == Mode.INIT or states.is_paused():
            time.sleep(0.01)
            continue
        if mode == Mode.RELOC:
            frame = states.get_frame()
            success = relocalization(frame, keyframes, factor_graph, retrieval_database)
            if success:
                states.set_mode(Mode.TRACKING)
            states.dequeue_reloc()
            continue
        idx = -1
        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks[0]
        if idx == -1:
            time.sleep(0.01)
            continue

        # Graph Construction
        kf_idx = []
        # k to previous consecutive keyframes
        n_consec = 1
        for j in range(min(n_consec, idx)):
            kf_idx.append(idx - 1 - j)
        frame = keyframes[idx]
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=True,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds

        lc_inds = set(retrieval_inds)
        lc_inds.discard(idx - 1)
        if len(lc_inds) > 0:
            print("Database retrieval", idx, ": ", lc_inds)

        kf_idx = set(kf_idx)  # Remove duplicates by using set
        kf_idx.discard(idx)  # Remove current kf idx if included
        kf_idx = list(kf_idx)  # convert to list
        frame_idx = [idx] * len(kf_idx)
        if kf_idx:
            factor_graph.add_factors(
                kf_idx, frame_idx, config["local_opt"]["min_match_frac"]
            )

        with states.lock:
            states.edges_ii[:] = factor_graph.ii.cpu().tolist()
            states.edges_jj[:] = factor_graph.jj.cpu().tolist()

        if config["use_calib"]:
            factor_graph.solve_GN_calib()
        else:
            factor_graph.solve_GN_rays()

        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks.pop(0)

MODEL_NAME = os.getenv("MODEL_NAME", "mast3r")
SERVER_URL = os.getenv("SERVER_URL", "ws://router:5000/ws/model")
if __name__ == "__main__":
    mp.set_start_method("spawn")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    device = "cuda:0"
    save_frames = False

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/base.yaml")
    args = parser.parse_args()

    load_config(args.config)
    
    mast3r_model = load_mast3r()
    mast3r_model.share_memory()

    model = MAST3RReconstructionModel(MODEL_NAME, SERVER_URL, mast3r_model)
    asyncio.run(model.connect())