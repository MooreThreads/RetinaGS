#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch, torchvision
from torch.utils.data import DataLoader, Dataset
from random import randint
from gaussian_renderer.render import render
import sys
from scene import SimpleScene
from scene.gaussian_nn_module import GaussianModel2
from utils.general_utils import safe_state
import uuid, math
from tqdm import tqdm
from lpipsPyTorch import LPIPS
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from utils.datasets import CameraListDataset, DatasetRepeater, GroupedItems, PatchListDataset
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from os import makedirs
from scene.cameras import Camera, EmptyCamera, Patch
import gaussian_renderer.pytorch_gs_render.pytorch_render as pytorch_render

H_DIV = 2
W_DIV = 2

def create_batch(patches:list):
    # assume all patch has the same complete shape
    batch_data, batch_size = {}, len(patches)
    example_p:Patch = patches[0]
    H, W = example_p.complete_height, example_p.complete_width

    batch_data['packed_views'] = torch.stack([p.pack_up() for p in patches])
    max_tile_num = max([len(p.all_tiles) for p in patches]) 
    batch_data['tile_maps'] = torch.zeros((batch_size, max_tile_num), dtype=torch.int, device='cuda')
    batch_data['tile_nums'] = []
    for i in range(batch_size):
        p:Patch = patches[i]
        num_valid_tile = len(p.all_tiles)
        batch_data['tile_maps'][i, :num_valid_tile] = p.all_tiles
        batch_data['tile_nums'].append(num_valid_tile)
    batch_data['tile_maps_sizes'] = torch.tensor([math.ceil(W/16), math.ceil(H/16)], device='cuda').view(-1, 2).repeat(batch_size, 1)    
    batch_data['tile_maps_sizes_list'] = [math.ceil(W/16), math.ceil(H/16)]
    batch_data['image_size_list'] = [math.ceil(W), math.ceil(H)]
    batch_data['patches_cpu'] = patches  # it's okay to use patch on gpu 
    return batch_data

def render_wrapper(viewpoint_cam, gaussians, pipe, background):
    batch = create_batch([viewpoint_cam.to_device('cuda')])
    rets = pytorch_render.render(gaussians, batch)
    return rets[0]

torch.set_grad_enabled(False)

import time
times = []

def render_set(model_path, name, iteration, views:Dataset, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    dataloader = DataLoader(views, batch_size=1, prefetch_factor=4, shuffle=False, drop_last=False, num_workers=32, collate_fn=SimpleScene.get_batch)

    for idx, batch in enumerate(tqdm(dataloader, desc="Rendering progress")):
        view:Patch = batch[0]

        ret = render_wrapper(view, gaussians, None, None)
        rendering = ret["render"]

        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel2(dataset.sh_degree, range_low=[-1000]*3, range_up=[1000]*3)
        scene = SimpleScene(dataset, load_iteration=-1) # set load_iteration=-1 to enable search .ply
        gaussians.load_ply(os.path.join(scene.model_path, "point_cloud", "iteration_" + str(iteration), "point_cloud.ply"))

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        _train_dataset, _test_dataset = scene.getTrainCameras(), scene.getTestCameras()
        train_dataset = PatchListDataset(_train_dataset, h_division=H_DIV, w_division=W_DIV)
        test_dataset = PatchListDataset(_test_dataset, h_division=1, w_division=1)

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, train_dataset, gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, test_dataset, gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)