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

import torch
from scene import Scene
import os, glob
from tqdm import tqdm
from os import makedirs
from gaussian_renderer.render_metric import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from gaussian_renderer import GaussianModel

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    num_pixel = 0
    num_GS = gaussians._xyz.shape[0]
    cnt_GSPP, cnt_MAPP, cnt_AT = 0, 0, 0    # Gaussian per Pixel, Mean Area per Pixel, Activated Times
    AGS_mask = torch.zeros((num_GS,1), dtype=torch.bool, device='cuda') # Activated Mask of Gaussian
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        
        render_pkg = render(view, gaussians, pipeline, background)
        rendering, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        depth, alpha = render_pkg["depth"], render_pkg["alpha"]

        cnt_GS, cnt_MA = render_pkg["cnt1"], render_pkg["cnt2"]
        num_pixel += (rendering.numel()/3)
        cnt_GSPP += cnt_GS.sum().item()
        cnt_MAPP += cnt_MA.sum().item()
        cnt_AT += cnt_GS.sum().item()

        # find activated gs by check grad
        gt:torch.Tensor = view.original_image[0:3, :, :].cuda()
        loss = torch.abs(gt - rendering).sum()
        loss.backward()
        with torch.no_grad():
            flag = torch.logical_or(
                gaussians._opacity.grad != 0,
                torch.abs(gaussians._scaling.grad).sum(dim=-1, keepdim=True) != 0
            )
            AGS_mask[flag] = 1
            gaussians.optimizer.zero_grad(set_to_none = True)

        if idx <= 10:
            pass
            # torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

    if num_pixel > 0:
        print("GSPP {}, MAPP {}, AGSR {}, AT {}".format(
             cnt_GSPP/num_pixel, cnt_MAPP/num_pixel, AGS_mask.sum()/num_GS, cnt_GSPP/num_GS
             ))
    else:
        print('empty!')    


def render_sets(dataset : ModelParams, opt, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.training_setup(opt)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    print('enable auto_grad to count the activated times of Gaussians')
    if not skip_train:
        print("train")
        render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

    if not skip_test:
        print('test')
        render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), op.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)