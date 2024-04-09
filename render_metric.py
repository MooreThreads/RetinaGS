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
from scene import SimpleScene
from torch.utils.data import DataLoader
import os, glob
from tqdm import tqdm
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
from utils.image_utils import psnr
from os import makedirs
from gaussian_renderer.render_metric import render
from gaussian_renderer.render import render as render_2
from gaussian_renderer.render_half_gs import render4BoundedGaussianModel as render_3
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from scene.gaussian_model import GaussianModel
import json

def render_wrapper(viewpoint_cam, gaussians, pipe, background):
    viewpoint_cam.to_device('cuda')
    return render(viewpoint_cam, gaussians, pipe, background) 

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, save_step, full_dict, per_view_dict):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    residual_path = os.path.join(model_path, name, "ours_{}".format(iteration), "residual")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(residual_path, exist_ok=True)
    
    full_dict[name] = {}
    per_view_dict[name] = {}

    num_image = len(views)    
    num_pixel = 0
    num_GS = gaussians._xyz.shape[0]
    cnt_ssim, cnt_psnr, cnt_lpips, cnt_GSPI, cnt_GSPP, cnt_MAPP, cnt_AT = 0, 0, 0, 0, 0, 0, 0    # Gaussian per Pixel, Mean Area per Pixel, Activated Times
    
    AGS_mask = torch.zeros((num_GS,1), dtype=torch.bool, device='cuda') # Activated Mask of Gaussian
    
    train_loader = DataLoader(views, batch_size=1, prefetch_factor=4, shuffle=False, drop_last=False, num_workers=32, collate_fn=SimpleScene.get_batch)
    
    for idx, data in enumerate(tqdm(train_loader, desc="metric progress")):        
        
        # gain data        
        view = data[0]        
        render_pkg = render_wrapper(view, gaussians, pipeline, background)
        rendering = render_pkg["render"]

        # compute metric        
        gt:torch.Tensor = view.original_image[0:3, :, :].cuda()        
        cnt_GS, cnt_MA = render_pkg["cnt1"], render_pkg["cnt2"]
        psnr_image = psnr(rendering, gt).mean()
        ssim_image = ssim(rendering, gt)
        lpips_image = lpips(rendering, gt)
        
        # find activated gs by check grad
        loss = torch.abs(gt - rendering).sum()
        loss.backward()
        with torch.no_grad():
            flag = torch.logical_or(
                gaussians._opacity.grad != 0,
                torch.abs(gaussians._scaling.grad).sum(dim=-1, keepdim=True) != 0
            )
            AGS_mask[flag] = 1
            GSPI_image = flag.sum().item()
            gaussians.optimizer.zero_grad(set_to_none = True)
        
        # accmulation
        num_pixel += (rendering.numel()/3)
        cnt_GSPP += cnt_GS.sum().item()
        cnt_MAPP += cnt_MA.sum().item()
        cnt_AT += cnt_GS.sum().item()
        cnt_GSPI += GSPI_image
        cnt_ssim += ssim_image.item()
        cnt_psnr += psnr_image.item()
        cnt_lpips += lpips_image.item()
       
        
        # save and log
        if save_step != -1 and idx % save_step == 0:
            torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name + ".png"))
            residual = torch.abs(gt-rendering)
            torchvision.utils.save_image(residual, os.path.join(residual_path, view.image_name + ".png"))            
        
        per_view_dict[name][view.image_name] = {
                                                "LPIPS": lpips_image.item(),                                                                        
                                                "SSIM": ssim_image.item(),
                                                "PSNR": psnr_image.item(),
                                                "GSPI": GSPI_image,
                                                "GSPP": cnt_GS.sum().item()/(rendering.numel()/3),
                                                "MAPP": cnt_MA.sum().item()                                              
                                                }
        
    full_dict[name].update({"Mean of SSIM": cnt_ssim/num_image,
                            "Mean of PSNR": cnt_psnr/num_image,
                            "Mean of LPIPS": cnt_lpips/num_image,
                            "Mean of LPIPS": cnt_lpips/num_image,
                            "Mean of GSPI": cnt_GSPI/num_image,
                            "Mean of GSPP": cnt_GSPP/num_pixel,
                            "Mean of MAPP": cnt_MAPP/num_pixel,
                            "GS": num_GS,
                            "AGSR": AGS_mask.sum().item()/num_GS,
                            "AT": cnt_GSPP/num_GS})

    if num_pixel > 0:
       print(name + ": Mean of SSIM {}, Mean of PSNR {}, Mean of LPIPS {}, Mean of GSPI {}, Mean of GSPP {}, Mean of MAPP {}, AGSR {}, AT {}".format(
             cnt_ssim/num_image, cnt_psnr/num_image, cnt_lpips/num_image, cnt_GSPI/num_image, cnt_GSPP/num_pixel, cnt_MAPP/num_pixel, AGS_mask.sum().item()/num_GS, cnt_GSPP/num_GS
             ))
    else:
        print('empty!')    


def render_sets(dataset : ModelParams, opt, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, save_step : int):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = SimpleScene(dataset, load_iteration=iteration) # scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    scene.load2gaussians(gaussians)
    gaussians.training_setup(opt)    

    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    print('enable auto_grad to count the activated times of Gaussians')
    full_dict =  {}
    per_view_dict = {}
    if not skip_train:
        print("train")
        render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, save_step, full_dict, per_view_dict)

    if not skip_test:
        print('test')
        render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, save_step, full_dict, per_view_dict)
    
    with open(dataset.model_path + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict, fp, indent=True)
                    
    with open(dataset.model_path + "/results.json", 'w') as fp:
        json.dump(full_dict, fp, indent=True)

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
    parser.add_argument("--save_step", default=-1, type=int)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), op.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.save_step)