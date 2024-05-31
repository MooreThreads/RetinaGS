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
import torch
from torch.utils.data import DataLoader
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer.render import render
import sys
from scene import SimpleScene
from scene.gaussian_nn_module import GaussianModel2
from utils.general_utils import safe_state
import uuid, math
from tqdm import tqdm
from utils.image_utils import psnr
from lpipsPyTorch import LPIPS
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.datasets import CameraListDataset, DatasetRepeater, GroupedItems, PatchListDataset
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

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

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel2(dataset.sh_degree, range_low=[-1000]*3, range_up=[1000]*3)
    scene = SimpleScene(dataset, load_iteration=-1) # set load_iteration=-1 to enable search .ply
    scene.load2gaussians(gaussians)
    gaussians.training_setup(opt)
    print('spatial_lr_scale is {}'.format(gaussians.spatial_lr_scale))
    if opt.perception_loss:
        CNN_IMAGE = LPIPS(net_type=opt.perception_net_type, 
                          version=opt.perception_net_version).to('cuda')
        print('lpips:{}.{}'.format(opt.perception_net_type, opt.perception_net_version))
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    _train_dataset, _test_dataset = scene.getTrainCameras(), scene.getTestCameras()
    train_dataset = PatchListDataset(_train_dataset, h_division=H_DIV, w_division=W_DIV)
    test_dataset = PatchListDataset(_test_dataset, h_division=1, w_division=1)

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    NUM_EPOCH = (opt.iterations - first_iter + len(train_dataset) - 1) // len(train_dataset)
    first_iter += 1 # origin code add 1 to avoid corner cases on iteration 0, we just follow it

    iteration = first_iter
    for i_epoch in range(NUM_EPOCH):
        train_loader = DataLoader(train_dataset, batch_size=1, prefetch_factor=4, shuffle=True, drop_last=False, num_workers=32, collate_fn=SimpleScene.get_batch)

        for data in train_loader:
            if iteration > opt.iterations:
                return
            iter_start.record()

            gaussians.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()

            # Pick a random Camera
            viewpoint_cam = data[0]

            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True

            bg = torch.rand((3), device="cuda") if opt.random_background else background

            render_pkg = render_wrapper(viewpoint_cam, gaussians, pipe, bg)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            if not opt.perception_loss:
                loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            else:
                loss = (1.0 - opt.lambda_dssim) * Ll1 \
                    + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) \
                    + opt.lambda_perception * torch.sum(CNN_IMAGE(image, gt_image))
            loss.backward()

            iter_end.record()

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                # Log and save
                training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, gaussians, render_wrapper, (pipe, background))
                if (iteration in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration, gaussians=gaussians)

                # Densification
                if iteration < opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

                    tb_writer.add_scalar('total_points', gaussians.get_xyz.shape[0], iteration)
                    
                # Optimizer step
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

                if (iteration in checkpoint_iterations):
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            iteration += 1

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : SimpleScene, gaussians, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        # torch.cuda.empty_cache()
        _train_dataset, _test_dataset = scene.getTrainCameras(), scene.getTestCameras()
        train_dataset = PatchListDataset(_train_dataset, h_division=H_DIV, w_division=W_DIV)
        test_dataset = PatchListDataset(_test_dataset, h_division=1, w_division=1)
        
        validation_configs = ({'name': 'test', 'cameras' : test_dataset}, 
                              {'name': 'train', 'cameras' : [train_dataset[idx] for idx in range(0, len(train_dataset), 8)]})       
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', gaussians.get_xyz.shape[0], iteration)
        # torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)
    args.checkpoint_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
