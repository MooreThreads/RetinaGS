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
from torch import nn

from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
import itertools, traceback

from random import randint
from utils.loss_utils import l1_loss, ssim
from utils.general_utils import is_point_in_batch, is_interval_in_batch
import sys
# from scene import Scene, GaussianModel
from scene.simple_scene_for_gsnn import SimpleScene4Gsnn
from scene.gaussian_nn_module import BoundedGaussianModel
from scene.cameras import Camera
from scene.colmap_loader import read_points3D_binary
from scene.dataset_readers import storePly
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = False
except ImportError:
    TENSORBOARD_FOUND = False
import time
import logging
from torch.profiler import profile, record_function, ProfilerActivity


def ddp_setup(rank, world_size, MASTER_ADDR, MASTER_PORT):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = str(MASTER_ADDR)
    os.environ["MASTER_PORT"] = str(MASTER_PORT)
    
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    

def render_wrapper(viewpoint_cam: Camera, gaussians_ddp, pipe, background):
    gaussians_ddp.module.clean_cached_features()
    # viewpoint_cam = viewpoint_cam.to_device('cuda')
    # call model(input) to trigger DDP mechanism
    return gaussians_ddp(viewpoint_cam, pipe, background)
    

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, args):
    cuda_device = 'cuda:{}'.format(dist.get_rank())
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    name = args.name
    first_iter = 0 # + world_size
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = BoundedGaussianModel(dataset.sh_degree, range_low=[-1000]*3, range_up=[1000]*3, max_size=int(6e7))
    scene = SimpleScene4Gsnn(dataset, load_iteration=-1) # set load_iteration=-1 to enable search .ply
    scene.load2gaussians(gaussians) # read from ply
    gaussians.training_setup(opt)
    os.makedirs(os.path.join(scene.model_path, 'profile_dp_{}'.format(name)), exist_ok=True)
    current_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) 
    logging.basicConfig(
        format='%(asctime)s-%(filename)s[line:%(lineno)d]-%(levelname)s: %(message)s',
        filemode='w',
        filename=os.path.join(scene.model_path, 'profile_dp_{}'.format(name), 'rank_dp_{}_{}.txt'.format(dist.get_rank(), world_size))
    )
    logger = logging.getLogger('rank_{}'.format(dist.get_rank()))
    logger.setLevel(logging.INFO)

    print('spatial_lr_scale is {}, active_sh_degree is {}'.format(gaussians.spatial_lr_scale, gaussians.active_sh_degree))

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=cuda_device)

    # first_iter += dist.get_rank()
    dataset_train = scene.getTrainCameras()
    len_actual = (len(dataset_train) // world_size) * world_size
    NUM_EPOCH = opt.epochs
    progress_bar = tqdm(range(NUM_EPOCH*len_actual), desc="Training progress") if dist.get_rank() == 0 else None

    logger.info('afetr build model memory_allocted {}'.format(torch.cuda.memory_allocated()/(1024**2)))
    dataloader = DataLoader(dataset_train, batch_size=1, num_workers=16, prefetch_factor=4, shuffle=False, collate_fn=SimpleScene4Gsnn.get_batch)        
    complete_train_data_list = []
    for _i, data in enumerate(dataloader):
        camera:Camera = data[0]
        camera_gpu = camera.to_device('cuda')
        complete_train_data_list.append(camera_gpu)
    train_data_list = [complete_train_data_list[i*world_size + rank] for i in range(len_actual//world_size) ]
    logger.info('load {} samples and memory_allocted {}'.format(len(train_data_list), torch.cuda.memory_allocated()/(1024**2)))
    
    iteration = rank
    gaussians.set_SHdegree(gaussians.max_sh_degree)
    gaussians_ddp = DDP(gaussians, device_ids=[dist.get_rank()], find_unused_parameters=True)
    dist.barrier()   
    torch.cuda.synchronize(); 

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(
            scene.model_path, 'profile_dp_{}'.format(name)
        )),
    ) as p:
        t0 = time.time()
        for _i_epoch in range(NUM_EPOCH):       
            for _i, data in enumerate(train_data_list):
                gaussians.update_learning_rate((iteration - rank)//world_size)

                with record_function("custom_forward"):     
                    viewpoint_cam = data
                    render_pkg = render_wrapper(viewpoint_cam, gaussians_ddp, pipe, background)
                    image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                    torch.cuda.synchronize()

                # Loss
                with record_function("custom_backward"):     
                    gt_image = viewpoint_cam.original_image.to(cuda_device)
                    Ll1 = l1_loss(image, gt_image)
                    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
                    loss.backward()
                    with torch.no_grad():
                        # Progress bar
                        if progress_bar is not None:
                            progress_bar.update(world_size)

                        # Optimizer 
                        # gaussians.optimizer.step()
                        # gaussians.optimizer.zero_grad(set_to_none = True)
                    iteration += world_size
                    torch.cuda.synchronize()
                p.step()
        logger.info('profile time {}'.format(time.time() - t0))

    logger.info('after training peak_memory_allocted: {}'.format(torch.cuda.max_memory_allocated()/(1024**2)))    
    table_cuda = p.key_averages().table(sort_by="cuda_time_total", row_limit=-1, max_src_column_width=200)   
    table_cpu = p.key_averages().table(sort_by="cpu_time_total", row_limit=-1, max_src_column_width=200) 
    with open(os.path.join(
        scene.model_path, 'profile_dp_{}'.format(name), 'dp{}of{}_cuda_{}.txt'.format(rank, world_size, current_time)
    ), 'w') as f:
        f.writelines(table_cuda)  

    with open(os.path.join(
        scene.model_path, 'profile_dp_{}'.format(name), 'dp{}of{}_cpu_{}.txt'.format(rank, world_size, current_time)
    ), 'w') as f:
        f.writelines(table_cpu) 


def prepare_output_and_logger(args):    
    if dist.get_rank() != 0:
        return None

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

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : SimpleScene4Gsnn, gaussians, renderFunc, renderArgs):
    if dist.get_rank() != 0:
        return None
    
    cuda_device = 'cuda:{}'.format(dist.get_rank())
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = viewpoint.original_image.to(cuda_device)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().item()
                    psnr_test += psnr(image, gt_image).mean().item()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def main(rank: int, world_size: int, MASTER_ADDR, MASTER_PORT, train_args):
    ddp_setup(rank, world_size, MASTER_ADDR, MASTER_PORT)
    training(*train_args)
    destroy_process_group()       

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=7809)
    parser.add_argument('--world_size', type=int, default=4)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000, 15_000, 20_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--name", type=str, default = 'default')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    # args.densification_interval = 104
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    MASTER_ADDR = args.ip
    MASTER_PORT = args.port
    world_size = args.world_size
    train_args = (lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args)

    # 原本处于Scene/dataset_reads.py，第一次读取时会把点云载入为ply，DDP会错，换到主进程进行
    ply_path = os.path.join(args.source_path, "sparse/0/points3D.ply")
    bin_path = os.path.join(args.source_path, "sparse/0/points3D.bin")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, happen before DDP Training")
        xyz, rgb, _ = read_points3D_binary(bin_path)
        storePly(ply_path, xyz, rgb)
        
    mp.spawn(main, args=(world_size, MASTER_ADDR, MASTER_PORT, train_args), nprocs=world_size)

    # All done
    print("\nTraining complete.")