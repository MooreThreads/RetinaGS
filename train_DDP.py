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
from scene.gaussian_nn_module import GaussianModel2
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
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import time


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
    

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    cuda_device = 'cuda:{}'.format(dist.get_rank())
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    first_iter = 0 # + world_size
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel2(dataset.sh_degree, range_low=[-1000]*3, range_up=[1000]*3)
    scene = SimpleScene4Gsnn(dataset, load_iteration=-1) # set load_iteration=-1 to enable search .ply
    scene.load2gaussians(gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=cuda_device)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress") if dist.get_rank() == 0 else None
    # first_iter += dist.get_rank()
    gaussians_ddp = DDP(gaussians, device_ids=[dist.get_rank()], find_unused_parameters=True)
    dataset_train = scene.getTrainCameras()

    def render_wrapper(viewpoint_cam: Camera, gaussians_ddp, pipe, background):
        gaussians_ddp.module.clean_cached_features()
        viewpoint_cam = viewpoint_cam.to_device('cuda')
        # call model(input) to trigger DDP mechanism
        return gaussians_ddp(viewpoint_cam, pipe, background)
    
    def eval_render_wrapper(viewpoint_cam: Camera, gaussians, pipe, background):
        gaussians.clean_cached_features()
        viewpoint_cam = viewpoint_cam.to_device('cuda')
        # call model(input) to trigger DDP mechanism
        return gaussians(viewpoint_cam, pipe, background)

    len_actually = (len(dataset_train) // world_size) * world_size
    NUM_EPOCH = (opt.iterations + (len_actually - 1)) // len_actually
    iteration, end_train = first_iter + rank, False
    # INDEX_SHUFFLED = torch.tensor(torch.vstack([torch.randperm(len(dataset_train)) for _ in range(NUM_EPOCH)]).to('cuda'), requires_grad=False)
    # dist.broadcast(INDEX_SHUFFLED, src=0)
    sampler = DistributedSampler(dataset_train, shuffle=True)    
    dataloader = DataLoader(dataset_train, batch_size=1, num_workers=16, prefetch_factor=4, shuffle=False, collate_fn=SimpleScene4Gsnn.get_batch, sampler=sampler)        
    for _i_epoch in range(NUM_EPOCH):
        # sampler = INDEX_SHUFFLED[_i_epoch][(rank*len_actually):(len_actually + rank*len_actually)].cpu().numpy()
        sampler.set_epoch(_i_epoch)
        for _i, data in enumerate(dataloader):
            iter_start.record()
            gaussians.update_learning_rate((iteration - rank)//world_size)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if (iteration - rank) % 1000 == 0 and (iteration - rank)>=1000:
                gaussians.oneupSHdegree()

            # Render
            if (iteration - rank) == debug_from:
                pipe.debug = True

            t0 = time.time()    
            viewpoint_cam = data[0]
            render_pkg = render_wrapper(viewpoint_cam, gaussians_ddp, pipe, background)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            t1 = time.time()
            if tb_writer:
                tb_writer.add_scalar('cnt/forward', t1-t0, iteration)
                tb_writer.add_scalar('cnt/size', gaussians._xyz.shape[0], iteration)
            # Loss
            gt_image = viewpoint_cam.original_image.to(cuda_device)
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss.backward()

            # if (iteration - rank - 32)%200 == 0:
            #     print('rank {} has check sync, {}, {}, {}, {}, {}'.format(dist.get_rank(), gaussians._xyz.max(), gaussians._xyz.mean(),
            #                     viewspace_point_tensor.grad, gaussians._means2D_meta.grad.mean(),       iteration   ))

            t2 = time.time()
            if tb_writer:
                tb_writer.add_scalar('cnt/backward', t2-t1, iteration)
            iter_end.record()
            
            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if dist.get_rank() == 0:
                    if iteration % 8 == 0 and iteration > 0:
                        progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                        progress_bar.update(8)
                    if iteration >= opt.iterations:
                        progress_bar.close()

                # Log and save
                training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, gaussians, eval_render_wrapper, (pipe, background))
                if ((iteration - rank) in saving_iterations and dist.get_rank() == 0):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration, gaussians=gaussians)

                # this empty_cache is necessary to prevent the sustained growth of GPU memory cost
                # but maybe we do not have to call it every iteration
                if ((iteration - rank) % opt.densification_interval == 0):
                    torch.cuda.empty_cache()
                try:
                    # Densification
                    if (iteration - rank) < opt.densify_until_iter:
                        # Keep track of max radii in image-space for pruning
                        gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                        # gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                        # DDP has averaged the _means2D.grad, add for radii<=0, grad is also 0, thus a simple addition is enough 

                        # if rank == 0:
                        #     print(gaussians._means2D_meta.grad, torch.norm(gaussians._means2D_meta.grad, dim=-1, keepdim=True))

                        gaussians.xyz_gradient_accum += world_size*gaussians._means2D_meta.grad
                        gaussians.denom[visibility_filter] += 1

                        if ((iteration - rank) > opt.densify_from_iter) and is_interval_in_batch((iteration - rank), world_size, opt.densification_interval):
                            # DDP has averaged the _means2D.grad for all ranks, no need to collect it
                            # dist.reduce(gaussians.xyz_gradient_accum, dst=0, op=dist.ReduceOp.SUM)
                            dist.reduce(gaussians.denom, dst=0, op=dist.ReduceOp.SUM)
                            dist.reduce(gaussians.max_radii2D, dst=0, op=dist.ReduceOp.MAX) 

                            t0 = time.time()
                            size_threshold = 20 if (iteration - rank) > opt.opacity_reset_interval else None
                            if dist.get_rank() == 0:
                                gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                                # print('\n rank {} has done densify_and_prune'.format(dist.get_rank()))
                                gaussians_shape = torch.zeros((1, ), dtype=torch.int64, device='cuda')
                                gaussians_shape[0] = gaussians._xyz.shape[0]
                                # print('new size: {}'.format(gaussians._xyz.shape[0]))
                            else:
                                gaussians_shape = torch.zeros((1, ), dtype=torch.int64, device='cuda')  

                            dist.broadcast(gaussians_shape, src=0) 

                            if dist.get_rank() == 0:
                                for group in gaussians.optimizer.param_groups:
                                    stored_state = gaussians.optimizer.state.get(group['params'][0])
                                    dist.broadcast(group['params'][0].data, src=0)
                                    dist.broadcast(stored_state['exp_avg'], src=0)  
                                    dist.broadcast(stored_state['exp_avg_sq'], src=0)  
                            else:
                                tensor_dict = {}
                                for group in gaussians.optimizer.param_groups:
                                    name, shape = group["name"], list(group['params'][0].shape)
                                    shape[0] = gaussians_shape
                                    dtype, device = group['params'][0].dtype, group['params'][0].device
                                    new_group = {
                                        "tensor": torch.zeros(size=shape, dtype=dtype, device=device),
                                        "exp_avg": torch.zeros(size=shape, dtype=dtype, device=device),
                                        "exp_avg_sq": torch.zeros(size=shape, dtype=dtype, device=device)
                                    }
                                    dist.broadcast(new_group['tensor'], src=0)
                                    dist.broadcast(new_group['exp_avg'], src=0)  
                                    dist.broadcast(new_group['exp_avg_sq'], src=0) 
                                    tensor_dict[name] = new_group
                                gaussians.clone_from_optimizer_dict(tensor_dict) 

                            # print('rank {} has cloned the data, {}, {}, {}, {}'.format(dist.get_rank(), gaussians._xyz.max(), 
                            #         gaussians._xyz.min(),   gaussians._xyz.mean(),    iteration   ))

                            # todo del ddp and assign new one    
                            del gaussians_ddp   # parameters are changed, discard old ddp model
                            gaussians_ddp = DDP(gaussians, device_ids=[dist.get_rank()], find_unused_parameters=True)
                            torch.cuda.empty_cache()
                            t1 = time.time()
                            if tb_writer:
                                tb_writer.add_scalar('cnt/densify_and_prune', t1-t0, iteration)      

                        if is_interval_in_batch((iteration - rank), world_size, opt.opacity_reset_interval, 1) or (dataset.white_background and (iteration - rank) == opt.densify_from_iter):
                            gaussians.reset_opacity()
                            print('rank {} has reset opacity, {}'.format(dist.get_rank(), iteration))
                        
                except Exception as e:
                    traceback.print_exc()         

                # Optimizer 
                if (iteration - rank) < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

                if ((iteration - rank) in checkpoint_iterations and dist.get_rank() == 0):
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            #
            iteration += world_size
            if (iteration - rank) > opt.iterations:
                end_train = True
                break
        if end_train:
            break
    

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
    train_args = (lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

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