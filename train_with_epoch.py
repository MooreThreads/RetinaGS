# Inherit from train_with_dataset.py

import os
import torch
from torch.utils.data import DataLoader
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer.render import render
from gaussian_renderer.render_metric import render as render_metric
import sys
from scene import SimpleScene, GaussianModel
from utils.general_utils import safe_state
from lpipsPyTorch import LPIPS
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

def render_wrapper(viewpoint_cam, gaussians, pipe, background):
    viewpoint_cam.to_device('cuda')
    return render(viewpoint_cam, gaussians, pipe, background)    

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = SimpleScene(dataset, load_iteration=-1)
    scene.load2gaussians(gaussians)
    train_dataset, test_dataset = scene.getTrainCameras(), scene.getTestCameras()    
    opt.position_lr_max_steps = opt.epochs * len(train_dataset) # Auto update max steps
    opt.scaling_lr_max_steps = opt.position_lr_max_steps
    gaussians.training_setup(opt)
    if opt.perception_loss:
        CNN_IMAGE = LPIPS(net_type=opt.perception_net_type, 
                    version=opt.perception_net_version).to('cuda')
        print('lpips:{}.{}'.format(opt.perception_net_type, opt.perception_net_version))
    else:
        CNN_IMAGE = None
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.position_lr_max_steps), desc="Training progress")
    first_iter += 1 # origin code add 1 to avoid corner cases on iteration 0, we just follow it
    iteration = first_iter
    
    epoch = 0  
    
    while epoch < int(opt.epochs): # for i_epoch in range(NUM_EPOCH):
        
        epoch += 1 
        train_loader = DataLoader(train_dataset, batch_size=1, prefetch_factor=4, shuffle=True, drop_last=False, num_workers=32, collate_fn=SimpleScene.get_batch)
        gaussians.update_learning_rate(iteration) # update lr every epoch    
        if opt.lr_scales_schedule:
            gaussians.update_learning_rate_scaling(iteration)     

        for data in train_loader:
            iter_start.record()

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
            image, viewspace_point_tensor, visibility_filter, radii, scales = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["scales"]

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            
            if opt.scales_reg_enable:
                scales_reg = scales.prod(dim=1).mean()
                loss += opt.scales_reg_lr * scales_reg
                
            if opt.scales_reg_sum_enable:
                scales_reg = scales.sum(dim=1).mean()
                loss += opt.scales_reg_sum_lr * scales_reg
            
            if opt.scales_reg_square_sum_enable:
                scales_reg = scales.pow(2).sum(dim=1).mean()
                loss += opt.scales_reg_square_sum_lr * scales_reg
            
            # Pending for aligning with conventional usage
            if opt.perception_loss:
                perception_loss_val = torch.sum(CNN_IMAGE(image, gt_image))
                loss += opt.lambda_perception * perception_loss_val
                
            loss.backward()

            iter_end.record()

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)

                # Log and save
                if iteration % 100 == 0:
                    training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, gaussians, render_wrapper, (pipe, background))

                # Densification
                if epoch < opt.densify_until_epoch: # iteration < opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                    
                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0] <= opt.max_gaussians: # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

                # Only purning
                if opt.only_prune_via_screen_space_enable:
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    if iteration % 200 == 0:
                        size_threshold = opt.screen_size_threshold 
                        gaussians.only_prune_via_screen_space(size_threshold)
                    
                
                # Optimizer step
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

                if (iteration in checkpoint_iterations):
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            iteration += 1

        with torch.no_grad():
            if epoch % opt.eval_epoch_interval == 0:            
                training_report_epoch(tb_writer, opt, epoch, Ll1, loss, l1_loss, CNN_IMAGE, iter_start.elapsed_time(iter_end), testing_iterations, scene, gaussians, render_wrapper, (pipe, background))            
                
            if opt.save_epoch_interval == -1 :
                if epoch == opt.epochs:                
                    print("\n[EPOCH {}] Saving Gaussians".format(epoch))    
                    scene.save(iteration, gaussians=gaussians)
            elif epoch % opt.save_epoch_interval == 0:
                print("\n[EPOCH {}] Saving Gaussians".format(epoch))    
                scene.save(iteration, gaussians=gaussians)

        
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
        if iteration % 100 == 0:
            tb_writer.add_scalar('total_points', gaussians.get_xyz.shape[0], iteration)
        
def training_report_epoch(tb_writer, opt : PipelineParams, epoch, Ll1, loss, l1_loss, CNN_IMAGE, elapsed, testing_iterations, scene : SimpleScene, gaussians, renderFunc, renderArgs):
    # Report test and 1/8 of training set
    torch.cuda.empty_cache()
    train_dataset = scene.getTrainCameras()
    validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                          {'name': 'train', 'cameras' :  [train_dataset[idx] for idx in range(0, len(train_dataset), 8)]})       
    for config in validation_configs:
        if config['cameras'] and len(config['cameras']) > 0:
            l1_test = 0.0
            psnr_test = 0.0
            ssim_test = 0.0
            lpips_test = 0.0
            for idx, viewpoint in enumerate(config['cameras']):                
                render_pkg = renderFunc(viewpoint, gaussians, *renderArgs)
                image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                if epoch % opt.visualization_epoch_interval == 0:                    
                    if tb_writer and (idx % int(len(config['cameras']) / opt.number_visualization) == 0):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=epoch)
                        if epoch == opt.visualization_epoch_interval:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=epoch)
                l1_test += l1_loss(image, gt_image).mean().double()
                psnr_test += psnr(image, gt_image).mean().double()
                ssim_test += ssim(image, gt_image)
                if opt.perception_loss:
                    lpips_test += torch.sum(CNN_IMAGE(image, gt_image))
            psnr_test /= len(config['cameras'])
            l1_test /= len(config['cameras'])          
            ssim_test /= len(config['cameras'])    
            lpips_test /= len(config['cameras'])       
            print("\n[EPOCH {}] Evaluating {}: L1 {} PSNR {} SSIM {} perception loss {}".format(epoch, config['name'], l1_test, psnr_test, ssim_test,lpips_test))
            
            
            if tb_writer:
                tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, epoch)
                tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, epoch)
                tb_writer.add_scalar(config['name'] + '/loss_viewpoint - SSIM', ssim_test, epoch)
                if opt.perception_loss:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - perception_loss ' + opt.perception_net_type + '.' + opt.perception_net_version, lpips_test, epoch)                     
                scales_reg = render_pkg["scales"].prod(dim=1).mean()               
                tb_writer.add_scalar(config['name'] + '/loss_viewpoint - scales_reg', scales_reg, epoch)
                scales_reg_sum = render_pkg["scales"].sum(dim=1).mean()               
                tb_writer.add_scalar(config['name'] + '/loss_viewpoint - scales_reg_sum', scales_reg_sum, epoch)
                scales_reg_square_sum = render_pkg["scales"].pow(2).sum(dim=1).mean()
                tb_writer.add_scalar(config['name'] + '/loss_viewpoint - scales_reg_square_sum', scales_reg_square_sum, epoch)
                    
    torch.cuda.empty_cache()
        
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
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
