import os
import torch
from torch.utils.data import DataLoader
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer.render import render
import sys, time
from scene import SimpleScene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from lpipsPyTorch import LPIPS
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from torch.profiler import profile, record_function, ProfilerActivity
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = False
except ImportError:
    TENSORBOARD_FOUND = False  

def render_wrapper(viewpoint_cam, gaussians, pipe, background):
    viewpoint_cam.to_device('cuda')
    return render(viewpoint_cam, gaussians, pipe, background)    

# only perfrom forward and backward, skip optimizer.step and all point-management  
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = SimpleScene(dataset, load_iteration=-1) # set load_iteration=-1 to enable search .ply
    scene.load2gaussians(gaussians)
    gaussians.training_setup(opt)
    gaussians.active_sh_degree = gaussians.max_sh_degree

    print('spatial_lr_scale is {}, active_sh_degree is {}'.format(gaussians.spatial_lr_scale, gaussians.active_sh_degree))
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    train_dataset, test_dataset = scene.getTrainCameras(), scene.getTestCameras()
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.epochs*len(train_dataset)), desc="Training progress")
    NUM_EPOCH = opt.epochs
    first_iter += 1 # origin code add 1 to avoid corner cases on iteration 0, we just follow it
    train_loader = DataLoader(train_dataset, batch_size=1, prefetch_factor=4, shuffle=True, drop_last=False, num_workers=32, 
                              collate_fn=SimpleScene.get_batch, pin_memory=True, pin_memory_device='cuda')
    iteration = first_iter

    train_data_list = []
    for _i, ids_data in enumerate(train_loader):
        if True:
            data = ids_data
            data_gpu = [_cmr.to_device('cuda') for _cmr in data]
            train_data_list.append(data_gpu)

    print('start of profiling')
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(
            scene.model_path, 'profiler'
        )),
    ) as p:
        for i_epoch in range(NUM_EPOCH):
            for data in train_data_list:
                # Pick a random Camera
                with record_function("custom_forward"):
                    viewpoint_cam = data[0]

                    bg = torch.rand((3), device="cuda") if opt.random_background else background
                    render_pkg = render_wrapper(viewpoint_cam, gaussians, pipe, bg)
                    image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

                    # Loss
                    gt_image = viewpoint_cam.original_image.cuda()
                    Ll1 = l1_loss(image, gt_image)
                    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
                    torch.cuda.synchronize()

                with record_function("custom_backward"):
                    loss.backward()
                    torch.cuda.synchronize()

                p.step()
                progress_bar.update(1)
                iteration += 1
    progress_bar.close()

    table = p.key_averages().table(sort_by="cpu_time_total", row_limit=-1, max_src_column_width=200) 
    print(table)
    current_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())   
    # p.export_chrome_trace(os.path.join(
    #     dataset.model_path, 'profile_{}.json'.format(current_time))
    #     )   
    with open(os.path.join(
        dataset.model_path, 'profile_{}.txt'.format(current_time)
    ), 'w') as f:
        f.writelines(table)          

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
