import os, sys
import traceback, uuid, logging, time, shutil, glob
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
import datetime
os.environ["NCCL_SOCKET_TIMEOUT"] = "60000"

from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

from utils.general_utils import safe_state
from parallel_utils.basic_parallel_trainer.trainer4tree_partition import Trainer4TreePartition

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
torch.multiprocessing.set_sharing_strategy('file_system')


def mp_setup(rank, world_size, LOCAL_RANK, MASTER_ADDR, MASTER_PORT):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = str(MASTER_ADDR)
    os.environ["MASTER_PORT"] = str(MASTER_PORT)
    
    init_process_group(backend="nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=6000))
    torch.cuda.set_device(LOCAL_RANK)

def main(rank: int, world_size: int, LOCAL_RANK: int, MASTER_ADDR, MASTER_PORT, train_args):
    mp_setup(rank, world_size, LOCAL_RANK, MASTER_ADDR, MASTER_PORT)
    args:Namespace; mdp:ModelParams; opt:OptimizationParams; pipe:PipelineParams 
    args, mdp, opt, pipe = train_args
    args.SCENE_GRID_SIZE = np.array([2*1024, 2*1024, 1], dtype=int)
    args.SPLIT_ORDERS = [0, 1]

    current_time = time.strftime("%Y_%m_%d_%H", time.localtime())
    debug_logger = logging.getLogger('debugger')

    os.makedirs('debugger/', exist_ok=True)
    file_handler = logging.FileHandler('debugger/debug_{}_{}.txt'.format(rank, current_time), mode='a')
    file_handler.setFormatter(logging.Formatter('%(asctime)s-%(filename)s[line:%(lineno)d]-%(levelname)s: %(message)s'))
    debug_logger.addHandler(file_handler)
    debug_logger.setLevel(logging.INFO)
    debug_logger.info('prepared')
    try:
        trainer = Trainer4TreePartition(None, mdp, opt, pipe, args, ply_iteration=args.ply_iteration)
        logger_level:int = args.log_level
        trainer.logger.setLevel(logger_level)
        if args.EVAL_ONLY:
            trainer.eval(train_iteration=args.ply_iteration)
        else:
            trainer.train()
    except:
        tb_str = traceback.format_exc()
        debug_logger.error(tb_str)    
    # destroy_process_group()       

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 15_000, 20_000, 30_000])
    parser.add_argument("--ply_iteration", type=int, default=-1)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--bvh_depth", type=int, default=2, help='num_model_would be 2**bvh_depth')
    parser.add_argument("--logdir", type=str, default='', help='path for log files')
    parser.add_argument("--CKPT_MAX_NUM", type=int, default=5)
    # grid parameters
    parser.add_argument("--DISABLE_TENSORBOARD", action='store_true', default=False)
    parser.add_argument("--ENABLE_REPARTITION", action='store_true', default=False)
    parser.add_argument("--REPARTITION_START_EPOCH", type=int, default=10)
    parser.add_argument("--REPARTITION_END_EPOCH", type=int, default=300)
    parser.add_argument("--REPARTITION_INTERVAL_EPOCH", type=int, default=50)
    parser.add_argument("--EVAL_PSNR_INTERVAL", type=int, default=8)
    parser.add_argument("--Z_NEAR", type=float, default=0.01)
    parser.add_argument("--Z_FAR", type=float, default=1000)
    parser.add_argument("--EVAL_INTERVAL_EPOCH", type=int, default=5)
    parser.add_argument("--SAVE_INTERVAL_EPOCH", type=int, default=5)
    parser.add_argument("--SAVE_INTERVAL_ITER", type=int, default=50000)
    parser.add_argument("--SKIP_PRUNE_AFTER_RESET", type=int, default=0)
    parser.add_argument("--SKIP_SPLIT", action='store_true', default=False)
    parser.add_argument("--SKIP_CLONE", action='store_true', default=False)

    parser.add_argument("--SHRAE_GS_INFO", action='store_true', default=False, help='transport the primitives on model boundary')
    parser.add_argument("--MAX_SIZE_SINGLE_GS", type=int, default=10_000_000)
    parser.add_argument("--MAX_LOAD", type=int, default=16)
    parser.add_argument("--MAX_BATCH_SIZE", type=int, default=4)
    parser.add_argument("--DATALOADER_FIX_SEED", action='store_true', default=False)
    parser.add_argument("--EVAL_ONLY", action='store_true', default=False)
    parser.add_argument("--SAVE_EVAL_IMAGE", action='store_true', default=False)
    parser.add_argument("--SAVE_EVAL_SUB_IMAGE", action='store_true', default=False)
    
    parser.add_argument("--log_level", type=int, default=10, 
        help='CRITICAL=50, FATAL=CRITICAL, ERROR=40, WARNING=30, WARN=WARNING, INFO=20, DEBUG=10, NOTSET=0')

    args = parser.parse_args(sys.argv[1:])

    # Initialize system state (RNG)
    safe_state(args.quiet, init_gpu=False)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    RANK = int(os.environ["RANK"])
    LOCAL_RANK  = int(os.environ["LOCAL_RANK"])
    MASTER_ADDR = os.environ["MASTER_ADDR"]
    MASTER_PORT = os.environ["MASTER_PORT"]

    assert WORLD_SIZE <= 2**args.bvh_depth
    train_args = (args, lp.extract(args), op.extract(args), pp.extract(args))
    main(RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT, train_args)

    # All done
    print("\nTraining complete.")