import os, sys
import traceback, uuid, logging, time, shutil, glob
from tqdm import tqdm
import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
import torchvision
import datetime
os.environ["NCCL_SOCKET_TIMEOUT"] = "60000"

from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from utils.general_utils import safe_state, is_interval_in_batch, is_point_in_batch, build_rotation
from utils.workload_utils import NaiveWorkloadBalancer
from gaussian_renderer.render_half_gs import RenderInfoFromGS, render4renderinfo

import parallel_utils.schedulers.optical_field_segment as pso
import parallel_utils.grid_utils.core as pgc
import parallel_utils.grid_utils.gaussian_grid as pgg
import parallel_utils.grid_utils.utils as ppu

from scene.gaussian_nn_module import BoundedGaussianModel, BoundedGaussianModelGroup
from scene.cameras import Camera, EmptyCamera, ViewMessage
from utils.datasets import CameraListDataset, DatasetRepeater, GroupedItems
from scene.scene4bounded_gaussian import SceneV3
from lpipsPyTorch import LPIPS
from typing import List, Dict, Tuple

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

MAX_GS_CHANNEL = 59*3   
# 59 is for 3 ranks of Spherical Harmonics, compute it manually or DO NOT ENABLE_REPARTITION
SPLIT_ORDERS = [0, 1]
ENABLE_REPARTITION = False
REPARTITION_START_EPOCH = 10
REPARTITION_END_EPOCH = 300
REPARTITION_INTERVAL_EPOCH = 50
SCENE_GRID_SIZE = np.array([2*1024, 2*1024, 1], dtype=int)
EVAL_PSNR_INTERVAL = 8
MAX_SIZE_SINGLE_GS = int(6e7)
Z_NEAR = 0.01
Z_FAR = 1*1000
EVAL_INTERVAL_EPOCH = 5
SAVE_INTERVAL_EPOCH = 1
SAVE_INTERVAL_ITER = 50000
SKIP_PRUNE_AFTER_RESET = 0
SKIP_SPLIT = False
SKIP_CLONE = False
PERCEPTION_LOSS = False
CNN_IMAGE = None
DATALOADER_FIX_SEED = False
GLOBAL_CKPT_CLEANER:pgc.ckpt_cleaner = None

def grid_setup(train_args, logger:logging.Logger):
    args = train_args[0]
    opt = train_args[2]
    global TENSORBOARD_FOUND; TENSORBOARD_FOUND = TENSORBOARD_FOUND and args.ENABLE_TENSORBOARD
    global ENABLE_REPARTITION; ENABLE_REPARTITION = args.ENABLE_REPARTITION
    global REPARTITION_START_EPOCH; REPARTITION_START_EPOCH = args.REPARTITION_START_EPOCH
    global REPARTITION_END_EPOCH; REPARTITION_END_EPOCH = args.REPARTITION_END_EPOCH
    global REPARTITION_INTERVAL_EPOCH; REPARTITION_INTERVAL_EPOCH = args.REPARTITION_INTERVAL_EPOCH 
    global EVAL_PSNR_INTERVAL; EVAL_PSNR_INTERVAL = args.EVAL_PSNR_INTERVAL
    global Z_NEAR; Z_NEAR = args.Z_NEAR
    global Z_FAR; Z_FAR = args.Z_FAR
    global EVAL_INTERVAL_EPOCH; EVAL_INTERVAL_EPOCH= args.EVAL_INTERVAL_EPOCH
    global SAVE_INTERVAL_EPOCH; SAVE_INTERVAL_EPOCH = args.SAVE_INTERVAL_EPOCH
    global SAVE_INTERVAL_ITER; SAVE_INTERVAL_ITER = args.SAVE_INTERVAL_ITER
    global SKIP_PRUNE_AFTER_RESET; SKIP_PRUNE_AFTER_RESET = args.SKIP_PRUNE_AFTER_RESET
    global SKIP_SPLIT; SKIP_SPLIT = args.SKIP_SPLIT
    global SKIP_CLONE; SKIP_CLONE = args.SKIP_CLONE
    global PERCEPTION_LOSS; PERCEPTION_LOSS = opt.perception_loss
    global CNN_IMAGE
    if PERCEPTION_LOSS:
        CNN_IMAGE = LPIPS(
            net_type=opt.perception_net_type,
            version=opt.perception_net_version
            ).to('cuda')
        logger.info("PERCEPTION_LOSS is {}.{}".format(opt.perception_net_type, opt.perception_net_version))
    logger.info("{}, {}".format(SKIP_SPLIT, SKIP_CLONE))   
    global GLOBAL_CKPT_CLEANER; GLOBAL_CKPT_CLEANER = pgc.ckpt_cleaner(max_len=args.CKPT_MAX_NUM)

torch.multiprocessing.set_sharing_strategy('file_system')

class CamerawithRelation(Dataset):
    def __init__(self, dataset:CameraListDataset, path2nodes:dict, sorted_leaf_nodes:list, logger:logging.Logger) -> None:
        super().__init__()
        self.dataset = dataset
        self.path2nodes = path2nodes
        self.sorted_leaf_nodes = sorted_leaf_nodes
        self.logger = logger
        self.z_far_small = 100.0
        self.z_far_big = 1000.0
        self.z_eps = 100.0

    def __len__(self):
        return len(self.dataset)   
    
    def load_related_depth(self, image_path):
        # i know it is awful to code in this way, but it saves my time
        # 2333
        depth_folder = '/jfs/shengyi.chen/HT/Data/MatrixCity/bdaibdai___MatrixCity/small_city_depth'
        normalized_path = os.path.normpath(image_path)
        parts = normalized_path.split(os.sep)
        aerial_street, train_test, block_order, filename = parts[-4:]
        pure_filename = os.path.splitext(filename)[0]
        depth_name = os.path.join(depth_folder, aerial_street, train_test, block_order + '_depth', pure_filename + '.exr')
        if os.path.exists(depth_name):
            depth = cv2.imread(depth_name, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[...,0]
            depth = depth / 100 # 量纲 1cm -> 1m
            return depth
        else:
            return None

    def __getitem__(self, idx):     
        camera: Camera = self.dataset[idx]
        info = self.dataset.cameras_infos[idx]
        gt_depth = self.load_related_depth(info.image_path)
        if gt_depth is not None:
            gt_depth_max = np.max(gt_depth)
            masks = (gt_depth == gt_depth_max)
            gt_depth[masks] = 0
            estimated_depth = np.max(gt_depth)
        else:
            estimated_depth = Z_FAR   

        max_depth = max(min(estimated_depth + 50, Z_FAR), 10) 
       
        relation_1_N = pgg.get_relation_vector(
            camera, 
            z_near=0.01, 
            z_far=max_depth, 
            path2nodes=self.path2nodes, 
            sorted_leaf_nodes=self.sorted_leaf_nodes,
            logger=self.logger
            )

        return max_depth, relation_1_N, camera 

def save_GS(iteration, model_path, gaussians_group:BoundedGaussianModelGroup, path2node:dict):    
    RANK = dist.get_rank()
    print("\n[rank {}, ITER {}] Saving Gaussians".format(RANK, iteration))
    point_cloud_path = os.path.join(model_path, "point_cloud/iteration_{}".format(iteration))
    GLOBAL_CKPT_CLEANER.add(point_cloud_path)
    for model_id in gaussians_group.all_gaussians: 
        _model: BoundedGaussianModel = gaussians_group.all_gaussians[model_id]
        # GS model may not be organized as nn.Module
        # thus save GS model as .ply and save optimizer as .pt
        _model.save_ply(os.path.join(point_cloud_path, "point_cloud_{}.ply".format(model_id)))
        torch.save(
            _model.optimizer.state_dict(),
            os.path.join(point_cloud_path, "adam_{}.pt".format(model_id))
            )
    
    if path2node is not None:
        ppu.save_BvhTree_on_3DGrid(path2node, os.path.join(point_cloud_path, "tree_{}.txt".format(RANK)))

def get_relation_matrix(train_dataset:CameraListDataset, path2nodes:dict, sorted_leaf_nodes:list, logger:logging.Logger):
    NUM_DATA, NUM_MODEL = len(train_dataset), len(sorted_leaf_nodes)
    complete_relation = np.zeros((NUM_DATA, NUM_MODEL), dtype=int) - 1

    data_loader = DataLoader(CamerawithRelation(train_dataset, path2nodes, sorted_leaf_nodes, logger), 
                            batch_size=16, num_workers=32, prefetch_factor=4, drop_last=False,
                            shuffle=False, collate_fn=SceneV3.get_batch)
    
    idx_start = 0
    for i, batch in tqdm(enumerate(data_loader)):
        for _data in batch:
            _max_depth, _relation_1_N, camera = _data
            assert isinstance(camera, (Camera, EmptyCamera))
            assert camera.uid == idx_start
            complete_relation[idx_start, :] = _relation_1_N
            if i%100 == 0:
                logger.info("{}, {}, {}, {}".format(camera.image_height, camera.original_image.shape, camera.uid, _max_depth))
            idx_start += 1
    # if a row is all -1, set it to all 0
    all_minus_one = (complete_relation.sum(axis=-1) == -NUM_MODEL)  
    complete_relation[all_minus_one, :] = 0
    logger.info('find {} mismatching samples, set their relation to all 0'.format(np.sum(all_minus_one)))

    return complete_relation 

def update_relation_matrix_dist(scene:SceneV3, path2nodes:dict, sorted_leaf_nodes:list, NUM_MODEL:int, logger:logging.Logger):
    RANK, WORLD_SIZE = dist.get_rank(), dist.get_world_size()

    train_dataset: CameraListDataset = scene.getTrainCameras() 
    eval_test_dataset: CameraListDataset = scene.getTestCameras()

    if RANK == 0:
        assert len(sorted_leaf_nodes) == NUM_MODEL
        trainset_relation_np = get_relation_matrix(train_dataset, path2nodes, sorted_leaf_nodes, logger)
        evalset_relation_np = get_relation_matrix(eval_test_dataset, path2nodes, sorted_leaf_nodes, logger)
        trainset_relation_tensor = torch.tensor(trainset_relation_np, dtype=torch.int, device='cuda')
        evalset_relation_tensor = torch.tensor(evalset_relation_np, dtype=torch.int, device='cuda')
    else:
        trainset_relation_tensor = torch.zeros((len(train_dataset), NUM_MODEL), dtype=torch.int, device='cuda')
        evalset_relation_tensor = torch.zeros((len(eval_test_dataset), NUM_MODEL), dtype=torch.int, device='cuda')

    dist.barrier(group=None)
    dist.broadcast(trainset_relation_tensor, src=0, group=None, async_op=False)
    logger.info(trainset_relation_tensor)
    if len(eval_test_dataset) > 0:
        dist.broadcast(evalset_relation_tensor, src=0, group=None, async_op=False)
    else:
        logger.warning('strange! empty eval dataset') 
    torch.cuda.synchronize()
    return trainset_relation_tensor.cpu(), evalset_relation_tensor.cpu()

def get_sampler_indices_dist(train_dataset:CameraListDataset, seed:int):
    RANK, WORLD_SIZE = dist.get_rank(), dist.get_world_size()

    N = len(train_dataset)
    positions = np.zeros((N, 3), dtype=float)
    for i in range(N):
        positions[i, :] = train_dataset.get_empty_item(i).camera_center

    if RANK == 0:
        # seed = np.random.randint(1000)
        g = torch.Generator()
        g.manual_seed(seed)
        indices_gpu = torch.randperm(len(train_dataset), generator=g, dtype=torch.int).to('cuda')
    else:
        g = torch.Generator()
        g.manual_seed(seed)
        indices_gpu = torch.randperm(len(train_dataset), generator=g, dtype=torch.int).to('cuda')

    dist.broadcast(indices_gpu, src=0, group=None, async_op=False)
    return indices_gpu.tolist()

def get_grouped_indices_dist(model2rank:dict, relation_matrix:torch.Tensor, shuffled_indices:np.ndarray, max_task:int, max_batch:int):
    RANK, WORLD_SIZE = dist.get_rank(), dist.get_world_size()
    groups_size_tensor = torch.zeros(len(relation_matrix), dtype=torch.int, device='cuda')

    if RANK == 0:
        balancer = NaiveWorkloadBalancer(num_rank=WORLD_SIZE, model2rank=model2rank)
        groups = balancer.get_groups(relation_matrix=relation_matrix, shuffled_indices=shuffled_indices, max_task=max_task, max_batch=max_batch)
        groups_size_array, num_groups = np.array([len(_g) for _g in groups]), len(groups)
        assert np.sum(groups_size_array) == len(relation_matrix), "size of groups mismatch"
        groups_size_tensor[:num_groups] += torch.tensor(groups_size_array, dtype=torch.int, device='cuda')
    
    dist.broadcast(groups_size_tensor, src=0, group=None, async_op=False)
    groups_size_np = groups_size_tensor.cpu().numpy()

    groups, g_head = [], 0
    for g_size in groups_size_np:
        if g_size > 0:
            groups.append(tuple(shuffled_indices[g_head:(g_head + g_size)]))
        g_head += g_size
    return groups  

def init_datasets_dist(scene:SceneV3, opt, path2nodes:dict, sorted_leaf_nodes:list, NUM_MODEL:int, logger:logging.Logger):
    RANK, WORLD_SIZE = dist.get_rank(), dist.get_world_size()

    train_dataset: CameraListDataset = scene.getTrainCameras() 
    eval_test_dataset: CameraListDataset = scene.getTestCameras()
    eval_train_list = DatasetRepeater(train_dataset, len(train_dataset)//EVAL_PSNR_INTERVAL, False, EVAL_PSNR_INTERVAL)
    torch.cuda.synchronize()
    if RANK == 0:
        assert len(sorted_leaf_nodes) == NUM_MODEL
        relation_path = os.path.join(scene.model_path, 'trainset_relation.pt')      
        if os.path.exists(relation_path):
            trainset_relation_tensor:torch.Tensor = torch.load(relation_path)
            trainset_relation_tensor = trainset_relation_tensor.cuda()
        else:    
            trainset_relation_np = get_relation_matrix(train_dataset, path2nodes, sorted_leaf_nodes, logger)
            trainset_relation_tensor = torch.tensor(trainset_relation_np, dtype=torch.int, device='cuda')
        evalset_relation_np = get_relation_matrix(eval_test_dataset, path2nodes, sorted_leaf_nodes, logger)
        evalset_relation_tensor = torch.tensor(evalset_relation_np, dtype=torch.int, device='cuda')
    else:
        trainset_relation_tensor = torch.zeros((len(train_dataset), NUM_MODEL), dtype=torch.int, device='cuda')
        evalset_relation_tensor = torch.zeros((len(eval_test_dataset), NUM_MODEL), dtype=torch.int, device='cuda')

    dist.barrier()
    dist.broadcast(trainset_relation_tensor, src=0, group=None, async_op=False)
    logger.info(trainset_relation_tensor)
    if len(eval_test_dataset) > 0:
        dist.broadcast(evalset_relation_tensor, src=0, group=None, async_op=False)
    else:
        logger.warning('strange! empty eval dataset')    

    torch.cuda.synchronize()
    if RANK == 0:
        torch.save(trainset_relation_tensor, os.path.join(scene.model_path, 'trainset_relation.pt'))

    logger.info('dataset is prepared')
    return train_dataset, eval_test_dataset, eval_train_list, trainset_relation_tensor.cpu(), evalset_relation_tensor.cpu()

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

def prepare_output_and_logger(args, all_args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/rank_{}".format(dist.get_rank()), unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
        try:
            cfg_log_f.write('\n'+str(all_args))
        except:
            pass    

    complete_args = all_args[0]
    RANK = dist.get_rank()
    logdir4rank = args.model_path
    if len(complete_args.logdir) > 0:
        logdir4rank = os.path.join(complete_args.logdir, 'rank_{}'.format(RANK))
        os.makedirs(logdir4rank, exist_ok=True)
        
    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(logdir4rank)
    else:
        print("Tensorboard not available: not logging progress")

    current_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) 

    logging.basicConfig(
        format='%(asctime)s-%(filename)s[line:%(lineno)d]-%(levelname)s: %(message)s',
        filemode='w',
        filename=os.path.join(logdir4rank, 'rank_{}_{}.txt'.format(RANK, current_time))
    )
    logger = logging.getLogger('rank_{}'.format(dist.get_rank()))
    logger.setLevel(logging.INFO)

    return tb_writer, logger

def training_report(
        tb_writer, logger:logging.Logger, iteration:int, Ll1:torch.tensor, loss:torch.tensor, batch_size:int,
        l1_loss:callable, render_func:callable, modelId2rank:dict, local_func_blender:callable,
        elapsed: float, testing_iterations: list, validation_configs:dict, 
        scene: SceneV3, gaussians_group: BoundedGaussianModelGroup, scheduler:pso.BasicSchedulerwithDynamicSpace):
    RANK = dist.get_rank()
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('cnt/memory', torch.cuda.memory_allocated()/(1024**3), iteration)
        # torch.cuda.empty_cache()

def unpack_data(cameraUid_taskId_mainRank:torch.Tensor, packages:torch.Tensor, cams_gpu:list, logger:logging.Logger): 
    view_messages = [ViewMessage(e, id=int(_id)) for e, _id in zip(packages, cameraUid_taskId_mainRank[:,1])]  
    task_id2cameraUid, uid2camera, task_id2camera = {}, {}, {}
    for camera in cams_gpu:
        assert isinstance(camera, Camera)
        uid2camera[camera.uid] = camera 
    for row in cameraUid_taskId_mainRank:
        uid, tid, mid = int(row[0]), int(row[1]), int(row[2])
        task_id2cameraUid[tid] = uid
        task_id2camera[tid] = uid2camera[uid]
    logger.debug(uid2camera)    
    
    return view_messages, task_id2cameraUid, uid2camera, task_id2camera

def gather_image_loss(main_rank_tasks:list, images:dict, task_id2camera:dict, opt, pipe, logger: logging.Logger):
    t0 = time.time() 
    loss_dict = {}
    loss_main_rank, Ll1_main_rank = torch.tensor(0.0, device='cuda'), torch.tensor(0.0, device='cuda')    
    _t_gather, _t_loss = 0, 0
    
    for k in images:
        _t0 = time.time()
        task_id, task_main_rank = k
        image = images[k]['render']
        ori_camera:Camera = task_id2camera[task_id]
        gt_image = ori_camera.original_image.to('cuda')

        _t1 = time.time()
        _t_gather += (_t1 - _t0)
        
        _t0 = time.time()
        # suppression of shooting noise
        if opt.huber_loss_replacement_enable:
            Ll1 = torch.nn.functional.huber_loss(image, gt_image)
        else:
            Ll1 = l1_loss(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        if PERCEPTION_LOSS:
            assert CNN_IMAGE is not None
            loss = (1.0 - opt.lambda_dssim) * Ll1 \
                + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) \
                + opt.lambda_perception * torch.sum(CNN_IMAGE(image, gt_image))
        loss_dict[k] = loss
                
        loss_main_rank += loss
        Ll1_main_rank += Ll1
        _t1 = time.time()
        _t_loss += (_t1 - _t0)
        logger.debug('taskid_mainRank {}, loss {}, l1 {}'.format(k, loss, Ll1))

    return loss_main_rank, Ll1_main_rank, loss_dict, _t_gather, _t_loss
   
def gather_render_info(local_pool:Dict, extra_pool:Dict, local_model_ids:List[int], logger:logging.Logger):
    id2infos = {int(_id):[] for _id in local_model_ids}
    for p in [local_pool, extra_pool]:
        for src_dst, info in p.items():
            src, dst = src_dst
            if dst not in id2infos:
                continue
            # make sure that render_info from the model itself occurs first
            if src == dst:
                id2infos[dst].insert(0, info)
            else:
                id2infos[dst].append(info)
    ret = {}
    for mid, info_list in id2infos.items():
        ret[mid] = RenderInfoFromGS(
            means3D = torch.cat([info.means3D for info in info_list], dim=0).contiguous(),
            means2D = torch.cat([info.means2D for info in info_list], dim=0).contiguous(),
            shs = torch.cat([info.shs for info in info_list], dim=0).contiguous(),
            opacity = torch.cat([info.opacity for info in info_list], dim=0).contiguous(),
            scales = torch.cat([info.scales for info in info_list], dim=0).contiguous(),
            rotations = torch.cat([info.rotations for info in info_list], dim=0).contiguous()
        )

    return ret

def gather_grad_of_extra_gs(extra_pool:dict, recv_extra_gs_tasks:List[pso.RecvExtraGS], logger:logging.Logger):
    render_info_gard_dict = {}
    for r in recv_extra_gs_tasks:
        assert isinstance(r, pso.RecvExtraGS)
        info:RenderInfoFromGS = extra_pool[(r.src_model_id, r.dst_model_id)]
        grads = []
        for e_id in range(len(info)):
            if info[e_id].grad is not None:
                grads.append(info[e_id].grad)
            else:
                grads.append(torch.zeros_like(info[e_id]))
                logger.info('tag {} element {} of shape {}, find None gard'.format((r.src_model_id, r.dst_model_id), e_id, info[e_id].shape))
        render_info_gard_dict[(r.src_model_id, r.dst_model_id)] = RenderInfoFromGS(*grads)
    return render_info_gard_dict

def training(args, dataset_args, opt, pipe, testing_iterations, ply_iteration, checkpoint_iterations, debug_from, LOGGERS):
    # training-constant states
    RANK, WORLD_SIZE, MAX_LOAD, MAX_BATCH_SIZE = dist.get_rank(), dist.get_world_size(), pipe.max_load, pipe.max_batch_size
    LOCAL_WORLD_SIZE = int(os.environ["LOCAL_WORLD_SIZE"]) # --nproc-per-node specified on torchrun
    NUM_NODE = WORLD_SIZE // LOCAL_WORLD_SIZE
    tb_writer:SummaryWriter = LOGGERS[0]
    logger:logging.Logger = LOGGERS[1]
    scene, BVH_DEPTH = SceneV3(dataset_args, None, shuffle=False), args.bvh_depth
    logger.info('space scale {}'.format(scene.cameras_extent))

    # find newest ply
    ply_iteration = pgc.find_ply_iteration(scene=scene, logger=logger) if ply_iteration <= 0 else ply_iteration
    if ply_iteration <= 0:
        SPACE_RANGE_LOW, SPACE_RANGE_UP, VOXEL_SIZE = pgg.init_grid_dist(scene=scene, SCENE_GRID_SIZE=SCENE_GRID_SIZE)
        path2node_info_dict = None
    else:
        SPACE_RANGE_LOW, SPACE_RANGE_UP, VOXEL_SIZE, path2node_info_dict = pgg.load_grid_dist(scene=scene, ply_iteration=ply_iteration, SCENE_GRID_SIZE=SCENE_GRID_SIZE)
        
    scene_3d_grid = ppu.Grid3DSpace(SPACE_RANGE_LOW, SPACE_RANGE_UP, VOXEL_SIZE)
    logger.info(f"grid parameters: {SPACE_RANGE_LOW}, {SPACE_RANGE_UP}, {VOXEL_SIZE}, {scene_3d_grid.grid_size}")
    
    # training-constant object
    first_iter = 0
    bg_color = [1, 1, 1] if dataset_args.white_background else [0, 0, 0]
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    final_background = torch.tensor(bg_color, dtype=torch.float32, device="cuda") if not opt.random_background else None 
    # handles: 
    zipper_extra_gs, unzipper_grad_extra_gs, space4render_info, unzipper_extra_gs = pgg.build_zip_unzip_for_render_info(dataset_args.sh_degree)
    space4extra_gs = lambda x: space4render_info(length=x.length)

    task_parser = pso.Parser4OpticalFieldSegment(PROCESS_WORLD_SIZE=WORLD_SIZE, GLOBAL_RANK=RANK, logger=logger)
    space_task_parser = pso.SpaceTaskMatcher(PROCESS_WORLD_SIZE=WORLD_SIZE, GLOBAL_RANK=RANK, logger=logger)
    scheduler = pso.BasicScheduler4OpticalFieldSegment(
        logger=logger, tb_writer=tb_writer, batch_isend_irecv_version=0,
        zip_extra_gs=zipper_extra_gs, unzip_extra_gs=unzipper_extra_gs, space_extra_gs=space4extra_gs,
        zip_grad_extra_gs=zipper_extra_gs, unzip_grad_extra_gs=unzipper_grad_extra_gs, space_grad_extra_gs=space4extra_gs,
        zip_render_ret=BoundedGaussianModelGroup.pack_up_render_ret, 
        unzip_render_ret=BoundedGaussianModelGroup.unpack_up_render_ret, 
        space_render_ret=pgc.func_space_for_render,
        zip_grad_render_ret=pgc.build_check_and_pack_up_grad(logger=logger),
        unzip_grad_render_ret=BoundedGaussianModelGroup.unpack_up_grad_of_render_ret,
        space_grad_render_ret=pgc.func_space_for_grad_of_render
    )

    local_func_blender = pgg.build_func_blender(final_background=final_background, logger=logger)

    #  create partition of space or load it 
    if ply_iteration <= 0:
        if RANK == 0:
            if os.path.exists(os.path.join(scene.model_path, "tree_{}.txt".format(RANK))):
                _space_low, _space_up, _grid_size, _path2node_info_dict = ppu.load_BvhTree_on_3DGrid(os.path.join(scene.model_path, "tree_{}.txt".format(RANK)))
                path2bvh_nodes, sorted_leaf_nodes, tree_str = pgg.load_model_division(scene_3d_grid, _path2node_info_dict, logger=logger)
                logger.info(f'load tree\n{tree_str}')
            else:    
                path2bvh_nodes, sorted_leaf_nodes, tree_str = pgg.divide_model_by_load(scene_3d_grid, BVH_DEPTH, load=None, position=scene.point_cloud.points, logger=logger, SPLIT_ORDERS=SPLIT_ORDERS)
                logger.info(f'get tree\n{tree_str}')
            # path2bvh_nodes, sorted_leaf_nodes, tree_str = pgg.divide_model_by_load(scene_3d_grid, BVH_DEPTH, load=None, position=scene.point_cloud.points, logger=logger, SPLIT_ORDERS=SPLIT_ORDERS)
            # logger.info(f'get tree\n{tree_str}')    
            ppu.save_BvhTree_on_3DGrid(path2bvh_nodes, os.path.join(scene.model_path, "tree_{}.txt".format(RANK)))
            if len(sorted_leaf_nodes) != 2**BVH_DEPTH:
                logger.warning(f'bad division! expect {2**BVH_DEPTH} leaf-nodes but get {len(sorted_leaf_nodes)}')             
        else:
            path2bvh_nodes, sorted_leaf_nodes, tree_str = None, None, ''
    else:
        if RANK == 0:
            path2bvh_nodes, sorted_leaf_nodes, tree_str = pgg.load_model_division(scene_3d_grid, path2node_info_dict, logger=logger)
            logger.info(f'load tree\n{tree_str}')
            ppu.save_BvhTree_on_3DGrid(path2bvh_nodes, os.path.join(scene.model_path, "tree_{}.txt".format(RANK)))
            if len(sorted_leaf_nodes) != 2**BVH_DEPTH:
                logger.warning(f'bad division! expect {2**BVH_DEPTH} leaf-nodes but get {len(sorted_leaf_nodes)}') 
        else:
            path2bvh_nodes, sorted_leaf_nodes, tree_str = None, None, ''

    # init GS models with partition
    model_id2box, model_id2rank, local_model_ids = pgg.init_GS_model_division_dist(sorted_leaf_nodes, logger)
    local_model_ids.sort()
    gaussians_group = BoundedGaussianModelGroup(
        sh_degree_list=[dataset_args.sh_degree] * len(local_model_ids),
        range_low_list=[(model_id2box[m].range_low * scene_3d_grid.voxel_size + scene_3d_grid.range_low) for m in local_model_ids],
        range_up_list=[(model_id2box[m].range_up * scene_3d_grid.voxel_size + scene_3d_grid.range_low) for m in local_model_ids],
        device_list=["cuda"] * len(local_model_ids), 
        model_id_list=local_model_ids,
        padding_width=0.0,
        max_size=MAX_SIZE_SINGLE_GS,
    )
    logger.info(gaussians_group.get_info())

    if ply_iteration <= 0:
        for mid in local_model_ids:
            _gau:BoundedGaussianModel = gaussians_group.get_model(mid)
            scene.loadPointCloud2Gaussians(
                _gau,
                range_low=_gau.range_low.cpu().detach().numpy(),
                range_up=_gau.range_up.cpu().detach().numpy(),
                padding_width=0.0
            )
        # init optimizer
        gaussians_group.training_setup(opt)   
    else:
        # load gs and optimizer
        pgc.load_gs_from_ply(opt, gaussians_group, local_model_ids, scene, ply_iteration, logger)
    del scene.point_cloud

    logger.info('models are initialized:' + gaussians_group.get_info())
    render_func = pgc.build_func_render(gaussians_group, pipe, background, logger, need_buffer=False)

    # prepare dataset after the division of model/space, as the blender_order is affected by division
    # train_rlt, evalset_rlt need to be updated after every division of model/space
    train_dataset, eval_test_dataset, eval_train_list, train_rlt, evalset_rlt = init_datasets_dist(
        scene=scene, opt=opt, path2nodes=path2bvh_nodes, sorted_leaf_nodes=sorted_leaf_nodes, NUM_MODEL=len(model_id2box), logger=logger 
    )
    validation_configs = ({'name':'test', 'cameras': eval_test_dataset, 'rlt':evalset_rlt}, 
                        {'name':'train', 'cameras': eval_train_list, 'rlt':train_rlt})
    iter_start, iter_end = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
    ema_loss_for_log, iteration, broadcast_task_cost, accum_model_grad, opti_step_time = 0.0, first_iter, 0.0, 0.0, 0.0
    step, data_cost = 0, 0.0
    train_loader = None
    if ply_iteration > 0:
        iteration = ply_iteration
    logger.info('start from iteration from {}'.format(iteration))
    
    # set up training args
    opt.iterations = len(train_dataset) * opt.epochs
    opt.scaling_lr_max_steps = len(train_dataset) * opt.epochs
    logger.info('set scaling_lr_max_steps {}'.format(opt.scaling_lr_max_steps))
    gaussians_group.training_setup(opt)
    gaussians_group.update_learning_rate(max(0, ply_iteration))

    start_epoch = int(round(iteration/len(train_dataset)))
    logger.info('start from epoch {}'.format(start_epoch))
    NUM_EPOCH = opt.epochs - start_epoch

    last_prune_iteration = -1
    progress_bar = tqdm(range(first_iter, NUM_EPOCH*len(train_dataset)), desc="Training progress") if RANK == 0 else None 

    for _i_epoch in range(NUM_EPOCH):
        i_epoch = _i_epoch + start_epoch
        seed = 0 if DATALOADER_FIX_SEED else i_epoch
        indices:list = get_sampler_indices_dist(train_dataset=train_dataset, seed=seed)
        if train_loader is not None:
            del train_loader

        groups = get_grouped_indices_dist(model2rank=model_id2rank, relation_matrix=train_rlt, shuffled_indices=indices, 
                                          max_task=MAX_LOAD, max_batch=MAX_BATCH_SIZE)   
        grouped_train_dataset = GroupedItems(train_dataset, groups) 
        logger.info("build groups of data items")

        train_loader = DataLoader(grouped_train_dataset, 
                                  batch_size=1, num_workers=2, prefetch_factor=2, drop_last=True,
                                  shuffle=False, collate_fn=SceneV3.get_batch, pin_memory=True, pin_memory_device='cuda')
     
        gaussians_group.update_learning_rate(iteration)  #  - ply_iteration
        for ids_data in train_loader:
            ids, data = ids_data[0] # [(tuple(int), tuple(camera))]
            batch_size = len(data)  # list of Camera/None, batchsize can be dynamic in the future    
            assert batch_size > 0, "get empty group"             
            iter_start.record()
            # update state by batch but not iteration 
            gaussians_group.clean_cached_features()
            gaussians_group.set_SHdegree(iteration//1000)   # safer than oneupSHdegree, GSs and their degree will be reset in dynamic division
            
            logger.info('start of iteration {}'.format(iteration))
            if iteration > 20:
                logger.setLevel(logging.INFO)
       
            cameraUid_taskId_mainRank = torch.zeros((batch_size, 3), dtype=torch.int, device='cuda') 
            packages = torch.zeros((batch_size, *Camera.package_shape()), dtype=torch.float32, device='cuda') 
            data_gpu = [_cmr.to_device('cuda') for _cmr in data]
            if RANK == 0:  
                for i, camera in enumerate(data_gpu):
                    cameraUid_taskId_mainRank[i, 0] = camera.uid
                    cameraUid_taskId_mainRank[i, 1] = iteration + i
                    cameraUid_taskId_mainRank[i, 2] = i % WORLD_SIZE
                    # cameraUid_taskId_mainRank[i, 2] = assign_task2rank_dist(train_rlt[camera.uid, :], i)
                    packages[i] = camera.pack_up(device='cuda')
            dist.broadcast(cameraUid_taskId_mainRank, src=0, async_op=False, group=None)
            logger.info('broadcast cameraUid_taskId_mainRank {}, iteration {}'.format(cameraUid_taskId_mainRank, iteration))
            dist.broadcast(packages, src=0, async_op=False, group=None)
            logger.info('broadcast packages, iteration {}'.format(iteration))
            dist.barrier()   
            # end of broadcast and sync, unpack training-data
         
            cameraUid_taskId_mainRank = cameraUid_taskId_mainRank.cpu()
            view_messages, task_id2cameraUid, uid2camera, task_id2camera = unpack_data(cameraUid_taskId_mainRank, packages, data_gpu, logger)
            mini_message = ' '.join(['(id={}, H={}, W={})'.format(e.id, e.image_height, e.image_width) for e in view_messages])
            _uids = cameraUid_taskId_mainRank[:,0].to(torch.long)
            logger.info("rank {} get task {}, relation \n{}".format(RANK, mini_message, train_rlt[_uids, :]))

            # find extra gs(render_info) that are necesssary to exchange, and exchange
            local_render_info_pool, GLOBAL_SEND_GS_AMOUNT_CPU = pgg.naive_render_info_gather(
                groups=gaussians_group, modelId2Boxes=model_id2box, scene_3d_grid=scene_3d_grid, logger=logger
            )
            send_extra_gs, recv_extra_gs = task_parser.parser_extra_gs_task(GLOBAL_SEND_GS_AMOUNT_CPU, modelId2rank=model_id2rank)
            extra_render_info_pool = scheduler._exchange_extra_gs(local_render_info_pool, send_extra_gs, recv_extra_gs)
            # merge render_info
            merged_info = gather_render_info(local_render_info_pool, extra_render_info_pool, local_model_ids, logger)
            # render_task, send_ret_task, recv_ret_task, main_rank_task
            send_render_ret, recv_render_ret, render_tasks, main_rank_tasks = task_parser.parse_task_tensor(
                model_id2rank, cameraUid_taskId_mainRank[:, [1,2]], train_rlt[_uids, :], view_messages
            )
            # perform render 
            local_render_rets = {} 
            for t in render_tasks:
                local_render_rets[(t.task_id, t.model_id)] = render4renderinfo(
                    viewpoint_camera=task_id2camera[t.task_id], 
                    GS=gaussians_group.get_model(t.model_id), 
                    info=merged_info[t.model_id],
                    pipe=pipe,
                    bg_color=background, 
                )
            # exchange render rets
            extra_render_rets = scheduler._exchange_render_ret(local_render_rets, send_render_ret, recv_render_ret) 
            # use copy of local render_ret to bledner, so that first auto_grad can be faster
            local_render_rets_copy = pgc.make_copy_for_blending(main_rank_tasks, local_render_rets)
            logger.debug("copy {} local render_rets for main rank task".format(len(local_render_rets_copy)))
            images = {}
            for m in main_rank_tasks:
                relation_vector = train_rlt[task_id2cameraUid[m.task_id], :]
                images[(m.task_id, m.rank)] = local_func_blender(local_render_rets_copy, extra_render_rets, m, relation_vector)
            # grad from loss to local/extra render_results  
            loss_main_rank, Ll1_main_rank = torch.tensor(0.0, device='cuda'), torch.tensor(0.0, device='cuda')
            if len(main_rank_tasks) > 0: 
                loss_main_rank, Ll1_main_rank, loss_dict, _t_gather, _t_loss = gather_image_loss(
                    main_rank_tasks=main_rank_tasks, images=images, task_id2camera=task_id2camera, opt=opt, pipe=pipe, logger=logger
                )
                loss_main_rank.backward(retain_graph=True)  
                logger.debug('{} main_rank image rets'.format(len(images))) 
            else:
                logger.debug('no main_rank image ret, also no need to backward')    
            # exchange gradient symmetrically with exchange render-contents
            # gradients shall be registered to the tensors in extra_render_rets, build_check_and_pack_up_grad would precess them

            extra_grad_render_rets = scheduler._exchange_gard_render_ret(extra_render_rets, send_render_ret, recv_render_ret)

            # grad from local/extra render_results to extraRenderInfo and local GS model
            send_local_tensor, extra_gard, main_local_tensor, main_grad = pgc.gather_tensor_grad_of_render_result(send_render_ret, local_render_rets, extra_grad_render_rets, local_render_rets_copy)
            logger.debug('seconda autograd in iteration {}, for {} main local grad, {} extra grad'.format(iteration, len(main_grad), len(extra_gard)))
            if (len(main_grad) + len(extra_gard)) > 0:
                torch.autograd.backward(main_local_tensor + send_local_tensor, main_grad + extra_gard, retain_graph=True) 
            else:
                logger.warning('iteration {}, find no main grad nor extra grad'.format(iteration)) 

            # extra grad of local render_info to local GS model, just ignore the extra max_radii, it would make thing even more complex
            grad_extra_gs = gather_grad_of_extra_gs(extra_render_info_pool, recv_extra_gs, logger)
            extra_grad_of_local_gs = scheduler._exchange_grad_extra_gs(grad_extra_gs, send_extra_gs, recv_extra_gs)
            local_gs_tensor_list, extra_grad_local_gs_list = pgg.gather_tensor_and_grad_of_render_info(send_extra_gs, local_render_info_pool, extra_grad_of_local_gs)
            if len(local_gs_tensor_list) > 0:
                torch.autograd.backward(local_gs_tensor_list, extra_grad_local_gs_list) 
            else:
                logger.info('iteration {}, find no extra grad for render info'.format(iteration)) 

            iter_end.record()
            torch.cuda.synchronize()
            iter_time = iter_start.elapsed_time(iter_end)
            
            with torch.no_grad():
                # torch.cuda.empty_cache()
                # Progress bar
                if dist.get_rank() == 0:
                    ema_loss_for_log = 0.4 * loss_main_rank.item() + 0.6 * ema_loss_for_log
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(batch_size)
                    if iteration >= opt.iterations:
                        progress_bar.close()

                # Log 
                training_report(
                    tb_writer, logger, iteration, Ll1_main_rank, loss_main_rank, batch_size,
                    l1_loss, render_func, model_id2rank, local_func_blender,
                    iter_time, testing_iterations, validation_configs, 
                    scene, gaussians_group, scheduler)
                
                pgc.update_densification_stat(iteration, opt, gaussians_group, local_render_rets, tb_writer, logger)
                pgc.densification(iteration, batch_size, 
                                  (iteration-last_prune_iteration)<SKIP_PRUNE_AFTER_RESET, 
                                  SKIP_CLONE, SKIP_SPLIT,
                                  opt, scene, gaussians_group, tb_writer, logger)
                if pgc.reset_opacity(iteration, batch_size, opt, dataset_args, gaussians_group, tb_writer, logger):
                    last_prune_iteration = iteration

                # Optimizer
                if True: # iteration < opt.iterations:
                    for _gau_name in gaussians_group.all_gaussians:
                        _gaussians = gaussians_group.all_gaussians[_gau_name]
                        _gaussians.optimizer.step()
                        _gaussians.optimizer.zero_grad(set_to_none = True)
 
            # logger.info(f'end of {iteration}')
            iteration += batch_size
            step += 1
            gaussians_group.clean_cached_features()
            # torch.cuda.empty_cache()
            t_iter_end = time.time()

        # after traversing dataset
        scheduler.record_info() 

        with torch.no_grad():
            if (i_epoch % SAVE_INTERVAL_EPOCH == 0) or (i_epoch == (NUM_EPOCH-1)):
                save_GS(iteration=iteration, model_path=scene.model_path, gaussians_group=gaussians_group, path2node=path2bvh_nodes)
            if (i_epoch % EVAL_INTERVAL_EPOCH == 0) or (i_epoch == (NUM_EPOCH-1)):  
                torch.cuda.empty_cache()  
                eval(
                    tb_writer, logger, iteration, pipe,
                    scene_3d_grid, model_id2box, model_id2rank,
                    l1_loss, local_func_blender, background,
                    validation_configs, gaussians_group, scheduler, task_parser)
                torch.cuda.empty_cache()
            if ENABLE_REPARTITION and (i_epoch % REPARTITION_INTERVAL_EPOCH == 0) and (REPARTITION_START_EPOCH<= i_epoch <= REPARTITION_END_EPOCH):
                t0 = time.time()
                logger.info('before resplit\n' + gaussians_group.get_info())
                new_path2bvh_nodes, new_sorted_leaf_nodes, new_tree_str, dst_model2box, dst_model2rank, dst_local_model_ids, dst_id2msgs = pso.eval_load_and_divide_grid_dist(
                    src_gaussians_group=gaussians_group, scr_path2bvh_nodes=path2bvh_nodes,
                    src_model2box=model_id2box, src_model2rank=model_id2rank, src_local_model_ids=local_model_ids,
                    scene_3d_grid=scene_3d_grid, space_task_parser=space_task_parser,
                    scheduler=scheduler, load_dataset=None, BVH_DEPTH=BVH_DEPTH, 
                    SPLIT_ORDERS=SPLIT_ORDERS, MAX_GS_CHANNEL=MAX_GS_CHANNEL, logger=logger
                )
                # re-initialize state/object about model 
                path2bvh_nodes, sorted_leaf_nodes, tree_str = new_path2bvh_nodes, new_sorted_leaf_nodes, new_tree_str
                model_id2box, model_id2rank, local_model_ids = dst_model2box, dst_model2rank, dst_local_model_ids 
                local_model_ids.sort()
                # unpack_up messages to dst_GS_group
                gaussians_group = BoundedGaussianModelGroup(
                    sh_degree_list=[dataset_args.sh_degree] * len(local_model_ids),
                    range_low_list=[(model_id2box[m].range_low * scene_3d_grid.voxel_size + scene_3d_grid.range_low) for m in local_model_ids],
                    range_up_list=[(model_id2box[m].range_up * scene_3d_grid.voxel_size + scene_3d_grid.range_low) for m in local_model_ids],
                    device_list=["cuda"] * len(local_model_ids), 
                    model_id_list=local_model_ids,
                    padding_width=0.0,
                    max_size=MAX_SIZE_SINGLE_GS,
                )
                
                for mid in local_model_ids:
                    _gau:BoundedGaussianModel = gaussians_group.get_model(mid)
                    pkg = torch.cat(dst_id2msgs[mid], dim=0)  
                    logger.info(f'model {mid} gets pkg of size {pkg.shape}')
                    del dst_id2msgs[mid]
                    _gau.un_pack_up(pkg, spatial_lr_scale=scene.cameras_extent, iteration=iteration, step=step, opt=opt)
                    del pkg 
                torch.cuda.empty_cache()
                # gaussians_group.training_setup(opt)   # training_setup was done in un_pack_up
                logger.info('after resplit\n' + gaussians_group.get_info())
                render_func = pgc.build_func_render(gaussians_group, pipe, background, logger, need_buffer=False)
                train_rlt, evalset_rlt = update_relation_matrix_dist(scene, path2bvh_nodes, sorted_leaf_nodes, len(model_id2box), logger)
                validation_configs = ({'name':'test', 'cameras': eval_test_dataset, 'rlt':evalset_rlt}, 
                                    {'name':'train', 'cameras': eval_train_list, 'rlt':train_rlt})
                t1 = time.time()
                # logger.info(f'end resplit after {iteration}, time cost {t1-t0}')
                if tb_writer:
                    tb_writer.add_scalar('resplit/total', t1-t0, iteration) 
                    
    # after all iterations
   

def eval(
        tb_writer, logger:logging.Logger, iteration:int, pipe,
        scene_3d_grid:pgc.Grid3DSpace, model_id2box, modelId2rank:dict,
        l1_loss:callable, local_func_blender:callable, background, 
        validation_configs:dict, gaussians_group: BoundedGaussianModelGroup, 
        scheduler:pso.BasicScheduler4OpticalFieldSegment, task_parser:pso.Parser4OpticalFieldSegment
        ):
    # Report test and samples of training set
    torch.cuda.empty_cache()
    # in eval just gather extra render info fo once

    # burning really starts to think cache the featue is a stupid idea
    # one can easily forget to clean cache !!!!
    gaussians_group.clean_cached_features()
    logger.info('exchange only once is enough for eval')
    local_render_info_pool, GLOBAL_SEND_GS_AMOUNT_CPU = pgg.naive_render_info_gather(
        groups=gaussians_group, modelId2Boxes=model_id2box, scene_3d_grid=scene_3d_grid, logger=logger
    )
    send_extra_gs, recv_extra_gs = task_parser.parser_extra_gs_task(GLOBAL_SEND_GS_AMOUNT_CPU, modelId2rank=modelId2rank)
    extra_render_info_pool = scheduler._exchange_extra_gs(local_render_info_pool, send_extra_gs, recv_extra_gs)
    # merge render_info
    merged_info = gather_render_info(local_render_info_pool, extra_render_info_pool, gaussians_group.model_id_list, logger)
           
    # all rank load the same database, thus the shapes can meet 
    for config in validation_configs:
        if config['cameras'] and len(config['cameras']) > 0:
            l1_test = 0.0
            psnr_test = 0.0
            main_rank_cnt = 0.0
            if isinstance(config['cameras'], DataLoader):
                eval_loader = config['cameras']
            else:
                eval_loader = DataLoader(config['cameras'], 
                                  batch_size=1, num_workers=32, prefetch_factor=2, drop_last=False,
                                  shuffle=False, collate_fn=SceneV3.get_batch, pin_memory=True, pin_memory_device='cuda')  
            for idx, data in enumerate(eval_loader):
                cameraUid_taskId_mainRank = torch.zeros((1, 3), dtype=torch.int, device='cuda') 
                packages = torch.zeros((1, 16, 4), dtype=torch.float32, device='cuda') 
                data_gpu = [_cmr.to_device('cuda') for _cmr in data]
                if RANK == 0:   
                    for i, camera in enumerate(data_gpu):
                        cameraUid_taskId_mainRank[i, 0] = camera.uid
                        cameraUid_taskId_mainRank[i, 1] = i
                        cameraUid_taskId_mainRank[i, 2] = 0
                        packages[i] = camera.pack_up(device='cuda')
                dist.broadcast(cameraUid_taskId_mainRank, src=0, async_op=False)
                dist.broadcast(packages, src=0, async_op=False)
                dist.barrier() 
                logger.info("validation: iteration {}".format(idx))  
                cameraUid_taskId_mainRank = cameraUid_taskId_mainRank.cpu()  
                view_messages, task_id2cameraUid, uid2camera, task_id2camera = unpack_data(cameraUid_taskId_mainRank, packages, data_gpu, logger)  
               
                send_render_ret, recv_render_ret, render_tasks, main_rank_tasks = task_parser.parse_task_tensor(
                    modelId2rank, cameraUid_taskId_mainRank[:, [1,2]], config['rlt'][cameraUid_taskId_mainRank[:,0].long(), :], view_messages
                )
                # perform render 
                local_render_rets = {} 
                for t in render_tasks:
                    local_render_rets[(t.task_id, t.model_id)] = render4renderinfo(
                        viewpoint_camera=task_id2camera[t.task_id], 
                        GS=gaussians_group.get_model(t.model_id), 
                        info=merged_info[t.model_id],
                        pipe=pipe,
                        bg_color=background, 
                    )

                # exchange render rets
                extra_render_rets = scheduler._exchange_render_ret(local_render_rets, send_render_ret, recv_render_ret) 
                # use copy of local render_ret to bledner, so that first auto_grad can be faster
                local_render_rets_copy = pgc.make_copy_for_blending(main_rank_tasks, local_render_rets)
                logger.debug("copy {} local render_rets for main rank task".format(len(local_render_rets_copy)))
             
                images = {}
                for m in main_rank_tasks:
                    relation_vector = config['rlt'][task_id2cameraUid[m.task_id], :]
                    images[(m.task_id, m.rank)] = local_func_blender(local_render_rets, extra_render_rets, m, relation_vector)

                if (len(images)>0) and RANK==0:
                    main_rank_cnt += 1
                    k = list(images.keys())[0]
                    task_id, _ = k
                    image = images[k]['render']
                    image = torch.clamp(image, 0.0, 1.0)
                    ori_camera:Camera = task_id2camera[task_id]
                    gt_image = ori_camera.original_image.to('cuda')

                    # torchvision.utils.save_image(image, os.path.join(scene.save_img_path, '{0:05d}_{1:05d}'.format(idx, iteration) + ".png"))
                    # torchvision.utils.save_image(gt_image, os.path.join(scene.save_gt_path, '{0:05d}'.format(idx) + ".png"))

                    if tb_writer and (idx < 1000) and (idx % 10==0):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(ori_camera.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(ori_camera.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().item()
                    psnr_test += psnr(image, gt_image).mean().item()

            psnr_test /= max(main_rank_cnt, 1) # len(config['cameras'])
            l1_test /= max(main_rank_cnt, 1) # len(config['cameras'])    
            logger.info("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))      
            print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
            if tb_writer:
                tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

    if tb_writer:
        for name in gaussians_group.all_gaussians:
            model_id = name
            _gaussians = gaussians_group.all_gaussians[name]
            tb_writer.add_scalar('total_points_{}'.format(model_id), _gaussians.get_xyz.shape[0], iteration)


def main(rank: int, world_size: int, LOCAL_RANK: int, MASTER_ADDR, MASTER_PORT, train_args):
    mp_setup(rank, world_size, LOCAL_RANK, MASTER_ADDR, MASTER_PORT)
    dataset_args = train_args[1]
    tb_writer, logger = prepare_output_and_logger(dataset_args, train_args)
    grid_setup(train_args, logger)
    try:
        training(*train_args, (tb_writer, logger))
    except:
        tb_str = traceback.format_exc()
        logger.error(tb_str)    
    destroy_process_group()       

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
    parser.add_argument("--ENABLE_TENSORBOARD", action='store_true', default=False)
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
    parser.add_argument("--PERCEPTION_LOSS", action='store_true', default=False)

    args = parser.parse_args(sys.argv[1:])

    # Initialize system state (RNG)
    safe_state(args.quiet, init_gpu=False)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    RANK = int(os.environ["RANK"])
    LOCAL_RANK  = int(os.environ["LOCAL_RANK"])
    MASTER_ADDR = os.environ["MASTER_ADDR"]
    MASTER_PORT = os.environ["MASTER_PORT"]

    args.model_path = os.path.join(args.model_path, 'rank_{}'.format(RANK))
    print("Optimizing " + args.model_path)

    assert WORLD_SIZE <= 2**args.bvh_depth
    train_args = (args, lp.extract(args), op.extract(args), pp.extract(args), 
                  args.test_iterations, 
                  args.ply_iteration,
                  args.checkpoint_iterations, 
                  args.debug_from)
    main(RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT, train_args)

    # All done
    print("\nTraining complete.")