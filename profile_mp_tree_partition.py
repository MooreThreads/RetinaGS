# load model trained on single gpu to profile  

import os, sys
import traceback, uuid, logging, time, shutil, glob
from tqdm import tqdm
import numpy as np
from plyfile import PlyData, PlyElement

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
import torchvision

from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from utils.general_utils import safe_state, is_interval_in_batch, is_point_in_batch
from utils.workload_utils import NaiveWorkloadBalancer, NaiveTimer

import parallel_utils.schedulers.dynamic_space as psd 
import parallel_utils.grid_utils.core as pgc
import parallel_utils.grid_utils.gaussian_grid as pgg
import parallel_utils.grid_utils.utils as ppu

from scene.gaussian_nn_module import BoundedGaussianModel, BoundedGaussianModelGroup
from scene.cameras import Camera, EmptyCamera, ViewMessage
from utils.datasets import CameraListDataset, DatasetRepeater, GroupedItems
from scene.scene4bounded_gaussian import SceneV3
from lpipsPyTorch import LPIPS
from torch.profiler import profile, record_function, ProfilerActivity

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = False
except ImportError:
    TENSORBOARD_FOUND = False

MAX_GS_CHANNEL = 59*3
SPLIT_ORDERS = [0, 1]
ENABLE_REPARTITION = False
SCENE_GRID_SIZE = np.array([2*1024, 2*1024, 1], dtype=int)
EVAL_PSNR_INTERVAL = 8
MAX_SIZE_SINGLE_GS = int(6e7)
Z_NEAR = 0.01
Z_FAR = 1*1000
EVAL_INTERVAL_EPOCH = 5
SAVE_INTERVAL_EPOCH = 1
SAVE_INTERVAL_ITER = 50000
SKIP_PRUNE_AFTER_RESET = 3000
SKIP_SPLIT = False
SKIP_CLONE = False
PERCEPTION_LOSS = False
CNN_IMAGE = None

def grid_setup(train_args, logger:logging.Logger):
    args = train_args[0]
    opt = train_args[2]

    global ENABLE_REPARTITION; ENABLE_REPARTITION = args.ENABLE_REPARTITION
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

torch.multiprocessing.set_sharing_strategy('file_system')

GLOBAL_CKPT_CLEANER = pgc.ckpt_cleaner(max_len=10)

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

    def __getitem__(self, idx):     
        camera: Camera = self.dataset[idx]
        max_depth = Z_FAR

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
                            batch_size=16, num_workers=16, prefetch_factor=16, drop_last=False,
                            shuffle=False, collate_fn=SceneV3.get_batch)
    
    idx_start = 0
    for i, batch in tqdm(enumerate(data_loader)):
        for _data in batch:
            _max_depth, _relation_1_N, camera = _data
            # assert isinstance(camera, (Camera, EmptyCamera))
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
    indices_gpu = torch.tensor(range(N), dtype=torch.int, device='cuda')
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
    
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
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

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")

    current_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) 

    logging.basicConfig(
        format='%(asctime)s-%(filename)s[line:%(lineno)d]-%(levelname)s: %(message)s',
        filemode='w',
        filename=os.path.join(args.model_path, 'rank_{}_{}.txt'.format(dist.get_rank(), current_time))
    )
    logger = logging.getLogger('rank_{}'.format(dist.get_rank()))
    logger.setLevel(logging.INFO)

    return tb_writer, logger

def training_report(
        tb_writer, logger:logging.Logger, iteration:int, Ll1:torch.tensor, loss:torch.tensor, batch_size:int,
        l1_loss:callable, render_func:callable, modelId2rank:dict, local_func_blender:callable,
        elapsed: float, testing_iterations: list, validation_configs:dict, 
        scene: SceneV3, gaussians_group: BoundedGaussianModelGroup, scheduler:psd.BasicSchedulerwithDynamicSpace):
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

def find_latest_ply(scene:SceneV3, logger):
    model_path:str = os.path.join(scene.model_path, '..')
    point_cloud_path = os.path.join(model_path, "point_cloud/iteration_*")
    all_ckpt_dir = glob.glob(point_cloud_path)
    try:
        all_ply_iteration = [int(os.path.basename(path).split('_')[-1]) for path in all_ckpt_dir]
        final_iteration = max(all_ply_iteration)
        logger.info('find ckpt at iteration {}'.format(final_iteration))
    except:
        final_iteration = -1
        logger.info('can not find ckpt')    
    return final_iteration

def load_gs_from_ply(opt, gaussians_group:BoundedGaussianModelGroup, local_model_ids:list, scene:SceneV3, ply_iteration:int, ply_data:PlyData, logger:logging.Logger):
    for mid in local_model_ids:
        _gau:BoundedGaussianModel = gaussians_group.get_model(mid)
        _gau.load_ply(path=None, ply_data_in_memory=ply_data)
        logging.info('model {} load {} gs'.format(mid, _gau._xyz.shape[0]))

        _gau.spatial_lr_scale = scene.cameras_extent 
        logger.info('set _gau.spatial_lr_scale as {}'.format(_gau.spatial_lr_scale))
        _gau.training_setup(opt)    # build optimizer

        # load optimizer state_dict if possibile
        adam_path = os.path.join(os.path.dirname(scene.model_path), "adam_{}.pt".format(mid))
        if os.path.exists(adam_path):
            _gau.optimizer.load_state_dict(torch.load(adam_path))
            logger.info("load from {}".format(adam_path))
        else:
            logger.info('find no adam optimizer')

        _gau.discard_gs_out_range()
        print('model {} has {} gs after discard_gs_out_range'.format(mid, _gau._xyz.shape[0]))
        logger.info('model {} has {} gs after discard_gs_out_range'.format(mid, _gau._xyz.shape[0]))
        
    gaussians_group.set_SHdegree(ply_iteration//1000)   

def training(args, dataset_args, opt, pipe, testing_iterations, ply_iteration, checkpoint_iterations, debug_from, LOGGERS):
    # training-constant states
    RANK, WORLD_SIZE, MAX_LOAD, MAX_BATCH_SIZE = dist.get_rank(), dist.get_world_size(), pipe.max_load, pipe.max_batch_size
    LOCAL_WORLD_SIZE = int(os.environ["LOCAL_WORLD_SIZE"]) # --nproc-per-node specified on torchrun
    NUM_NODE = WORLD_SIZE // LOCAL_WORLD_SIZE
    tb_writer:SummaryWriter = LOGGERS[0]
    logger:logging.Logger = LOGGERS[1]
    # do not search iteration in model_path but find newest pre-trained ply in parent directory
    scene, BVH_DEPTH = SceneV3(dataset_args, None, shuffle=False, load_iteration=None), args.bvh_depth
    ply_iteration = find_latest_ply(scene=scene, logger=logger) 
    # just use scene.info to set up SPACE_RANGE_LOW, SPACE_RANGE_UP, VOXEL_SIZE
    SPACE_RANGE_LOW, SPACE_RANGE_UP, VOXEL_SIZE = pgg.init_grid_dist(scene=scene, SCENE_GRID_SIZE=SCENE_GRID_SIZE)
    path2node_info_dict = None
    # load the pre-trained ply
    _model_path:str = os.path.join(scene.model_path, '..')
    point_cloud_path = os.path.join(_model_path, "point_cloud/iteration_{}/point_cloud.ply".format(ply_iteration))
    plydata = PlyData.read(point_cloud_path)

    scene_3d_grid = ppu.Grid3DSpace(SPACE_RANGE_LOW, SPACE_RANGE_UP, VOXEL_SIZE)
    logger.info(f"grid parameters: {SPACE_RANGE_LOW}, {SPACE_RANGE_UP}, {VOXEL_SIZE}, {scene_3d_grid.grid_size}")
    
    # training-constant object
    first_iter = 0
    bg_color = [1, 1, 1] if dataset_args.white_background else [0, 0, 0]
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    final_background = torch.tensor(bg_color, dtype=torch.float32, device="cuda") if not opt.random_background else None 

    task_parser = psd.TaskParser(PROCESS_WORLD_SIZE=WORLD_SIZE, GLOBAL_RANK=RANK, logger=logger)
    space_task_parser = psd.SpaceTaskMatcher(PROCESS_WORLD_SIZE=WORLD_SIZE, GLOBAL_RANK=RANK, logger=logger)
    scheduler = psd.BasicSchedulerwithDynamicSpace(
        task_parser=task_parser, logger=logger, tb_writer=tb_writer,
        func_pack_up=BoundedGaussianModelGroup.pack_up_render_ret,
        func_grad_pack_up=pgc.build_check_and_pack_up_grad(logger=logger),
        func_unpack_up=BoundedGaussianModelGroup.unpack_up_render_ret,
        func_grad_unpack_up=BoundedGaussianModelGroup.unpack_up_grad_of_render_ret,
        func_space=pgc.func_space_for_render,
        func_grad_space=pgc.func_space_for_grad_of_render,
        batch_isend_irecv_version='0+profiling' # use nccl batched_isend_irecv and add profile.record_function
    )
    local_func_blender = pgg.build_func_blender(final_background=final_background, logger=logger)

    # create partition of space on pre-trained model
    if RANK == 0:
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"])), axis=1)
        logger.info('shape of xyz used in partition of space {}'.format(xyz.shape))
        path2bvh_nodes, sorted_leaf_nodes, tree_str = pgg.divide_model_by_load(
            scene_3d_grid, BVH_DEPTH, load=None, 
            position=xyz,
            logger=logger, SPLIT_ORDERS=SPLIT_ORDERS)
        logger.info(f'get tree\n{tree_str}')
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

    load_gs_from_ply(opt, gaussians_group, local_model_ids, scene, ply_iteration, plydata, logger)
    del scene.point_cloud, plydata

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
    opt.scaling_lr_max_steps = len(train_dataset) * opt.epochs
    logger.info('set scaling_lr_max_steps {}'.format(opt.scaling_lr_max_steps))
    gaussians_group.training_setup(opt)
    gaussians_group.update_learning_rate(max(0, ply_iteration))

    start_epoch = 0
    logger.info('start from epoch {}'.format(start_epoch))
    NUM_EPOCH = opt.epochs - start_epoch

    indices:list = get_sampler_indices_dist(train_dataset=train_dataset, seed=0)
    groups = get_grouped_indices_dist(model2rank=model_id2rank, relation_matrix=train_rlt, shuffled_indices=indices, 
                    max_task=MAX_LOAD, max_batch=MAX_BATCH_SIZE)   
    grouped_train_dataset = GroupedItems(train_dataset, groups) 
    logger.info("build groups of data items")
    train_loader = DataLoader(grouped_train_dataset, 
                        batch_size=1, num_workers=2, prefetch_factor=2, drop_last=True,
                        shuffle=False, collate_fn=SceneV3.get_batch, pin_memory=True, pin_memory_device='cuda')
    gaussians_group.set_SHdegree(3)
    current_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) 

    logger.info('afetr build model memory_allocted {}'.format(torch.cuda.memory_allocated()/(1024**2)))
    # load some data to gpu 
    train_data_list = []
    for _i, ids_data in enumerate(train_loader):
        if True:
            ids, data = ids_data[0]
            data_gpu = [_cmr.to_device('cuda') for _cmr in data]
            train_data_list.append(data_gpu)
    # pop last one manually for garden        
    train_data_list.pop(-1)        
    progress_bar = tqdm(range(first_iter, NUM_EPOCH*len(train_dataset)), desc="Training progress") if RANK == 0 else None         

    logger.info('after load data memory_allocted: {}'.format(torch.cuda.memory_allocated()/(1024**2)))

    dist.barrier()   
    torch.cuda.synchronize(); 
 
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(
            scene.model_path, 'profiler'
        )),
    ) as p:
        t0 = time.time()
        for _i_epoch in range(NUM_EPOCH):
            i_epoch = _i_epoch + start_epoch
            gaussians_group.update_learning_rate(iteration)  #  - ply_iteration

            for data_gpu in train_data_list:
                with record_function("custom_control_info"):
                    batch_size = len(data_gpu)  # list of Camera/None, batchsize can be dynamic in the future    
                    assert batch_size > 0, "get empty group"             
                    iter_start.record()
                    # update state by batch but not iteration 
                    gaussians_group.clean_cached_features()
            
                    # rank 0 samples cameras and broadcast the data 
                    cameraUid_taskId_mainRank = torch.zeros((batch_size, 3), dtype=torch.int, device='cuda') 
                    packages = torch.zeros((batch_size, *Camera.package_shape()), dtype=torch.float32, device='cuda') 
                    if RANK == 0:  
                        for i, camera in enumerate(data_gpu):
                            cameraUid_taskId_mainRank[i, 0] = camera.uid
                            cameraUid_taskId_mainRank[i, 1] = iteration + i
                            cameraUid_taskId_mainRank[i, 2] = i % WORLD_SIZE
                            # cameraUid_taskId_mainRank[i, 2] = assign_task2rank_dist(train_rlt[camera.uid, :], i)
                            packages[i] = camera.pack_up(device='cuda')
                    torch.cuda.synchronize(); 

                with record_function("custom_broadcast_data"):        
                    dist.broadcast(cameraUid_taskId_mainRank, src=0, async_op=False, group=None)
                    logger.info('broadcast cameraUid_taskId_mainRank {}, iteration {}'.format(cameraUid_taskId_mainRank, iteration))
                    dist.broadcast(packages, src=0, async_op=False, group=None)
                    logger.info('broadcast packages, iteration {}'.format(iteration))
                    dist.barrier()   
                    torch.cuda.synchronize(); 

                with record_function('custom_forward'):
                    cameraUid_taskId_mainRank = cameraUid_taskId_mainRank.cpu()
                    view_messages, task_id2cameraUid, uid2camera, task_id2camera = unpack_data(cameraUid_taskId_mainRank, packages, data_gpu, logger)
                
                    # logger.info(task_id2camera)
                    mini_message = ' '.join(['(id={}, H={}, W={})'.format(e.id, e.image_height, e.image_width) for e in view_messages])
                    _uids = cameraUid_taskId_mainRank[:,0].to(torch.long)
                    logger.info("rank {} get task {}, relation \n{}".format(RANK, mini_message, train_rlt[_uids, :]))

                    local_render_rets, send_tasks, recv_tasks, render_tasks, main_rank_tasks = scheduler.render_pass(
                        func_render=render_func,
                        modelId2rank=model_id2rank,
                        _task_main_rank=cameraUid_taskId_mainRank[:, [1,2]],
                        _relation_matrix=train_rlt[_uids, :],
                        views=view_messages)
                    torch.cuda.synchronize(); 

                with record_function('custom_sr_forward'):
                    extra_render_rets = scheduler.comm_pass(local_render_rets, send_tasks, recv_tasks, render_tasks, main_rank_tasks)
                    torch.cuda.synchronize(); 

                # grad from loss to local/extra render_results  
                with record_function("custom_img_backward"):
                    # use copy of local render_ret to bledner, so that first auto_grad can be faster
                    local_render_rets_copy = pgc.make_copy_for_blending(main_rank_tasks, local_render_rets)
                    logger.debug("copy {} local render_rets for main rank task".format(len(local_render_rets_copy)))
                    images = {}
                    for m in main_rank_tasks:
                        relation_vector = train_rlt[task_id2cameraUid[m.task_id], :]
                        images[(m.task_id, m.rank)] = local_func_blender(local_render_rets_copy, extra_render_rets, m, relation_vector)
                        # save rgb images
                        # image = images[(m.task_id, m.rank)]['render']
                        # image = torch.clamp(image, 0.0, 1.0)
                        # ori_camera:Camera = task_id2camera[m.task_id]
                        # torchvision.utils.save_image(image, os.path.join(scene.save_img_path, '{0:05d}_{1:05d}'.format(ori_camera.uid, iteration) + ".png"))

                    loss_main_rank, Ll1_main_rank = torch.tensor(0.0, device='cuda'), torch.tensor(0.0, device='cuda')
                    if len(main_rank_tasks) > 0: 
                        loss_main_rank, Ll1_main_rank, loss_dict, _t_gather, _t_loss = gather_image_loss(
                            main_rank_tasks=main_rank_tasks, images=images, task_id2camera=task_id2camera, opt=opt, pipe=pipe, logger=logger
                        )
                    
                        loss_main_rank.backward(retain_graph=True)  
                    
                        logger.debug('{} main_rank image rets'.format(len(images))) 
                    else:
                        _t_gather, _t_loss, _t_image_grad = 0, 0, 0
                        logger.debug('no main_rank image ret, also no need to backward')    
                    torch.cuda.synchronize(); 

                # exchange gradient symmetrically with exchange render-contents
                # gradients shall be registered to the tensors in extra_render_rets
                
                with record_function("custom_grad_exchange"):
                    grad_from_other_rank = scheduler.backward_pass(extra_render_rets)
                    torch.cuda.synchronize();

                # grad from local/extra render_results to GS.modelParameters
                with record_function('custom_model_backward'):
                    send_tasks = scheduler.saved_for_backward_dict['send_tasks']
                    send_local_tensor, extra_gard, main_local_tensor, main_grad = pgc.gather_tensor_grad_of_render_result(send_tasks, local_render_rets, grad_from_other_rank, local_render_rets_copy)
                    logger.debug('seconda autograd in iteration {}, for {} main local grad, {} extra grad'.format(iteration, len(main_grad), len(extra_gard)))
                    if (len(main_grad) + len(extra_gard)) > 0:
                        torch.autograd.backward(main_local_tensor + send_local_tensor, main_grad + extra_gard) 
                    else:
                        logger.warning('iteration {}, find no main grad nor extra grad'.format(iteration))  
                                
                    with torch.no_grad():
                        # Progress bar
                        if dist.get_rank() == 0:
                            progress_bar.update(batch_size)
                    iteration += batch_size
                    step += 1
                    gaussians_group.clean_cached_features()
                    t_iter_end = time.time()
                    torch.cuda.synchronize()

                p.step()
            # after traversing dataset
        logger.info('profile time {}'.format(time.time() - t0))
    scheduler.record_info() 
    if RANK == 0:
        progress_bar.close()

    logger.info('after training peak_memory_allocted: {}'.format(torch.cuda.max_memory_allocated()/(1024**2)))    
    table_cuda = p.key_averages().table(sort_by="cuda_time_total", row_limit=-1, max_src_column_width=200)   
    table_cpu = p.key_averages().table(sort_by="cpu_time_total", row_limit=-1, max_src_column_width=200) 
    # p.export_chrome_trace(os.path.join(
    #     dataset_args.model_path, 'profile_{}.json'.format(current_time))
    #     )
    with open(os.path.join(
        dataset_args.model_path, 'profile_cuda_{}.txt'.format(current_time)
    ), 'w') as f:
        f.writelines(table_cuda)  

    with open(os.path.join(
        dataset_args.model_path, 'profile_cpu_{}.txt'.format(current_time)
    ), 'w') as f:
        f.writelines(table_cpu)      

    # after all iterations
    logger.info('all time cost in broadcast_task {}'.format(broadcast_task_cost))
    logger.info('all time cost in accum_grad {}'.format(accum_model_grad))
    logger.info('all time cost in optimizer step {}'.format(opti_step_time))
    logger.info('all time cost in prepare data {}'.format(data_cost))

    final_metric = torch.tensor(
        [broadcast_task_cost, accum_model_grad, opti_step_time, data_cost, scheduler.send_recv_forward_cost, scheduler.send_recv_backward_cost],
        dtype=torch.float32, device='cuda')
    dist.all_reduce(final_metric, op=dist.ReduceOp.SUM, group=None, async_op=False)
    logger.info('final_metric {}'.format(final_metric))

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
    # grid parameters
    parser.add_argument("--ENABLE_REPARTITION", action='store_true', default=False)
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
    parser.add_argument("--name", type=str, default = '')

    args = parser.parse_args(sys.argv[1:])

    # Initialize system state (RNG)
    safe_state(args.quiet, init_gpu=False)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    RANK = int(os.environ["RANK"])
    LOCAL_RANK  = int(os.environ["LOCAL_RANK"])
    MASTER_ADDR = os.environ["MASTER_ADDR"]
    MASTER_PORT = os.environ["MASTER_PORT"]

    time_str = time.strftime("%Y_%m_%d_%H_%M", time.localtime()) 
    tag = time_str if len(args.name) <= 0 else args.name
    args.model_path = os.path.join(args.model_path, 'rank_{}_of_{}_{}'.format(RANK, WORLD_SIZE, tag))
    torch.multiprocessing.set_start_method('spawn')
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