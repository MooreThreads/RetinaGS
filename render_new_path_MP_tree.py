import os, sys, math
import traceback, uuid, logging, time, shutil, glob
from tqdm import tqdm
import numpy as np

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
import parallel_utils.schedulers.dynamic_space as psd 
import parallel_utils.grid_utils.core as pgc
import parallel_utils.grid_utils.gaussian_grid as pgg
import parallel_utils.grid_utils.utils as ppu

from scene.gaussian_nn_module import BoundedGaussianModel, BoundedGaussianModelGroup
from scene.cameras import Camera, EmptyCamera, ViewMessage
from utils.datasets import CameraListDataset, PartOfDataset, EmptyCameraListDataset
from scene.scene4bounded_gaussian import SceneV3

from scipy.interpolate import splprep, splev
from scipy.spatial.transform import Rotation, RotationSpline
from scipy.special import comb

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# todo: add arg parser for GLOBAL_VARIABLE
MAX_GS_CHANNEL = 59*3
SPLIT_ORDERS = [0, 1]
SCENE_GRID_SIZE = np.array([2*1024, 2*1024, 1], dtype=int)
Z_NEAR = 0.01
Z_FAR = 10*1000
MAX_SIZE_SINGLE_GS = 1e7
torch.multiprocessing.set_sharing_strategy('file_system')
GLOBAL_LOGGER:logging.Logger = None

def build_new_path():
    file = '/jfs/shengyi.chen/HT/Predict/MatrixCity/aerial_street_track/track_1.npy'
    WIDTH = 1920
    HEIGHT = 1080
    # WIDTH = 1920/4
    # HEIGHT = 1080/4
    
    # FoVx = math.atan(WIDTH/2/2317.6449482429634)*2
    # FoVy = math.atan(HEIGHT/2/2317.6449482429634)*2
    Fov_angle_y = 70.320092
    
    FoVy = (Fov_angle_y / 180.0) * math.pi
    focal_y = (HEIGHT / 2) / math.tan(FoVy / 2)
    focal_x = focal_y
    FoVx  = math.atan(WIDTH/2/focal_x)*2
    
    # Poses is in W2C
    poses = np.load(file)
    
    for i in range(0, poses.shape[0]):
        # 手系变换（Pose是OpenGL的在线浏览器导出，但是代码是根据COLMAP格式写的，change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)）
        poses[i, 0:2, :] *= -1
        # scene.cameras.EmptyCamera的要求R是C2W，T是W2C（R is stored transposed due to 'glm' in CUDA code）
        pose_inv = np.linalg.inv(poses[i])
        poses[i, 0:3, 0:3] = pose_inv[0:3, 0:3]

    # v4，引入人类经验的三阶/四节点贝塞尔曲线
    # from scipy.interpolate import splprep, splev
    # from scipy.spatial.transform import Rotation, RotationSpline
    # from scipy.special import comb
    
    def ratation_compute_multi(rotations, rs, start, end):        
        rots = Rotation.from_matrix(rotations)
        spline = RotationSpline(np.linspace(0, 1, rotations.shape[0]), rots)  
        N = end - start
        ratios = np.linspace(0, 1, N)        
        for i in range(0, N):
            rs[start + i, :, :] = spline(ratios[i]).as_matrix()
            
    def find_nearest_rotation(position, poses):
        position_in_original = poses[:, :3, 3]
        distances = np.linalg.norm(position_in_original - position, axis=1)
        index = np.argmin(distances)

        return poses[index, :3, :3]
    
    # 先验
    N_curve = 3
    curve_rank = 5 # 指节点数
    step_curve = 1
    N_frames_per_curve = 120   

    # N_curve = 3
    # curve_rank = 5 # 指节点数
    # step_curve = 1
    # N_frames_per_curve = 120   
    
    # 生成
    rs = np.zeros((N_frames_per_curve * N_curve + 1, 3, 3))
    ps = np.zeros((N_frames_per_curve * N_curve + 1, 3))    
    
    # 计算平移，贝塞尔曲线
    for i_curve in range(N_curve):       
              
        # 控制点，左闭右闭，保证重合（需要N_curve*(curve_rank - 1) + 1个关键点）
        start_in_original = i_curve * step_curve * (curve_rank - 1)
        end_in_original =  (i_curve + 1) * step_curve * (curve_rank - 1)        
        order_in_original = range(start_in_original, end_in_original + 1, step_curve)                 
        
        # 计算点
        control_points = np.zeros((curve_rank, 4, 4))
        for i in range(0, curve_rank):
            control_points[i, :, :] = poses[order_in_original[i], :, :]
        # 将点分解为单独的坐标数组
        x = control_points[:, 0, 3]
        y = control_points[:, 1, 3]
        z = control_points[:, 2, 3]
        # 用splprep进行插值
        # tck, u = splprep([x, y, z], s=0, k=curve_rank-1)
        tck, u = splprep([x, y, z], k=curve_rank-1)
        new_points = splev(np.linspace(0, 1, N_frames_per_curve + 1), tck)
        # new_points现在包含插值点的x，y和z坐标
        new_x = new_points[0]
        new_y = new_points[1]
        new_z = new_points[2]
        # 更新值
        start_in_new = i_curve * N_frames_per_curve
        end_in_new = (i_curve + 1) * N_frames_per_curve + 1
        ps[start_in_new:end_in_new, 0] = new_x
        ps[start_in_new:end_in_new, 1] = new_y
        ps[start_in_new:end_in_new, 2] = new_z
        
        # # 计算旋转 - 区间插值
        # step_rataion = 40
        # section = range(start_in_new, end_in_new, step_rataion)
        # section_ratation = np.zeros((len(section), 3, 3))
        # for i_section in range(len(section)):
        #     section_ratation[i_section] = find_nearest_rotation(ps[section[i_section], :], poses)          
        # # 区间连续插值
        # ratation_compute_multi(section_ratation, rs, start_in_new, end_in_new)
    
    # 计算旋转，全部插值
    section_ratation = poses[:, 0:3, 0:3]
    ratation_compute_multi(section_ratation, rs, 0, N_frames_per_curve * N_curve + 1)
        
    # 变为程序可读
    P = []    
    for i in range(0, len(rs)):
        pose = np.eye(4, dtype=np.float32)        
        pose[:3, :3] = rs[i]
        pose[0, 3] = ps[i, 0]
        pose[1, 3] = ps[i, 1]
        pose[2, 3] = ps[i, 2]
        P.append(pose)
    
    ret = []  
    for pose in P:
        ret.append(
            EmptyCamera(
                colmap_id=0,
                R=pose[:3, :3],
                T=pose[:3, 3],
                FoVx=FoVx,
                FoVy=FoVy,
                width_height=(WIDTH, HEIGHT),
                gt_alpha_mask=None,
                image_name='',
                uid=0,
                data_device='cpu'
            )
        )
    # return ret
    return EmptyCameraListDataset(ret)

def find_views_on_split_plane(example_views:CameraListDataset, split_dim:int=0, split_value:float=0):
    camera_x_align_world_axis = [
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]),
        np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]),
    ]

    sample_rate = 6
    fake_views = []
    for i in range(0, len(example_views), sample_rate):
        # scene.cameras.EmptyCamera的要求R是C2W，T是W2C（R is stored transposed due to 'glm' in CUDA code）
        camera:EmptyCamera = example_views.get_empty_item(i)
        x_axis = np.array([1, 0, 0])
        y_axis = np.array([0, 1, 0])

        for x_offset in [-0.5, 0, 0.5]:
            for y_angle in [-np.pi/6, 0 , np.pi/6]:
                for x_angle in [0]:
                    fake_camera_center:np.ndarray = camera.camera_center.cpu().numpy()
                    fake_camera_center[split_dim] = split_value + x_offset

                    fake_r_y: Rotation = Rotation.from_rotvec(y_axis*y_angle)
                    Ry_Camera:np.ndarray = fake_r_y.as_matrix()
                    fake_r_x: Rotation = Rotation.from_rotvec(x_axis*x_angle)
                    Rx_Camera:np.ndarray = fake_r_x.as_matrix()
                    R_Camera = np.matmul(Rx_Camera, Ry_Camera)

                    fake_C2W_R:np.ndarray = np.matmul(camera_x_align_world_axis[split_dim], R_Camera)
                    fake_W2T_T:np.ndarray = -np.matmul(fake_C2W_R.T, np.reshape(fake_camera_center, (3,1)))
                    # GLOBAL_LOGGER.info('example {} {}'.format(camera.image_width, camera.image_height))
                    fake_views.append(
                        EmptyCamera(
                            colmap_id=0,
                            R=fake_C2W_R,
                            T=fake_W2T_T.reshape(3),
                            FoVx=camera.FoVx,
                            FoVy=camera.FoVy,
                            width_height=(camera.image_width, camera.image_height),
                            gt_alpha_mask=None,
                            image_name='',
                            uid=0,
                            data_device='cpu'
                        )
                    )

    return fake_views

def find_views_on_split_plane_dist(path2bvh_nodes:dict, example_views:CameraListDataset):
    RANK, WORLD_SIZE = dist.get_rank(), dist.get_world_size()
    split_dim_tensor:torch.Tensor = torch.zeros((1), device='cuda', dtype=torch.int)
    split_value_tensor:torch.Tensor = torch.zeros((1), device='cuda', dtype=torch.float)

    if RANK == 0:
        root:ppu.BvhTreeNodeon3DGrid = path2bvh_nodes['']
        split_dim_tensor[0] = root.split_dim
        split_value_tensor[0] = root.get_split_position_in_world()
    dist.broadcast(split_dim_tensor, src=0, group=None, async_op=False)
    dist.broadcast(split_value_tensor, src=0, group=None, async_op=False) 

    ret = find_views_on_split_plane(example_views, split_dim=split_dim_tensor[0], split_value=split_value_tensor[0])
    return EmptyCameraListDataset(ret)

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
            assert isinstance(camera, (Camera, EmptyCamera))
            assert camera.uid == idx_start
            complete_relation[idx_start, :] = _relation_1_N
            if i%100 == 0:
                logger.info("{}, {}, {}, {}".format(camera.image_height, camera.image_width, camera.uid, _max_depth))
            idx_start += 1
    # if a row is all -1, set it to all 0
    all_minus_one = (complete_relation.sum(axis=-1) == -NUM_MODEL)  
    complete_relation[all_minus_one, :] = 0
    logger.info('find {} mismatching samples, set their relation to all 0'.format(np.sum(all_minus_one)))

    return complete_relation 

def init_datasets_dist(scene:SceneV3, opt, path2nodes:dict, sorted_leaf_nodes:list, NUM_MODEL:int, logger:logging.Logger):
    RANK, WORLD_SIZE = dist.get_rank(), dist.get_world_size()
    # opt is global args from parser
    # train_dataset: CameraListDataset = scene.getTrainCameras() 
    if opt.watch_split_plane:
        example_dataset: CameraListDataset = scene.getTrainCameras()
        train_dataset = find_views_on_split_plane_dist(path2bvh_nodes=path2nodes, example_views=example_dataset)
    else:    
        train_dataset:CameraListDataset = build_new_path()
    eval_test_dataset: CameraListDataset = []
    eval_train_list = train_dataset
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
        filename=os.path.join(args.model_path, 'rank_rendering{}_{}.txt'.format(dist.get_rank(), current_time))
    )
    logger = logging.getLogger('rank_rendering_{}'.format(dist.get_rank()))
    logger.setLevel(logging.INFO)

    return tb_writer, logger

def unpack_data(cameraUid_taskId_mainRank:torch.Tensor, packages:torch.Tensor, cams_gpu:list, logger:logging.Logger): 
    view_messages = [ViewMessage(e, id=int(_id)) for e, _id in zip(packages, cameraUid_taskId_mainRank[:,1])]  
    task_id2cameraUid, uid2camera, task_id2camera = {}, {}, {}
    for camera in cams_gpu:
        # assert isinstance(camera, Camera)
        uid2camera[camera.uid] = camera 
    logger.info(uid2camera)    
    for row in cameraUid_taskId_mainRank:
        uid, tid, mid = int(row[0]), int(row[1]), int(row[2])
        task_id2cameraUid[tid] = uid
        task_id2camera[tid] = uid2camera[uid]
        
    return view_messages, task_id2cameraUid, uid2camera, task_id2camera

def rendering(args, dataset_args, opt, pipe, testing_iterations, ply_iteration, checkpoint_iterations, debug_from, LOGGERS):
    # training-constant states
    RANK, WORLD_SIZE, MAX_LOAD, MAX_BATCH_SIZE = dist.get_rank(), dist.get_world_size(), pipe.max_load, pipe.max_batch_size
    LOCAL_WORLD_SIZE = int(os.environ["LOCAL_WORLD_SIZE"]) # --nproc-per-node specified on torchrun
    NUM_NODE = WORLD_SIZE // LOCAL_WORLD_SIZE
    tb_writer:SummaryWriter = LOGGERS[0]
    logger:logging.Logger = LOGGERS[1]
    scene, BVH_DEPTH = SceneV3(dataset_args, None, shuffle=False), args.bvh_depth
    time_str = time.strftime("%Y_%m_%d_%H_%M", time.localtime()) 
    scene.save_img_path = os.path.join(scene.model_path, 'img_{}'.format(time_str))
    os.makedirs(scene.save_img_path, exist_ok=True)

    # find newest ply
    ply_iteration = pgc.find_ply_iteration(scene=scene, logger=logger) if ply_iteration <= 0 else ply_iteration
    assert ply_iteration > 0, 'can not find ply'
    SPACE_RANGE_LOW, SPACE_RANGE_UP, VOXEL_SIZE, path2node_info_dict = pgg.load_grid_dist(scene=scene, ply_iteration=ply_iteration, SCENE_GRID_SIZE=SCENE_GRID_SIZE)
        
    scene_3d_grid = ppu.Grid3DSpace(SPACE_RANGE_LOW, SPACE_RANGE_UP, VOXEL_SIZE)
    logger.info(f"grid parameters: {SPACE_RANGE_LOW}, {SPACE_RANGE_UP}, {VOXEL_SIZE}, {scene_3d_grid.grid_size}")
    
    # training-constant object
    first_iter = 0
    bg_color = [1, 1, 1] if dataset_args.white_background else [0, 0, 0]
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    final_background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")    

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
        batch_isend_irecv_version=0 # use nccl batched_isend_irecv 
    )
    local_func_blender = pgg.build_func_blender(final_background=final_background, logger=logger)

    # load partition of space 
    if RANK == 0:
        path2bvh_nodes, sorted_leaf_nodes, tree_str = pgg.load_model_division(scene_3d_grid, path2node_info_dict, logger=logger)
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
    pgc.load_gs_from_ply(opt, gaussians_group, local_model_ids, scene, ply_iteration, logger)
    gaussians_group.set_SHdegree(ply_iteration//1000) 
    del scene.point_cloud

    logger.info('models are initialized:' + gaussians_group.get_info())
    render_func = pgc.build_func_render(gaussians_group, pipe, background, logger, need_buffer=False)

    # prepare dataset after the division of model/space, as the blender_order is affected by division
    # train_rlt, evalset_rlt need to be updated after every division of model/space
    train_dataset, eval_test_dataset, eval_train_list, train_rlt, evalset_rlt = init_datasets_dist(
        scene=scene, opt=args, path2nodes=path2bvh_nodes, sorted_leaf_nodes=sorted_leaf_nodes, NUM_MODEL=len(model_id2box), logger=logger 
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
   
    partOftrain = PartOfDataset(train_dataset, empty=False, start=0, end=None)
    train_loader = DataLoader(partOftrain, 
                                batch_size=1, num_workers=8, prefetch_factor=2, drop_last=False,
                                shuffle=False, collate_fn=SceneV3.get_batch)
    progress_bar = tqdm(range(first_iter, len(partOftrain)), desc="Training progress") if RANK == 0 else None 

    step, t_iter_end = 0, time.time()
    for ids_data in train_loader:
        step += 1
        t_data = time.time()
        if tb_writer:
            tb_writer.add_scalar('cnt/loader_iter', t_data-t_iter_end, iteration) 
        logger.info('loader iter time {}'.format(t_data-t_iter_end))

        data = ids_data
        batch_size = len(data)  # list of Camera/None, batchsize can be dynamic in the future    
        assert batch_size > 0, "get empty group"             
        iter_start.record()
      
        # rank 0 samples cameras and broadcast the data 
        t0 = time.time()
        cameraUid_taskId_mainRank = torch.zeros((batch_size, 3), dtype=torch.int, device='cuda') 
        packages = torch.zeros((batch_size, *Camera.package_shape()), dtype=torch.float32, device='cuda') 
        data_gpu = [_cmr.to_device('cuda') for _cmr in data]
        if RANK == 0:  
            for i, camera in enumerate(data_gpu):
                cameraUid_taskId_mainRank[i, 0] = camera.uid
                cameraUid_taskId_mainRank[i, 1] = iteration + i
                cameraUid_taskId_mainRank[i, 2] = 0
                packages[i] = camera.pack_up(device='cuda')
        dist.broadcast(cameraUid_taskId_mainRank, src=0, async_op=False, group=None)
        logger.info('broadcast cameraUid_taskId_mainRank {}, iteration {}'.format(cameraUid_taskId_mainRank, iteration))
        dist.broadcast(packages, src=0, async_op=False, group=None)
        logger.info('broadcast packages, iteration {}'.format(iteration))
        dist.barrier()   
        t1 = time.time()
        broadcast_task_cost += (t1-t0)
        if tb_writer:
            tb_writer.add_scalar('cnt/broadcast_sync', t1-t0, iteration) 
        # end of broadcast and sync, unpack training-data
        t0 = time.time()
        cameraUid_taskId_mainRank = cameraUid_taskId_mainRank.cpu()
        view_messages, task_id2cameraUid, uid2camera, task_id2camera = unpack_data(cameraUid_taskId_mainRank, packages, data_gpu, logger)
        t1 = time.time()
        data_cost += (t1-t0)
        if tb_writer:
            tb_writer.add_scalar('cnt/prepare_data', t1-t0, iteration) 

        # logger.info(task_id2camera)
        mini_message = ' '.join(['(id={}, H={}, W={})'.format(e.id, e.image_height, e.image_width) for e in view_messages])
        _uids = cameraUid_taskId_mainRank[:,0].to(torch.long)
        logger.info("rank {} get task {}, relation \n{}".format(RANK, mini_message, train_rlt[_uids, :]))
        # default debug_from is -1, never debug
        if debug_from in range(iteration, iteration + batch_size):
            pipe.debug = True

        t0 = time.time()    
        # render contents and exchange them between ranks
        main_rank_tasks, local_render_rets, extra_render_rets = scheduler.forward_pass(
            func_render=render_func,
            modelId2rank=model_id2rank,
            _task_main_rank=cameraUid_taskId_mainRank[:, [1,2]],
            _relation_matrix=train_rlt[_uids, :],
            views=view_messages)
        t1 = time.time()
        if tb_writer:
            tb_writer.add_scalar('cnt/forward_pass', t1-t0, iteration)

        t0 = time.time() 
        # use copy of local render_ret to bledner, so that first auto_grad can be faster
        local_render_rets_copy = pgc.make_copy_for_blending(main_rank_tasks, local_render_rets)
        logger.debug("copy {} local render_rets for main rank task".format(len(local_render_rets_copy)))
        images = {}
        for m in main_rank_tasks:
            relation_vector = train_rlt[task_id2cameraUid[m.task_id], :]
            images[(m.task_id, m.rank)] = local_func_blender(local_render_rets_copy, extra_render_rets, m, relation_vector)
        t1 = time.time()
        if tb_writer:
            tb_writer.add_scalar('cnt/blender', t1-t0, iteration)

        if RANK==0:
            for k in local_render_rets:
                task_id, model_id = k
                ori_camera:Camera = task_id2camera[task_id]
                idx = ori_camera.uid
                torchvision.utils.save_image(
                    local_render_rets[k]['render'], 
                    os.path.join(scene.save_img_path, '{0:05d}_{1:05d}_{2}_{3}'.format(idx, iteration, task_id, model_id) + ".png")
                    )
            for k in extra_render_rets:
                task_id, model_id = k
                ori_camera:Camera = task_id2camera[task_id]
                idx = ori_camera.uid
                torchvision.utils.save_image(
                    extra_render_rets[k]['render'], 
                    os.path.join(scene.save_img_path, '{0:05d}_{1:05d}_{2}_{3}'.format(idx, iteration, task_id, model_id) + ".png")
                )

        if (len(images)>0) and RANK==0:
            k = list(images.keys())[0]
            task_id, _ = k
            image = images[k]['render']
            image = torch.clamp(image, 0.0, 1.0)
            ori_camera:Camera = task_id2camera[task_id]
            # gt_image = ori_camera.original_image.to('cuda')
            torchvision.utils.save_image(image, os.path.join(scene.save_img_path, '{0:05d}_{1:05d}'.format(ori_camera.uid, iteration) + ".png"))
            # torchvision.utils.save_image(gt_image, os.path.join(scene.save_gt_path, '{0:05d}'.format(idx) + ".png"))    

        iter_end.record()
        torch.cuda.synchronize()
        iter_time = iter_start.elapsed_time(iter_end)
        
        # Progress bar
        if dist.get_rank() == 0:
            progress_bar.update(batch_size)
            if iteration >= opt.iterations:
                progress_bar.close()
                
        step += 1
        torch.cuda.empty_cache()
        t_iter_end = time.time()

    # after traversing dataset
    scheduler.record_info() 
             
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
    global GLOBAL_LOGGER
    GLOBAL_LOGGER = logger
    with torch.no_grad():
        try:
            rendering(*train_args, (tb_writer, logger))
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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[400, 1500, 7_000, 15_000, 20_000, 30_000])
    parser.add_argument("--ply_iteration", type=int, default=-1)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--watch_split_plane", action="store_true")
    parser.add_argument("--bvh_depth", type=int, default=2, help='num_model_would be 2**bvh_depth')

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