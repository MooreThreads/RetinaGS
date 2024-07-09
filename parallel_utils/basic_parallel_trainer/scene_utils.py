import os, sys
import traceback, uuid, logging, time, shutil, glob
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from utils.general_utils import is_interval_in_batch, is_point_in_batch
from utils.workload_utils import NaiveWorkloadBalancer
from utils.datasets import DatasetRepeater, GroupedItems, CameraListDataset
import parallel_utils.grid_utils.utils as pgu

from scene.gaussian_nn_module import BoundedGaussianModel, BoundedGaussianModelGroup
from scene.cameras import Camera, EmptyCamera
from scene.scene4bounded_gaussian import SceneV3

from typing import List, Dict, Tuple

def init_grid_dist(scene: SceneV3, SCENE_GRID_SIZE: List[int]):
    RANK = dist.get_rank()
    grid_tensor = torch.zeros((3, 3), dtype=torch.float32, device='cuda')
    if RANK == 0:
        _SPACE_RANGE_LOW, _SPACE_RANGE_UP = scene.point_cloud.points.min(axis=0, keepdims=False), scene.point_cloud.points.max(axis=0, keepdims=False)
        _VOXEL_SIZE = (_SPACE_RANGE_UP - _SPACE_RANGE_LOW)/SCENE_GRID_SIZE - 1e-7
        grid_np = np.array([_SPACE_RANGE_LOW, _SPACE_RANGE_UP, _VOXEL_SIZE])
        grid_tensor = torch.tensor(grid_np, dtype=torch.float32, device='cuda')

    dist.broadcast(grid_tensor, src=0, group=None, async_op=False)
    grid_numpy = grid_tensor.cpu().numpy()
    SPACE_RANGE_LOW, SPACE_RANGE_UP, VOXEL_SIZE = grid_numpy[0], grid_numpy[1], grid_numpy[2]
    scene_3d_grid = pgu.Grid3DSpace(SPACE_RANGE_LOW, SPACE_RANGE_UP, VOXEL_SIZE)
    return scene_3d_grid

def init_BvhTreeNode_Rank0(scene_3d_grid:pgu.Grid3DSpace, bvh_depth:int, load:np.ndarray, position:np.ndarray, SPLIT_ORDERS:list, pre_load_grid:np.ndarray, conflict:Dict[str, pgu.BvhTreeNodeon3DGrid], logger:logging.Logger):
    """
    load: (N,), position (N, 3)
    pre_load_grid: shape of scene_3d_grid.load_cnt
    SPLIT_ORDERS is a list of [0/1/2] which specify the splited dimensions of bvh tree 
    """
    scene_3d_grid.clean_load()
    if pre_load_grid is not None:
        scene_3d_grid.accum_load_grid(load_np=pre_load_grid)
    if position is not None:
        load = np.ones((position.shape[0],), dtype=float) if load is None else load
        scene_3d_grid.accum_load(load_np=load, position=position)

    path2bvh_nodes:Dict[str, pgu.BvhTreeNodeon3DGrid] = pgu.build_BvhTree_on_3DGrid(scene_3d_grid, max_depth=bvh_depth, split_orders=SPLIT_ORDERS, example_path2bvh_nodes=conflict)
    # find leaf-nodes as model space
    sorted_leaf_nodes:List[pgu.BvhTreeNodeon3DGrid] = []
    for path in path2bvh_nodes:
        node = path2bvh_nodes[path]
        assert isinstance(node, pgu.BvhTreeNodeon3DGrid)
        if node.is_leaf:
            sorted_leaf_nodes.append(node)
    sorted_leaf_nodes.sort(key=lambda node:node.path)
    # path2bvh_nodes[''] is the root node
    tree_str = ''.join(pgu.format_BvhTree_on_3DGrid(path2bvh_nodes['']))
    logger.info(f'build tree\n{tree_str}')
    print(f'build tree\n{tree_str}')
    # check size
    size_list = [node.size for node in sorted_leaf_nodes]
    assert sum(size_list) == path2bvh_nodes[''].size, "size mismatch"
    if len(sorted_leaf_nodes) != 2**bvh_depth:
        logger.warning(f'bad division! expect {2**bvh_depth} leaf-nodes but get {len(sorted_leaf_nodes)}')   

    return path2bvh_nodes, sorted_leaf_nodes

def init_BvhTree_on_3DGrid_dist(scene:SceneV3, SCENE_GRID_SIZE:List[int], bvh_depth:int, load:np.ndarray, position:np.ndarray, SPLIT_ORDERS:list, pre_load_grid:np.ndarray, conflict:Dict[str, pgu.BvhTreeNodeon3DGrid], logger:logging.Logger):
    RANK = dist.get_rank()
    scene_3d_grid = init_grid_dist(scene, SCENE_GRID_SIZE)
    path2bvh_nodes, sorted_leaf_nodes = None, None
    if RANK == 0:
        path2bvh_nodes, sorted_leaf_nodes = init_BvhTreeNode_Rank0(scene_3d_grid, bvh_depth, load, position, SPLIT_ORDERS, pre_load_grid, conflict, logger)
    
    return scene_3d_grid, path2bvh_nodes, sorted_leaf_nodes

def resplit_BvhTree_on_3DGrid_dist(scene_3d_grid:pgu.Grid3DSpace, bvh_depth:int, load:np.ndarray, position:np.ndarray, SPLIT_ORDERS:list, pre_load_grid:np.ndarray, conflict:Dict[str, pgu.BvhTreeNodeon3DGrid], logger:logging.Logger):
    RANK = dist.get_rank()
    path2bvh_nodes, sorted_leaf_nodes = None, None
    if RANK == 0:
        path2bvh_nodes, sorted_leaf_nodes = init_BvhTreeNode_Rank0(scene_3d_grid, bvh_depth, load, position, SPLIT_ORDERS, pre_load_grid, conflict, logger)
    
    return scene_3d_grid, path2bvh_nodes, sorted_leaf_nodes
        
class CkptCleaner():
    def __init__(self, max_len=10) -> None:
        self.folder_buffer = []
        self.max_len = max_len

    def _clean(self):
        if len(self.folder_buffer) > self.max_len:
            trash = self.folder_buffer.pop(0)
            if os.path.isdir(trash):
                shutil.rmtree(trash, ignore_errors=True)
            else:    
                os.remove(trash)
            print('rm {}'.format(trash))

    def add(self, folder):
        self.folder_buffer.append(folder)
        self._clean()

def save_BvhTree_on_3DGrid(cleaner:CkptCleaner, iteration:int, model_path:str, gaussians_group:BoundedGaussianModelGroup, path2node:dict):    
    RANK = dist.get_rank()
    print("\n[rank {}, ITER {}] Saving Gaussians".format(RANK, iteration))
    point_cloud_path = os.path.join(model_path, "point_cloud/iteration_{}".format(iteration))
    cleaner.add(point_cloud_path)
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
        pgu.save_BvhTree_on_3DGrid(path2node, os.path.join(point_cloud_path, "tree_{}.txt".format(RANK)))
    return point_cloud_path

def load_BvhTree_on_3DGrid_dist(scene:SceneV3, ply_iteration:int, SCENE_GRID_SIZE:List[int], logger:logging.Logger):
    RANK = dist.get_rank()
    model_path = scene.model_path
    point_cloud_path = os.path.join(model_path, "point_cloud/iteration_{}".format(ply_iteration))
    tree_file_path = os.path.join(point_cloud_path, "tree_{}.txt".format(RANK))

    grid_tensor = torch.zeros((3, 3), dtype=torch.float32, device='cuda')
    path2node_info_dict = None
    if RANK == 0:
        _SPACE_RANGE_LOW, _SPACE_RANGE_UP, SAVED_GRID_SIZE, path2node_info_dict = pgu.load_BvhTree_on_3DGrid(tree_file_path)
        if SAVED_GRID_SIZE[0]==SCENE_GRID_SIZE[0] and SAVED_GRID_SIZE[1]==SCENE_GRID_SIZE[1] and SAVED_GRID_SIZE[2]==SCENE_GRID_SIZE[2]:
            logger.info('load from {}'.format(tree_file_path))
        else:
            logger.info('SAVED_GRID_SIZE is {}, mismatch {}'.format(SAVED_GRID_SIZE, SCENE_GRID_SIZE))
            raise RuntimeError('SAVED_GRID_SIZE is {}, mismatch {}'.format(SAVED_GRID_SIZE, SCENE_GRID_SIZE))
        _VOXEL_SIZE = (_SPACE_RANGE_UP - _SPACE_RANGE_LOW)/SCENE_GRID_SIZE - 1e-7
        grid_np = np.array([_SPACE_RANGE_LOW, _SPACE_RANGE_UP, _VOXEL_SIZE])
        grid_tensor = torch.tensor(grid_np, dtype=torch.float32, device='cuda')

    dist.broadcast(grid_tensor, src=0, group=None, async_op=False)
    grid_numpy = grid_tensor.cpu().numpy()
    SPACE_RANGE_LOW, SPACE_RANGE_UP, VOXEL_SIZE = grid_numpy[0], grid_numpy[1], grid_numpy[2]
    scene_3d_grid = pgu.Grid3DSpace(SPACE_RANGE_LOW, SPACE_RANGE_UP, VOXEL_SIZE)
    logger.info(f"grid parameters: {SPACE_RANGE_LOW}, {SPACE_RANGE_UP}, {VOXEL_SIZE}, {scene_3d_grid.grid_size}")

    path2bvh_nodes:Dict[str, pgu.BvhTreeNodeon3DGrid] = None
    sorted_leaf_nodes:List[pgu.BvhTreeNodeon3DGrid] = None
    if RANK == 0:
        sorted_leaf_nodes = []
        path2bvh_nodes = pgu.build_BvhTree_on_3DGrid_with_info(path2node_info_dict, scene_3d_grid)
        for path in path2bvh_nodes:
            node = path2bvh_nodes[path]
            assert isinstance(node, pgu.BvhTreeNodeon3DGrid)
            if node.is_leaf:
                sorted_leaf_nodes.append(node)
        sorted_leaf_nodes.sort(key=lambda node:node.path)
        # path2bvh_nodes[''] is the root node
        tree_str = ''.join(pgu.format_BvhTree_on_3DGrid(path2bvh_nodes['']))
        logger.info(f'load tree\n{tree_str}')
        print(f'load tree\n{tree_str}')

    return scene_3d_grid, path2bvh_nodes, sorted_leaf_nodes

def naive_assign_model2rank_dist(num_model):
    # neighbor models tend to be assigned to different rank
    RANK, WORLD_SIZE = dist.get_rank(), dist.get_world_size()
    model2rank = {i: i%WORLD_SIZE for i in range(num_model)}
    return model2rank

def get_model_assignment_dist(sorted_leaf_nodes:List[pgu.BvhTreeNodeon3DGrid], logger:logging.Logger):
    RANK, WORLD_SIZE = dist.get_rank(), dist.get_world_size()
    num_models_tensor = torch.tensor(0, dtype=torch.int, device='cuda')
    if RANK == 0:
        num_models_tensor = torch.tensor(len(sorted_leaf_nodes), dtype=torch.int, device='cuda')
    dist.broadcast(num_models_tensor, src=0, group=None, async_op=False)
    num_model = int(num_models_tensor) 

    models_range_tensor = torch.zeros((num_model, 6), dtype=torch.int, device='cuda')
    if RANK == 0:
        models_range_tensor = torch.tensor([node.range_low + node.range_up for node in sorted_leaf_nodes], dtype=torch.int, device='cuda')     
    dist.broadcast(models_range_tensor, src=0, group=None, async_op=False)
    models_range = models_range_tensor.cpu().numpy()

    model2box = {i: pgu.BoxinGrid3D(models_range[i, 0:3], models_range[i, 3:6]) for i in range(num_model)}
    model2rank = naive_assign_model2rank_dist(num_model=num_model)

    logger.info(f'{num_model} models for {RANK} ranks')
    str_model2box = '\n'.join([f'model {model}: {str(box)}' for model,box in model2box.items()])
    logger.info(f'details about model space:\n{str_model2box}')

    local_model_ids = [model for model,r in model2rank.items() if r==RANK]
    local_model_ids.sort()
    if len(local_model_ids) <= 0:
        logger.error(f'find not model for current rank {RANK}!')
        raise RuntimeError('empty rank')

    return model2box, model2rank, local_model_ids

def get_relation_vector(camera:Camera, z_near:float, z_far:float, path2nodes:Dict[str, pgu.BvhTreeNodeon3DGrid], sorted_leaf_nodes:List[pgu.BvhTreeNodeon3DGrid], logger:logging.Logger):
    NUM_DATA, NUM_MODEL = 1, len(sorted_leaf_nodes)
    complete_relation = np.zeros((NUM_DATA, NUM_MODEL), dtype=int) - 1
    # assert isinstance(camera, (Camera, EmptyCamera))
    # get the depth of all nodes
    all_node_depth = {'':0}
    for path,node in path2nodes.items():
        a = path2nodes[path]
        a.is_leaf
        if node.is_leaf or (node.split_position_grid is None):
            continue
        if node.in_right(camera.camera_center):
            all_node_depth[path+'1'] = 0
            all_node_depth[path+'0'] = 1
        else:
            all_node_depth[path+'1'] = 1
            all_node_depth[path+'0'] = 0

    # logger.info('{}\n{}'.format(i, all_node_depth))
    # sort leaf-nodes by the path(from root to node) 
    mid_pathDepth_list = []
    for mid, node in enumerate(sorted_leaf_nodes):
        path:str = node.path
        pathDepth = tuple(all_node_depth[path[:(prefix+1)]] for prefix in range(len(path)))
        mid_pathDepth_list.append((mid, pathDepth))
    mid_pathDepth_list.sort(key=lambda x:x[-1])
    # if node is not overlapped with view frustum, set relation as -1  
    for order, mid_pathDepth in enumerate(mid_pathDepth_list):
        mid, pathDepth = mid_pathDepth
        node:pgu.BvhTreeNodeon3DGrid = sorted_leaf_nodes[mid]
        space_box = pgu.SpaceBox(node.range_low_in_world, node.range_up_in_world) 
        try:
            is_overlap = pgu.is_overlapping_SpaceBox_View(space_box, camera, z_near=z_near, z_far=z_far)
        except Exception as e:
            is_overlap = True
            logger.warning(traceback.format_exc())
            logger.warning('set is_overlap=True for camera uid {} and mid {}'.format(camera.uid, mid))

        if is_overlap:
            complete_relation[0, mid] = order
        else:
            complete_relation[0, mid] = -1

    # if a row is all -1, set it to all 0
    return complete_relation   

class CamerawithRelation(Dataset):
    def __init__(self, dataset:CameraListDataset, path2nodes:dict, sorted_leaf_nodes:list, Z_MIN:float, Z_MAX:float, logger:logging.Logger) -> None:
        super().__init__()
        self.dataset = dataset
        self.path2nodes:Dict[str, pgu.BvhTreeNodeon3DGrid] = path2nodes
        self.sorted_leaf_nodes:List[pgu.BvhTreeNodeon3DGrid] = sorted_leaf_nodes
        self.Z_MIN:float = Z_MIN
        self.Z_MAX:float = Z_MAX
        self.logger = logger
        if isinstance(self.dataset, CameraListDataset):
            self.get_camera = lambda i:self.dataset.get_empty_item(i)
        else:
            self.get_camera = lambda i:self.dataset[i]
    
    def __len__(self):
        return len(self.dataset)   

    def __getitem__(self, idx):     
        camera: EmptyCamera = self.get_camera(idx)       
        relation_1_N = get_relation_vector(
            camera, 
            z_near=self.Z_MIN, 
            z_far=self.Z_MAX, 
            path2nodes=self.path2nodes, 
            sorted_leaf_nodes=self.sorted_leaf_nodes,
            logger=self.logger
            )
        return None, relation_1_N, camera 

def get_camera_model_relation_Rank0(camera_list:CameraListDataset, path2bvh_nodes:Dict[str, pgu.BvhTreeNodeon3DGrid], sorted_leaf_nodes:List[pgu.BvhTreeNodeon3DGrid], z_min:float, z_max:float, logger:logging.Logger):
    NUM_DATA, NUM_MODEL = len(camera_list), len(sorted_leaf_nodes)
    complete_relation = np.zeros((NUM_DATA, NUM_MODEL), dtype=int) - 1

    data_loader = DataLoader(
        CamerawithRelation(camera_list, path2bvh_nodes, sorted_leaf_nodes, z_min, z_max, logger), 
        batch_size=16, num_workers=32, prefetch_factor=4, drop_last=False,
        shuffle=False, collate_fn=SceneV3.get_batch
    )
    idx_start = 0
    for i, batch in tqdm(enumerate(data_loader)):
        for _data in batch:
            _max_depth, _relation_1_N, camera = _data
            assert isinstance(camera, (Camera, EmptyCamera))
            complete_relation[idx_start, :] = _relation_1_N
            if i%100 == 0:
                logger.debug("{}, {}, {}, {}".format(camera.image_height, camera.image_width, camera.uid, _max_depth))
            idx_start += 1
    # if a row is all -1, set it to all 0
    all_minus_one = (complete_relation.sum(axis=-1) == -NUM_MODEL)  
    complete_relation[all_minus_one, :] = 0
    logger.info('find {} mismatching samples, set their relation to all 0'.format(np.sum(all_minus_one)))

    return complete_relation 

def get_camera_model_relation_dist(ckpt:str, camera_list:CameraListDataset, NUM_MODELS:int, path2bvh_nodes:Dict[str, pgu.BvhTreeNodeon3DGrid], sorted_leaf_nodes:List[pgu.BvhTreeNodeon3DGrid], z_min:float, z_max:float, logger:logging.Logger):
    # if serialization/unserialization method of BVHTreeNode are implemented
    # camera_model_relation can be calculated by multiple nodes (single rank can already launch multiple processes via Dataloader)
    # as burning is familiar with burning's laziness, he decided not to do these 
    RANK, WORLD_SIZE = dist.get_rank(), dist.get_world_size()
    NUM_SAMPLES = len(camera_list)
    find_available_ckpt = False
    if RANK == 0:
        if (ckpt is not None) and len(ckpt)>0 and os.path.exists(ckpt): 
            relation_tensor_cpu:torch.Tensor = torch.load(ckpt)
            if relation_tensor_cpu.shape[0]==len(camera_list) and relation_tensor_cpu.shape[1]==len(sorted_leaf_nodes):
                relation_tensor = relation_tensor_cpu.cuda()
                find_available_ckpt = True
                logger.info('load relation matrix from {}'.format(ckpt))
            else:
                logger.info('ckpt is of shape {}, mismatch {}'.format(relation_tensor_cpu.shape, (len(camera_list),len(sorted_leaf_nodes))))    
        if not find_available_ckpt:
            relation_np = get_camera_model_relation_Rank0(camera_list, path2bvh_nodes, sorted_leaf_nodes, z_min, z_max, logger)
            relation_tensor = torch.tensor(relation_np, dtype=torch.int, device='cuda')
    else:    
        relation_tensor = torch.zeros((NUM_SAMPLES, NUM_MODELS), dtype=torch.int, device='cuda')
   
    dist.barrier(group=None)
    if len(relation_tensor) <= 0:
        logging.warning('empty dataset')
    else:
        dist.broadcast(relation_tensor, src=0, group=None, async_op=False)
    torch.cuda.synchronize()
    return relation_tensor.cpu()

def find_ply_iteration(scene:SceneV3, logger:logging.Logger):
    model_path = scene.model_path
    point_cloud_path = os.path.join(model_path, "point_cloud/iteration_*")
    all_ckpt_dir = glob.glob(point_cloud_path)
    try:
        all_ply_iteration = [int(os.path.basename(path).split('_')[-1]) for path in all_ckpt_dir]
        final_iteration = max(all_ply_iteration)
        logger.info('find ckpt at iteration {}'.format(final_iteration))
    except:
        final_iteration = 0
        logger.info('can not find ckpt')    
    return final_iteration

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

