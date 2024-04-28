import os
import sys
import traceback, uuid, logging, time, shutil, glob
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from utils.general_utils import is_interval_in_batch, is_point_in_batch
from utils.datasets import DatasetRepeater, GroupedItems, CameraListDataset
import parallel_utils.grid_utils.utils as ppu
import parallel_utils.grid_utils.core as pgc
from parallel_utils.schedulers.core import SendTask, RecvTask, RenderTask, MainRankTask
import parallel_utils.schedulers.dynamic_space as psd 

from scene.gaussian_nn_module import BoundedGaussianModel, BoundedGaussianModelGroup
from scene.cameras import Camera, EmptyCamera, ViewMessage
from scene.scene4bounded_gaussian import SceneV3

"""
path2nodes:dict of nodes in BVH tree where key is str(01) path
sorted_leaf_nodes:list of nodes in BVH tree where sorted_leaf_nodes[i] is node of model_i
"""

# this blender merges results from GS with convex boundary, we move it from core as it is not very universal
# the blending order is read from _relation_vector
def build_func_blender(final_background:torch.tensor, logger: logging.Logger):
    def local_func_blender(local_render_rets, extra_render_rets, main_rank_task:MainRankTask, _relation_vector:torch.Tensor):
        """
        local_render_rets: {(task_id, model_id): render_ret}
        extra_render_rets: {(task_id, model_id): render_ret}
        relation_vector (NUM_MODEL, ) int tensor, 
            1) -1 indicate not-related to task
            2) non-negative value is the blending order of model ouput for task
        """ 
        if isinstance(_relation_vector, torch.Tensor):
            relation_vector = _relation_vector.cpu().numpy()
        else:
            relation_vector = _relation_vector
        NUM_RELATED = (relation_vector >= 0).sum()
        related_ret = [] 
        # logger.info("blending (taskid={}, rank={}) in current rank {}, {}".format(
        #     main_rank_task.task_id, main_rank_task.rank, RANK, relation_vector))
        # collect related render_result from local render_rets and extra_render_rets
        gather_from_local = 'gather_from_local: '
        for k in local_render_rets:
            task_id, model_id = k
            if (task_id == main_rank_task.task_id) and (relation_vector[model_id]>=0):
                related_ret.append((task_id, model_id, local_render_rets[k], relation_vector[model_id]))
                gather_from_local += ('(task,model):' + str(k))
        logger.debug(gather_from_local)

        gather_from_extra = 'gather_from_extra: '
        for k in extra_render_rets:
            task_id, model_id = k
            if (task_id == main_rank_task.task_id) and (relation_vector[model_id]>=0):
                related_ret.append((task_id, model_id, extra_render_rets[k], relation_vector[model_id]))
                gather_from_extra += ('(task,model):' + str(k))
        logger.debug(gather_from_extra)        

        if len(related_ret) != NUM_RELATED:
            logger.error(f"rank {dist.get_rank()} tries to blender task {main_rank_task.task_id}, but NUM_RELATED mis-match!")
            raise RuntimeError("NUM_RELATED mis-match")

        # sort by order provided in _relation_vector
        related_ret.sort(key=lambda x: x[-1], reverse=False)    
        sorted_image, sorted_alpha, sorted_depth = [], [], []
        for task_id, model_id, pkg, order in related_ret:
            sorted_image.append(pkg['render'])  # (c,H,W)
            sorted_alpha.append(pkg['alpha'])   # (1,H,W)
            sorted_depth.append(pkg['depth'])   # (1,H,W), this is a depth map not c_depth

        blender_ret = {}
        sorted_alpha.insert(0, torch.zeros_like(sorted_alpha[0]))

        _cated_tsprt = torch.cat(sorted_alpha, dim=0)    # (num_img+1,H,W)
        _cated_tsprt = torch.cumprod(1 - _cated_tsprt, dim=0)
        cated_tsprt = _cated_tsprt[0:-1]
        cated_depth = torch.cat(sorted_depth, dim=0)
        cated_image = torch.stack(sorted_image, dim=0)  # (num_img,c,H,W)

        accum_alpha = 1 - _cated_tsprt[-1:]
        accum_depth = (cated_tsprt * cated_depth).sum(dim=0, keepdim=True)
        accum_image = (cated_tsprt.unsqueeze(1) * cated_image).sum(dim=0, keepdim=False)    # (num_img,1,H,W)*(num_img,c,H,W).sum(dim=0, keepdim=False)

        bkg_color = final_background if final_background is not None else torch.rand((3), device="cuda")
        accum_image_with_bkg = accum_image + bkg_color.view(-1,1,1) * _cated_tsprt[-1].unsqueeze(0)

        blender_ret = {"render":accum_image_with_bkg, "depth":accum_depth, "alpha":accum_alpha}
        return blender_ret

    return local_func_blender


def init_grid(scene: SceneV3, SCENE_GRID_SIZE: int):
    _SPACE_RANGE_LOW, _SPACE_RANGE_UP = scene.point_cloud.points.min(axis=0, keepdims=False), scene.point_cloud.points.max(axis=0, keepdims=False)
    _VOXEL_SIZE = (_SPACE_RANGE_UP - _SPACE_RANGE_LOW)/SCENE_GRID_SIZE - 1e-7
    return _SPACE_RANGE_LOW, _SPACE_RANGE_UP, _VOXEL_SIZE

def init_grid_dist(scene: SceneV3, SCENE_GRID_SIZE: int):
    RANK = dist.get_rank()
    if RANK == 0:
        _SPACE_RANGE_LOW, _SPACE_RANGE_UP = scene.point_cloud.points.min(axis=0, keepdims=False), scene.point_cloud.points.max(axis=0, keepdims=False)
        _VOXEL_SIZE = (_SPACE_RANGE_UP - _SPACE_RANGE_LOW)/SCENE_GRID_SIZE - 1e-7
        grid_np = np.array([_SPACE_RANGE_LOW, _SPACE_RANGE_UP, _VOXEL_SIZE])
        grid_tensor = torch.tensor(grid_np, dtype=torch.float32, device='cuda')
    else:     
        grid_tensor = torch.zeros((3, 3), dtype=torch.float32, device='cuda')

    dist.broadcast(grid_tensor, src=0, group=None, async_op=False)
    grid_numpy = grid_tensor.cpu().numpy()
    SPACE_RANGE_LOW, SPACE_RANGE_UP, VOXEL_SIZE = grid_numpy[0], grid_numpy[1], grid_numpy[2]
    return SPACE_RANGE_LOW, SPACE_RANGE_UP, VOXEL_SIZE

def load_grid(scene:SceneV3, ply_iteration:int, SCENE_GRID_SIZE: int):
    RANK = 0
    model_path = scene.model_path
    point_cloud_path = os.path.join(model_path, "point_cloud/iteration_{}".format(ply_iteration))
    tree_path = os.path.join(point_cloud_path, "tree_{}.txt".format(RANK))

    _SPACE_RANGE_LOW, _SPACE_RANGE_UP, UNUSED_GRID_SIZE, path2node_info_dict = ppu.load_BvhTree_on_3DGrid(tree_path)
    _VOXEL_SIZE = (_SPACE_RANGE_UP - _SPACE_RANGE_LOW)/SCENE_GRID_SIZE - 1e-7
    grid_np = np.array([_SPACE_RANGE_LOW, _SPACE_RANGE_UP, _VOXEL_SIZE])
    grid_tensor = torch.tensor(grid_np, dtype=torch.float32, device='cuda')
    grid_numpy = grid_tensor.cpu().numpy()
    SPACE_RANGE_LOW, SPACE_RANGE_UP, VOXEL_SIZE = grid_numpy[0], grid_numpy[1], grid_numpy[2]
    return SPACE_RANGE_LOW, SPACE_RANGE_UP, VOXEL_SIZE, path2node_info_dict

def load_grid_dist(scene:SceneV3, ply_iteration:int, SCENE_GRID_SIZE: int):
    RANK = dist.get_rank()
    model_path = scene.model_path
    point_cloud_path = os.path.join(model_path, "point_cloud/iteration_{}".format(ply_iteration))
    tree_path = os.path.join(point_cloud_path, "tree_{}.txt".format(RANK))

    if RANK == 0:
        _SPACE_RANGE_LOW, _SPACE_RANGE_UP, UNUSED_GRID_SIZE, path2node_info_dict = ppu.load_BvhTree_on_3DGrid(tree_path)
        _VOXEL_SIZE = (_SPACE_RANGE_UP - _SPACE_RANGE_LOW)/SCENE_GRID_SIZE - 1e-7
        grid_np = np.array([_SPACE_RANGE_LOW, _SPACE_RANGE_UP, _VOXEL_SIZE])
        grid_tensor = torch.tensor(grid_np, dtype=torch.float32, device='cuda')
    else:     
        path2node_info_dict = None
        grid_tensor = torch.zeros((3, 3), dtype=torch.float32, device='cuda')

    dist.broadcast(grid_tensor, src=0, group=None, async_op=False)
    grid_numpy = grid_tensor.cpu().numpy()
    SPACE_RANGE_LOW, SPACE_RANGE_UP, VOXEL_SIZE = grid_numpy[0], grid_numpy[1], grid_numpy[2]
    return SPACE_RANGE_LOW, SPACE_RANGE_UP, VOXEL_SIZE, path2node_info_dict

def divide_model_by_load(scene_3d_grid:ppu.Grid3DSpace, bvh_depth:int, load:np.ndarray, position:np.ndarray, logger:logging.Logger, SPLIT_ORDERS:list):
    """
    SPLIT_ORDERS is a list of [0/1/2] which specify the splited dimensions of bvh tree 
    """
    scene_3d_grid.clean_load()
    if load is None:
        load = np.ones((position.shape[0], ), dtype=float)
    scene_3d_grid.accum_load(load_np=load, position=position)
    path2bvh_nodes = ppu.build_BvhTree_on_3DGrid(scene_3d_grid, max_depth=bvh_depth, split_orders=SPLIT_ORDERS)
    # find leaf-nodes as model space
    sorted_leaf_nodes = []
    for path in path2bvh_nodes:
        node = path2bvh_nodes[path]
        assert isinstance(node, ppu.BvhTreeNodeon3DGrid)
        if node.is_leaf:
            sorted_leaf_nodes.append(node)
    sorted_leaf_nodes.sort(key=lambda node:node.path)
    size_list = [node.size for node in sorted_leaf_nodes]
    format_str_list = ppu.format_BvhTree_on_3DGrid(path2bvh_nodes['']) 

    print('build tree \n', ''.join(format_str_list))
    assert sum(size_list) == path2bvh_nodes[''].size, "size mismatch"

    return path2bvh_nodes, sorted_leaf_nodes, ''.join(format_str_list)

def load_model_division(scene_3d_grid:ppu.Grid3DSpace, path2node_info_dict:dict, logger:logging.Logger):
    path2bvh_nodes = ppu.build_BvhTree_on_3DGrid_with_info(path2node_info_dict, scene_3d_grid)
    # find leaf-nodes as model space
    sorted_leaf_nodes = []
    for path in path2bvh_nodes:
        node = path2bvh_nodes[path]
        assert isinstance(node, ppu.BvhTreeNodeon3DGrid)
        if node.is_leaf:
            sorted_leaf_nodes.append(node)
    sorted_leaf_nodes.sort(key=lambda node:node.path)
    size_list = [node.size for node in sorted_leaf_nodes]
    assert sum(size_list) == path2bvh_nodes[''].size, "size mismatch"
    format_str_list = ppu.format_BvhTree_on_3DGrid(path2bvh_nodes['']) 
    return path2bvh_nodes, sorted_leaf_nodes, ''.join(format_str_list)

def assign_model2rank_dist(num_model):
    # neighbor models tend to be assigned to different rank
    RANK, WORLD_SIZE = dist.get_rank(), dist.get_world_size()
    model2rank = {i: i%WORLD_SIZE for i in range(num_model)}
    return model2rank

def init_GS_model_division(all_leaf_node:list, logger:logging.Logger):
    # all_leaf_node: list of BvhTreeNodeon3DGrid
    RANK, WORLD_SIZE = 0, 1
    num_models_tensor = torch.tensor(len(all_leaf_node), dtype=torch.int, device='cuda')
    num_model = int(num_models_tensor) 

    models_range_tensor = torch.tensor([node.range_low + node.range_up for node in all_leaf_node], dtype=torch.int, device='cuda')
    models_range = models_range_tensor.cpu().numpy()

    model2box = {i: ppu.BoxinGrid3D(models_range[i, 0:3], models_range[i, 3:6]) for i in range(num_model)}
    # model2rank = assign_model2rank_dist(num_model=num_model)
    model2rank = {i: 0 for i in range(num_model)}

    logger.info(f'{num_model} models for {RANK} ranks')
    str_model2box = '\n'.join([f'model {model}: {str(box)}' for model,box in model2box.items()])
    logger.info(f'details about model space:\n{str_model2box}')

    local_model_ids = [model for model,r in model2rank.items() if r==RANK]
    local_model_ids.sort()
    if len(local_model_ids) <= 0:
        logger.error(f'find not model for current rank {RANK}!')
        raise RuntimeError('empty rank')

    return model2box, model2rank, local_model_ids

def init_GS_model_division_dist(all_leaf_node:list, logger:logging.Logger):
    # all_leaf_node: list of BvhTreeNodeon3DGrid
    RANK, WORLD_SIZE = dist.get_rank(), dist.get_world_size()
    if RANK == 0:
        num_models_tensor = torch.tensor(len(all_leaf_node), dtype=torch.int, device='cuda')
    else:
        num_models_tensor = torch.tensor(0, dtype=torch.int, device='cuda')

    dist.broadcast(num_models_tensor, src=0, group=None, async_op=False)
    num_model = int(num_models_tensor) 

    if RANK == 0:
        models_range_tensor = torch.tensor([node.range_low + node.range_up for node in all_leaf_node], dtype=torch.int, device='cuda')
    else:        
        models_range_tensor = torch.zeros((num_model, 6), dtype=torch.int, device='cuda')
    dist.broadcast(models_range_tensor, src=0, group=None, async_op=False)
    models_range = models_range_tensor.cpu().numpy()

    model2box = {i: ppu.BoxinGrid3D(models_range[i, 0:3], models_range[i, 3:6]) for i in range(num_model)}
    model2rank = assign_model2rank_dist(num_model=num_model)

    logger.info(f'{num_model} models for {RANK} ranks')
    str_model2box = '\n'.join([f'model {model}: {str(box)}' for model,box in model2box.items()])
    logger.info(f'details about model space:\n{str_model2box}')

    local_model_ids = [model for model,r in model2rank.items() if r==RANK]
    local_model_ids.sort()
    if len(local_model_ids) <= 0:
        logger.error(f'find not model for current rank {RANK}!')
        raise RuntimeError('empty rank')

    return model2box, model2rank, local_model_ids

def init_datasets_dist(scene:SceneV3, opt, path2nodes:dict, sorted_leaf_nodes:list, batch_size:int, NUM_MODEL:int, logger:logging.Logger, get_range:callable, EVAL_PSNR_INTERVAL=8):
    RANK, WORLD_SIZE = dist.get_rank(), dist.get_world_size()

    train_dataset: CameraListDataset = scene.getTrainCameras() 
    eval_test_dataset: CameraListDataset = scene.getTestCameras()
    eval_train_list = DatasetRepeater(train_dataset, len(train_dataset)//EVAL_PSNR_INTERVAL, False, EVAL_PSNR_INTERVAL)
    torch.cuda.synchronize()
    if RANK == 0:
        assert len(sorted_leaf_nodes) == NUM_MODEL
        # train_dataloader = DataLoader(repeat_train_dataset, batch_size=batch_size, num_workers=4, prefetch_factor=2, shuffle=True, drop_last=True, collate_fn=SceneV3.get_batch)
        default_rlt_path = os.path.join(scene.model_path, 'trainset_relation.pt')
        if False and os.path.exists(default_rlt_path):
            trainset_relation_tensor = torch.load(default_rlt_path)
            logger.info('load trainset_relation_tensor form {}'.format(default_rlt_path))
        else: 
            trainset_relation_np = get_relation_matrix(train_dataset, path2nodes, sorted_leaf_nodes, logger, get_range)
            trainset_relation_tensor = torch.tensor(trainset_relation_np, dtype=torch.int, device='cuda')

        evalset_relation_np = get_relation_matrix(eval_test_dataset, path2nodes, sorted_leaf_nodes, logger, get_range)
        evalset_relation_tensor = torch.tensor(evalset_relation_np, dtype=torch.int, device='cuda')
    else:
        # train_dataloader = DataLoader(repeat_train_dataset, batch_size=batch_size, drop_last=True, collate_fn=SceneV3.get_batch)
        trainset_relation_tensor = torch.zeros((len(train_dataset), NUM_MODEL), dtype=torch.int, device='cuda')
        evalset_relation_tensor = torch.zeros((len(eval_test_dataset), NUM_MODEL), dtype=torch.int, device='cuda')

    dist.barrier()
    dist.broadcast(trainset_relation_tensor, src=0, group=None, async_op=False)
    logger.info(trainset_relation_tensor.shape)
    logger.info(trainset_relation_tensor)
    if len(eval_test_dataset) > 0:
        dist.broadcast(evalset_relation_tensor, src=0, group=None, async_op=False)
    else:
        logger.warning('strange! empty eval dataset')    

    torch.cuda.synchronize()
    if RANK == 0:
        torch.save(trainset_relation_tensor, os.path.join(scene.model_path, 'trainset_relation.pt'))

    logger.info('dataloader is prepared')
    return train_dataset, eval_test_dataset, eval_train_list, trainset_relation_tensor.cpu(), evalset_relation_tensor.cpu()

def get_relation_matrix(train_dataset:CameraListDataset, path2nodes:dict, sorted_leaf_nodes:list, logger:logging.Logger, get_range:callable):
    NUM_DATA, NUM_MODEL = len(train_dataset), len(sorted_leaf_nodes)
    complete_relation = np.zeros((NUM_DATA, NUM_MODEL), dtype=int) - 1

    for i in tqdm(range(len(train_dataset))):
        camera = train_dataset.get_empty_item(idx=i)    # only need view frustum
        # assert isinstance(camera, (Camera, EmptyCamera))
        # get the depth of all nodes
        all_node_depth = {'':0}
        for path,node in path2nodes.items():
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
            node:ppu.BvhTreeNodeon3DGrid = sorted_leaf_nodes[mid]
            space_box = ppu.SpaceBox(node.range_low_in_world, node.range_up_in_world) 
            try:
                _near, _far = get_range(camera)
                is_overlap = ppu.is_overlapping_SpaceBox_View(space_box, camera, z_near=_near, z_far=_far)
            except Exception as e:
                is_overlap = True
                logger.warning(traceback.format_exc())
                logger.warning('set is_overlap=True for camera uid {} and mid {}'.format(i, mid))

            if is_overlap:
                complete_relation[i, mid] = order
            else:
                complete_relation[i, mid] = -1

    # if a row is all -1, set it to all 0
    all_minus_one = (complete_relation.sum(axis=-1) == -NUM_MODEL)  
    complete_relation[all_minus_one, :] = 0
    logger.info('find {} mismatching samples, set their relation to all 0'.format(np.sum(all_minus_one)))

    return complete_relation   

def get_relation_vector(camera:Camera, z_near:float, z_far, path2nodes:dict, sorted_leaf_nodes:list, logger:logging.Logger):
    NUM_DATA, NUM_MODEL = 1, len(sorted_leaf_nodes)
    complete_relation = np.zeros((NUM_DATA, NUM_MODEL), dtype=int) - 1
    # assert isinstance(camera, (Camera, EmptyCamera))
    # get the depth of all nodes
    all_node_depth = {'':0}
    for path,node in path2nodes.items():
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
        node:ppu.BvhTreeNodeon3DGrid = sorted_leaf_nodes[mid]
        space_box = ppu.SpaceBox(node.range_low_in_world, node.range_up_in_world) 
        try:
            is_overlap = ppu.is_overlapping_SpaceBox_View(space_box, camera, z_near=z_near, z_far=z_far)
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

def densification(iteration, batch_size, opt, dataset_arg, scene:SceneV3, gaussians_group:BoundedGaussianModelGroup, local_render_rets:dict, tb_writer, logger:logging.Logger):
    if iteration < opt.densify_until_iter:
        # Keep track of max radii in image-space for pruning
        # update max_radii and denom by each render_task while update xyz_gradient_accum just once
        # visibility_filter/radii is part of each independent forward_render_ret 
        # but _means2D_meta.grad is accumulated for all render_task as they involve in the same torch.autograd.backward
        activated_model_ids = set()
                    
        for k in local_render_rets:
            task_id, model_id = k
            activated_model_ids.add(model_id)
            _visibility_filter, _radii = local_render_rets[k]['visibility_filter'], local_render_rets[k]['radii']
            _gaussians:BoundedGaussianModel = gaussians_group.get_model(model_id)
            assert _gaussians is not None
            _gaussians.max_radii2D[_visibility_filter] = torch.max(_gaussians.max_radii2D[_visibility_filter], _radii[_visibility_filter])
            _gaussians.denom[_visibility_filter] += 1
        logger.debug('at iteration {}, update max radii, means2D_gard for activated_models {}'.format(
            iteration, str(activated_model_ids)
        ))   
        # 1.1 zeros gard may occur in visible gaussian, but unvisible gaussians must has zero grad, as they are not mapped with any tiles
        # 1.2 thus only considering value, +=_means2D_meta.grad is equal to +=_means2D_meta.grad[mask]  
        # 2.1 effection of all extra_grads from other ranks is accumulated in a same backward 
        # 2.2 if had to follow original code, namely calling add_densification_stats, it's unavoidable to backforward for every render_ret independently  
        for model_id in activated_model_ids:
            _gaussians:BoundedGaussianModel = gaussians_group.get_model(model_id)
            _gaussians.xyz_gradient_accum += _gaussians._means2D_meta.grad

        if (iteration > opt.densify_from_iter) and is_interval_in_batch(iteration, batch_size, opt.densification_interval):
            t0 = time.time()
            size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            for model_id in gaussians_group.model_id_list:
                _gaussians:BoundedGaussianModel = gaussians_group.get_model(model_id)
                _gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                gaussians_shape = torch.zeros((1, ), dtype=torch.int64, device='cuda')
                gaussians_shape[0] = _gaussians._xyz.shape[0]
                logger.info("model {} update to size {} at iteration {}".format(model_id, _gaussians._xyz.shape[0], iteration))
                if tb_writer:
                    tb_writer.add_scalar('total_points_{}'.format(model_id), _gaussians.get_xyz.shape[0], iteration)
            torch.cuda.empty_cache()
            t1 = time.time()
            if tb_writer:
                tb_writer.add_scalar('cnt/densify_and_prune', t1-t0, iteration)           

        if is_interval_in_batch(iteration, batch_size, opt.opacity_reset_interval) or \
            (dataset_arg.white_background and is_point_in_batch(iteration, batch_size, opt.densify_from_iter)):
            for model_id in gaussians_group.model_id_list:
                _gaussians:BoundedGaussianModel = gaussians_group.get_model(model_id)
                _gaussians.reset_opacity()
            logger.info('rank {} has reset opacity at iteration {}'.format(dist.get_rank(), iteration))

