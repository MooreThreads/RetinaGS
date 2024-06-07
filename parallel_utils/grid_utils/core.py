import os, sys
import traceback, uuid, logging, time, shutil, glob
from tqdm import tqdm

import numpy as np
import torch
import torch.distributed as dist

from parallel_utils.grid_utils.utils import BoxinGrid3D, Grid3DSpace
import parallel_utils.schedulers.dynamic_space as psd 
from parallel_utils.schedulers.core import SendTask, RecvTask, RenderTask, MainRankTask
from scene.gaussian_nn_module import BoundedGaussianModel, BoundedGaussianModelGroup
from scene.cameras import Camera, EmptyCamera, ViewMessage
from scene.scene4bounded_gaussian import SceneV3
from utils.general_utils import is_interval_in_batch, is_point_in_batch


class ckpt_cleaner():
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

def func_space_for_render(Rtask: RecvTask):
    return BoundedGaussianModelGroup.space_for_task(Rtask.task)

def func_space_for_grad_of_render(Stask: SendTask):
    return BoundedGaussianModelGroup.space_for_grad_of_task(Stask.task)

# pipe and background are passed to GS model without any modification 
def build_func_render(gaussians_group: BoundedGaussianModelGroup, pipe, background, logger: logging.Logger, need_buffer:bool=False):
    def func_render(render_task: RenderTask):
        model:BoundedGaussianModel = gaussians_group.get_model(render_task.model_id)
        if model is None:
            logger.error(f"rank {dist.get_rank()} tries to render task {render_task.task_id} with model {render_task.model_id}, but get None model!")
            raise RuntimeError("model missing")
        return model.forward(viewpoint_cam=render_task.task, pipe=pipe, background=background, need_buffer=need_buffer)
    
    return func_render

# GS render_ret must provide necessary keys
def make_copy_for_blending(main_rank_tasks:list, local_render_rets:dict):
    # make smallest copy of render_ret from GS to cut off auto-grad 
    main_task_ids = [m.task_id for m in main_rank_tasks]
    local_render_rets_copy = {}
    for tid_mid in local_render_rets:
        task_id, model_id = tid_mid
        if task_id in main_task_ids:
            local_render_rets_copy[tid_mid] = {
                "render": local_render_rets[tid_mid]["render"].clone().detach().requires_grad_(True),
                "depth": local_render_rets[tid_mid]["depth"].clone().detach().requires_grad_(True),
                "alpha": local_render_rets[tid_mid]["alpha"].clone().detach().requires_grad_(True),
            }
    return local_render_rets_copy  

def build_check_and_pack_up_grad(logger: logging.Logger):
    def check_and_pack_up_grad(render_ret):
        assert render_ret["render"].requires_grad
        assert render_ret["alpha"].requires_grad
        assert render_ret["depth"].requires_grad

        assert render_ret["render"].grad is not None
        return BoundedGaussianModelGroup.pack_up_grad_of_render_ret(render_ret)

    return check_and_pack_up_grad

def gather_tensor_grad_of_render_result(send_tasks, local_render_rets, grad_from_other_rank, local_render_rets_copy):
    # gather tensor/grad from send_out tensor/recv_in gard
    # local_render_rets/grad_from_other_rank: {'render':Tensor, 'alpha':Tensor, 'depth':Tensor}
    send_local_tensor, extra_gard = [], []
    for s_task in send_tasks:
        assert isinstance(s_task, SendTask)
        task_id, model_id = s_task.task_id, s_task.model_id
        # send_out tensors must be stored in render_rets    
        send_local_tensor.append(local_render_rets[(task_id, model_id)]["render"])
        send_local_tensor.append(local_render_rets[(task_id, model_id)]["alpha"])
        send_local_tensor.append(local_render_rets[(task_id, model_id)]["depth"])
        extra_gard.append(grad_from_other_rank[(task_id, model_id)]["render"])
        extra_gard.append(grad_from_other_rank[(task_id, model_id)]["alpha"])
        extra_gard.append(grad_from_other_rank[(task_id, model_id)]["depth"])
    # gather tensor/grad from local render_ret/copy
    main_local_tensor, main_grad = [], []
    for tid_mid in local_render_rets_copy:
        if local_render_rets_copy[tid_mid]['render'].grad is not None:
            main_local_tensor.append(local_render_rets[tid_mid]['render'])
            main_grad.append(local_render_rets_copy[tid_mid]['render'].grad)
        if local_render_rets_copy[tid_mid]['alpha'].grad is not None:
            main_local_tensor.append(local_render_rets[tid_mid]['alpha'])
            main_grad.append(local_render_rets_copy[tid_mid]['alpha'].grad)  
        if local_render_rets_copy[tid_mid]['depth'].grad is not None:
            main_local_tensor.append(local_render_rets[tid_mid]['depth'])
            main_grad.append(local_render_rets_copy[tid_mid]['depth'].grad) 

    return send_local_tensor, extra_gard, main_local_tensor, main_grad    

def gather_tiles_touched(P:int, geomBuffer:torch.Tensor, logger:logging.Logger):
    b = geomBuffer.cpu().numpy().tobytes()

    idx = 0
    # depths = np.frombuffer(b[idx:idx+P*4], dtype=np.float32)

    idx = int(np.ceil(P*4/128)*128)
    # clamped = np.frombuffer(b[idx:idx+P*3], dtype=np.uint8).reshape((P,3))

    idx = int(np.ceil((idx+P*3)/128)*128)
    # internal_radii = np.frombuffer(b[idx:idx+P*4], dtype=np.int32)

    idx = int(np.ceil((idx+P*4)/128)*128)
    # means2D = np.frombuffer(b[idx:idx+P*8], dtype=np.float32).reshape((P,2))

    idx = int(np.ceil((idx+P*8)/128)*128)
    # zw = np.frombuffer(b[idx:idx+P*8], dtype=np.float32).reshape((P,2))

    idx = int(np.ceil((idx+P*8)/128)*128)
    # cov3D = np.frombuffer(b[idx:idx+P*4*6], dtype=np.float32).reshape((P,6))

    idx = int(np.ceil((idx+P*4*6)/128)*128)
    # conic_opacity = np.frombuffer(b[idx:idx+P*16], dtype=np.float32).reshape((P,4))

    idx = int(np.ceil((idx+P*16)/128)*128)
    # rgb = np.frombuffer(b[idx:idx+P*4*3], dtype=np.float32).reshape((P,3))

    idx = int(np.ceil((idx+P*4*3)/128)*128)
    tiles_touched = np.frombuffer(b[idx:idx+P*4], dtype=np.uint32)

    idx = int(np.ceil((idx+P*4)/128)*128)
    # point_offsets = np.frombuffer(b[idx:idx+P*4], dtype=np.uint32)

    idx = int(np.ceil((idx+P*4)/128)*128)

    # logger.warning('{}, {}, {}'.format(tiles_touched.max(), tiles_touched.min(), tiles_touched.mean()) )
    return tiles_touched.astype(float).clip(min=0, max=100) 

def gather_msg_in_grid(complete_pkg:tuple, box:BoxinGrid3D, scene_3d_grid:Grid3DSpace, padding:float=0.0):
    pkg, scale = complete_pkg   # packed BoundedGaussianModel

    # int range 2 float range in world coordinate
    max_radii, _idx = torch.max(scale, dim=-1, keepdim=True)
    max_radii = (max_radii*3).clamp(min=0)

    range_low_np = (box.range_low - padding) * scene_3d_grid.voxel_size + scene_3d_grid.range_low
    range_up_np = (box.range_up + padding) * scene_3d_grid.voxel_size + scene_3d_grid.range_low
    range_low_gpu = torch.tensor(range_low_np, dtype=torch.float, device='cuda')
    range_up_gpu = torch.tensor(range_up_np, dtype=torch.float, device='cuda')

    flag1 = (pkg[:, :3] + max_radii) >= range_low_gpu
    flag2 = (pkg[:, :3] - max_radii) <= range_up_gpu
    _flag = torch.logical_and(flag1, flag2)
    flag = torch.all(_flag, dim=-1, keepdim=False)
    return pkg[flag, :].contiguous()

def update_densification_stat(iteration, opt, gaussians_group:BoundedGaussianModelGroup, local_render_rets:dict, tb_writer, logger:logging.Logger):
    if iteration < opt.densify_until_iter:    
        activated_model_ids = set()         
        for k in local_render_rets:
            task_id, model_id = k
            activated_model_ids.add(model_id)
            _visibility_filter, _radii = local_render_rets[k]['visibility_filter'], local_render_rets[k]['radii']
            _gaussians:BoundedGaussianModel = gaussians_group.get_model(model_id)
            assert _gaussians is not None
            valid_length = _gaussians._xyz.shape[0]
            _visibility_filter, _radii = _visibility_filter[:valid_length], _radii[:valid_length]
            _gaussians.max_radii2D[_visibility_filter] = torch.max(_gaussians.max_radii2D[_visibility_filter], _radii[_visibility_filter])
            _gaussians.denom[_visibility_filter] += 1
            
        logger.debug('at iter {}, update max radii, means2D_gard for activated_models {}'.format(
            iteration, str(activated_model_ids)
        ))   
        # 1.1 zeros gard may occur in visible gaussian, but unvisible gaussians must has zero grad, as they are not mapped with any tiles
        # 1.2 thus only considering value, +=_means2D_meta.grad is equal to +=_means2D_meta.grad[mask]  
        # 2.1 effection of all extra_grads from other ranks is accumulated in a same backward 
        # 2.2 if had to follow original code, namely calling add_densification_stats, it's unavoidable to backforward for every render_ret independently  
        for model_id in activated_model_ids:
            _gaussians:BoundedGaussianModel = gaussians_group.get_model(model_id)
            _gaussians.xyz_gradient_accum += _gaussians._means2D_meta.grad

def densification(iteration, batch_size, skip_prune:bool, skip_clone:bool, skip_split:bool, opt, scene:SceneV3, gaussians_group:BoundedGaussianModelGroup, tb_writer, logger:logging.Logger):
    # Keep track of max radii in image-space for pruning
    # update max_radii and denom by each render_task while update xyz_gradient_accum just once
    # visibility_filter/radii is part of each independent forward_render_ret 
    # but _means2D_meta.grad is accumulated for all render_task as they involve in the same torch.autograd.backward
    if iteration < opt.densify_until_iter:    
        if (iteration >= opt.densify_from_iter) and is_interval_in_batch(iteration, batch_size, opt.densification_interval):
            t0 = time.time()
            size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            for model_id in gaussians_group.model_id_list:
                _gaussians:BoundedGaussianModel = gaussians_group.get_model(model_id)
                logger.info('skip_prune is {}, skip_clone is {}, skip_split is {} at iteration {}'.format(skip_prune, skip_clone, skip_split, iteration))
                _gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, 
                                             skip_prune=skip_prune, skip_clone=skip_clone, skip_split=skip_split)
                gaussians_shape = torch.zeros((1, ), dtype=torch.int64, device='cuda')
                gaussians_shape[0] = _gaussians._xyz.shape[0]
                logger.info("model {} update to size {} at iteration {}".format(model_id, _gaussians._xyz.shape[0], iteration))
                if tb_writer:
                    tb_writer.add_scalar('total_points_{}'.format(model_id), _gaussians.get_xyz.shape[0], iteration)
            torch.cuda.empty_cache()
            t1 = time.time()
            if tb_writer:
                tb_writer.add_scalar('cnt/densify_and_prune', t1-t0, iteration) 

def reset_opacity(iteration, batch_size, opt, dataset_arg, gaussians_group:BoundedGaussianModelGroup, tb_writer, logger:logging.Logger):
    try:
        RANK = dist.get_rank()
    except:
        RANK = -1    
    if iteration < opt.densify_until_iter: 
        if is_interval_in_batch(iteration, batch_size, opt.opacity_reset_interval) or \
            (dataset_arg.white_background and is_point_in_batch(iteration, batch_size, opt.densify_from_iter)):
            for model_id in gaussians_group.model_id_list:
                _gaussians:BoundedGaussianModel = gaussians_group.get_model(model_id)
                _gaussians.reset_opacity()
            logger.info('rank {} has reset opacity at iteration {}'.format(RANK, iteration))

        return True
    return False 

# reasons for why we store GS in .ply and its optimizer in .pt:
# 1. in MP training, you may find the ckpts so large that you want to discard some content
# 2. momentums in optimizer are actually not very important for GS model
def load_gs_from_ply(opt, gaussians_group:BoundedGaussianModelGroup, local_model_ids:list, scene:SceneV3, ply_iteration:int, logger:logging.Logger):
    model_path:str = scene.model_path
    
    glob_path = os.path.join(
        os.path.abspath(os.path.join(model_path, '..')), '*'
    )
    logger.info('glob in dir {}'.format(glob_path))
    point_cloud_path = os.path.join(glob_path, "point_cloud/iteration_{}".format(ply_iteration))

    for mid in local_model_ids:
        _gau:BoundedGaussianModel = gaussians_group.get_model(mid)
        _ply_path = os.path.join(point_cloud_path, "point_cloud_{}.ply".format(mid))
        ply_path = glob.glob(_ply_path)[-1]

        _gau.load_ply(ply_path)
        # this setup is necassary even optimizer would load lr from .pt
        # we must set it as GS update some lr with get_expon_lr_func
        # you can just consider load_ply + set spatial_lr_scale = create_from_pcd
        _gau.spatial_lr_scale = scene.cameras_extent 
        logger.info('set _gau.spatial_lr_scale as {}'.format(_gau.spatial_lr_scale))

        # build optimizer
        _gau.training_setup(opt)
        # load optimizer state_dict
        adam_path = os.path.join(os.path.dirname(ply_path), "adam_{}.pt".format(mid))
        if os.path.exists(adam_path):
            _gau.optimizer.load_state_dict(torch.load(adam_path))
            logger.info("load from {}".format(adam_path))
        else:
            logger.info('find no adam optimizer')

        logger.info('load from {}'.format(ply_path))
        
    gaussians_group.set_SHdegree(ply_iteration//1000)   

def load_gs_from_single_ply(opt, gaussians_group:BoundedGaussianModelGroup, local_model_ids:list, scene:SceneV3, ply_path:str, logger:logging.Logger):
    for mid in local_model_ids:
        _gau:BoundedGaussianModel = gaussians_group.get_model(mid)
        _gau.load_ply(ply_path)
        # this setup is necassary even optimizer would load lr from .pt
        # we must set it as GS update some lr with get_expon_lr_func
        # you can just consider load_ply + set spatial_lr_scale = create_from_pcd
        _gau.spatial_lr_scale = scene.cameras_extent 
        logger.info('set _gau.spatial_lr_scale as {}'.format(_gau.spatial_lr_scale))

        # build optimizer
        _gau.training_setup(opt)
        # load optimizer state_dict
        adam_path = os.path.join(os.path.dirname(ply_path), "adam_{}.pt".format(mid))
        if os.path.exists(adam_path):
            _gau.optimizer.load_state_dict(torch.load(adam_path))
            logger.info("load from {}".format(adam_path))
        else:
            logger.info('find no adam optimizer')

        _gau.discard_gs_out_range()
        print('model {} has {} gs after discard_gs_out_range'.format(mid, _gau._xyz.shape[0]))
        logger.info('model {} has {} gs after discard_gs_out_range'.format(mid, _gau._xyz.shape[0])) 

def find_ply_iteration(scene:SceneV3, logger:logging.Logger):
    model_path = scene.model_path
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

