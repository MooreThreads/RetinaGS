import os, sys
import traceback, uuid, logging, time, shutil, glob
from tqdm import tqdm
import numpy as np
import torch

from abc import ABC
import torch.distributed as dist
from utils.general_utils import is_interval_in_batch, is_point_in_batch
from arguments import ModelParams, PipelineParams, OptimizationParams

from scene.gaussian_nn_module import BoundedGaussianModel, BoundedGaussianModelGroup
from scene.cameras import Camera, EmptyCamera
from scene.scene4bounded_gaussian import SceneV3

from parallel_utils.basic_parallel_trainer.task_utils import RenderResult

from gaussian_renderer.render_half_gs import gradNormHelpFunction, RenderInfoFromGS
from typing import List, Dict, Tuple, Callable, NamedTuple

def update_densification_stat(iteration, opt:OptimizationParams, gaussians_group:BoundedGaussianModelGroup, local_render_rets:Dict[tuple, RenderResult], tb_writer, logger:logging.Logger):
    # tuple is key of (task_id, model_id) of a RenderTask
    if iteration < opt.densify_until_iter:    
        activated_model_ids = set()         
        for k in local_render_rets:
            task_id, model_id = k
            activated_model_ids.add(model_id)
            _visibility_filter, _radii = local_render_rets[k].raw['visibility_filter'], local_render_rets[k].raw['radii']
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

def densification(iteration, batch_size, skip_prune:bool, skip_clone:bool, skip_split:bool, opt:OptimizationParams, scene:SceneV3, gaussians_group:BoundedGaussianModelGroup, tb_writer, logger:logging.Logger):
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

def reset_opacity(iteration, batch_size, opt:OptimizationParams, model_para:ModelParams, gaussians_group:BoundedGaussianModelGroup, tb_writer, logger:logging.Logger):
    # burning knows model_para is not ModelParams but he is too lazy to format so many data
    try:
        RANK = dist.get_rank()
    except:
        RANK = -1    
    if iteration < opt.densify_until_iter: 
        if is_interval_in_batch(iteration, batch_size, opt.opacity_reset_interval) or \
            (model_para.white_background and is_point_in_batch(iteration, batch_size, opt.densify_from_iter)):
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
            _gau.optimizer.load_state_dict(torch.load(adam_path, map_location='cuda'))
            logger.info("load from {}".format(adam_path))
        else:
            logger.info('find no adam optimizer')

        logger.info('load from {}'.format(ply_path))
        
    gaussians_group.set_SHdegree(ply_iteration//1000)   


def load_gs_from_single_ply(opt, gaussians_group:BoundedGaussianModelGroup, local_model_ids:list, scene:SceneV3, load_iteration:int, logger:logging.Logger):
    ply_path =  os.path.join(os.path.dirname(scene.model_path), "point_cloud", "iteration_"+str(load_iteration), "point_cloud.ply")
    for mid in local_model_ids:
        _gau:BoundedGaussianModel = gaussians_group.get_model(mid)
        _gau.load_ply(ply_path)
        _gau.spatial_lr_scale = scene.cameras_extent 
        logger.info('set _gau.spatial_lr_scale as {}'.format(_gau.spatial_lr_scale))
        
        _gau.training_setup(opt)

        # _gau.discard_gs_out_range()
        # print('model {} has {} gs after discard_gs_out_range'.format(mid, _gau._xyz.shape[0]))
        # logger.info('model {} has {} gs after discard_gs_out_range'.format(mid, _gau._xyz.shape[0])) 

        # build optimizer
        
    gaussians_group.set_SHdegree(load_iteration//1000)   
        
