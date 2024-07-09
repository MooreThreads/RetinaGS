'''
these utils:
    + prepare the content_pool/space_pool for Scheduler
    + collect/merge the results of ModelForwardTask like RenderResult/SharedGSInfo/GSParametsers 
    + do not handel nccl job
'''
import torch
import numpy as np
import logging, time
from typing import NamedTuple, Callable, List, Dict, Any, Tuple
import traceback
import torch.distributed as dist    #  environment informartion

from parallel_utils.basic_parallel_trainer.task_utils import SendSharedGSInfo, SendGSParameters, SendRenderResult, RecvGSParameters, RecvRenderResult, RecvSharedGSInfo
from parallel_utils.basic_parallel_trainer.task_utils import RenderTask, MainRankTask, RenderResult, SharedGSInfo, GSParametsers
import parallel_utils.grid_utils.utils as pgu

from gaussian_renderer.render_half_gs import render4renderinfo, gradNormHelpFunction
from scene.gaussian_nn_module import BoundedGaussianModel, BoundedGaussianModelGroup


def gather_RenderResult_pool(gaussians:BoundedGaussianModelGroup, render_info_pool:Dict[int, SharedGSInfo], render_tasks:List[RenderTask], pipe, background, logger:logging.Logger):
    '''
    Tuple here is (int(task_id), model_id)
    '''
    ret: Dict[Tuple, RenderResult] = {}
    for task in render_tasks:
        task_id, model_id = int(task.task_id), task.model 
        ret[(task_id, model_id)] = RenderResult(
            render4renderinfo(
                viewpoint_camera=task.info,
                GS=gaussians.get_model(model_id),
                info=render_info_pool[model_id],
                pipe=pipe,
                bg_color=background
            )
        )
    return ret


def gather_local_SharedGSInfo(gaussians:BoundedGaussianModelGroup, local_model_ids:List[int]):
    ret: Dict[int, SharedGSInfo] = {}
    for mid in local_model_ids:
        model: BoundedGaussianModel = gaussians.get_model(mid)
        ret[mid] = SharedGSInfo(
            means3D=model.get_xyz,
            means2D=gradNormHelpFunction(model._means2D_meta),
            opacity=model.get_opacity,
            scales=model.get_scaling,
            rotations=model.get_rotation,
            shs=model.get_features,
        )
    return ret    

def gather_SharedGSInfo_with_boxes(src_id:int, gaussians: BoundedGaussianModel, dst_models:List[int], modelId2Boxes: Dict[int, pgu.BoxinGrid3D], scene_3d_grid:pgu.Grid3DSpace, logger:logging.Logger):
    '''
    we assume the BoundedGS maintain GS locates in (low, high]
    and we also assume it would behave as the optical field in (low, high]
    '''
    main_info = SharedGSInfo(
        means3D=gaussians.get_xyz,
        means2D=gradNormHelpFunction(gaussians._means2D_meta),
        opacity=gaussians.get_opacity,
        scales=gaussians.get_scaling,
        rotations=gaussians.get_rotation,
        shs=gaussians.get_features
    )

    # find the small amount GS that go beyond optical field
    scene_voxel_size = torch.tensor(scene_3d_grid.voxel_size, device='cuda')
    scene_range_low = torch.tensor(scene_3d_grid.range_low, device='cuda')

    src_box = modelId2Boxes[src_id]
    range_low_gpu = (torch.tensor(src_box.range_low, device='cuda') * scene_voxel_size + scene_range_low).view(-1)
    range_up_gpu = (torch.tensor(src_box.range_up, device='cuda') * scene_voxel_size + scene_range_low).view(-1)

    # for SharedGSInfo, primitives that overlap box shall be shared
    max_radii, _idx = torch.max(main_info.scales, dim=-1, keepdim=True)
    max_radii = (max_radii*3).clamp(min=0)

    max_xyz = main_info.means3D + max_radii
    min_xyz = main_info.means3D - max_radii
    flag1 = max_xyz >= range_up_gpu # beyond upper bound
    flag2 = min_xyz <= range_low_gpu # beyond lower bound
    _flag = torch.logical_or(flag1, flag2)
    flag = torch.any(_flag, dim=-1, keepdim=False)

    logger.debug('candidate gs for exchange: {}'.format(flag.sum()))
    candidate = SharedGSInfo(
        means3D=main_info.means3D[flag].contiguous(),
        means2D=main_info.means2D[flag].contiguous(),
        shs=main_info.shs[flag].contiguous(),
        opacity=main_info.opacity[flag].contiguous(),
        scales=main_info.scales[flag].contiguous(),
        rotations=main_info.rotations[flag].contiguous()
    )
    subset_max_xyz = max_xyz[flag].contiguous()
    subset_min_xyz = min_xyz[flag].contiguous()
    # match for every dst model (improvement is available with extra priors )
    ret: Dict[tuple, SharedGSInfo] = {}
    ret[(src_id, src_id)] = main_info
    for dst_model in dst_models:
        box = modelId2Boxes[dst_model]
        range_low_gpu = (torch.tensor(box.range_low, device='cuda') * scene_voxel_size + scene_range_low).view(-1)
        range_up_gpu = (torch.tensor(box.range_up, device='cuda') * scene_voxel_size + scene_range_low).view(-1)
    
        flag1 = subset_max_xyz >= range_low_gpu
        flag2 = subset_min_xyz <= range_up_gpu
        _flag = torch.logical_and(flag1, flag2)
        flag = torch.all(_flag, dim=-1, keepdim=False)

        ret[(src_id, dst_model)] = SharedGSInfo(
            means3D=candidate.means3D[flag].contiguous(),
            means2D=candidate.means2D[flag].contiguous(),
            shs=candidate.shs[flag].contiguous(),
            opacity=candidate.opacity[flag].contiguous(),
            scales=candidate.scales[flag].contiguous(),
            rotations=candidate.rotations[flag].contiguous()
        )

    return ret    

def gather_SharedGSInfo_pool(groups:BoundedGaussianModelGroup, modelId2Box: Dict[int, pgu.BoxinGrid3D], scene_3d_grid:pgu.Grid3DSpace, logger:logging.Logger):
    '''
    dist is {(src_modelId, dst_modelId): SharedGSInfo}
    '''
    NUM_MODELS  = len(modelId2Box)
    # just match a optical field segment with all others
    render_info_pool:Dict[Tuple[int, int], SharedGSInfo] = {}
    for str_id in groups.all_gaussians:
        src_model = groups.all_gaussians[str_id]
        src_id = int(str_id)
        dst_models = [_id for _id in range(NUM_MODELS) if _id != src_id]
        infos = gather_SharedGSInfo_with_boxes(
            src_id=src_id, gaussians=src_model, 
            dst_models=dst_models, modelId2Boxes=modelId2Box, scene_3d_grid=scene_3d_grid, logger=logger)

        pool_size = len(render_info_pool)
        render_info_pool.update(infos)
        assert len(render_info_pool) - pool_size == len(infos), 'find repeated info in gather_SharedGSInfo_pool'

    # src-models and dst-models are just the all gaussian-model
    SEND_AMOUNT_CPU = torch.zeros((NUM_MODELS, NUM_MODELS), dtype=torch.int, device='cpu')
    for src_dst in render_info_pool:
        SEND_AMOUNT_CPU[src_dst[0], src_dst[1]] = render_info_pool[src_dst].means3D.shape[0]

    return render_info_pool, SEND_AMOUNT_CPU


def gather_GSParametsers_in_box(pkg:torch.Tensor, box:pgu.BoxinGrid3D, scene_3d_grid:pgu.Grid3DSpace):
    range_low_np = (box.range_low) * scene_3d_grid.voxel_size + scene_3d_grid.range_low
    range_up_np = (box.range_up) * scene_3d_grid.voxel_size + scene_3d_grid.range_low
    range_low_gpu = torch.tensor(range_low_np, dtype=torch.float, device='cuda')
    range_up_gpu = torch.tensor(range_up_np, dtype=torch.float, device='cuda')
    # GS info would be shared among models if necessary  
    # thus no need to set mirrors on different models for primitive
    # here, a model stores primitives whose center locate in (low, high]
    flag1 = pkg[:, :3] > range_low_gpu
    flag2 = pkg[:, :3] <= range_up_gpu
    _flag = torch.logical_and(flag1, flag2)
    flag = torch.all(_flag, dim=-1, keepdim=False)
    return pkg[flag, :].contiguous()

def gather_GSParametsers_pool_and_release_srcGS(src_gaussians:BoundedGaussianModelGroup, src_model2Box:Dict[int, pgu.BoxinGrid3D], dst_model2Box:Dict[int, pgu.BoxinGrid3D], scene_3d_grid:pgu.Grid3DSpace, logger:logging.Logger):
    '''
    tuple here is (src_id, dst_id)
    '''
    # burning still wants to remind that exchanging GSParametsers can easily cause OOM
    # this implementation is unsafe
    NUM_SRC_MODELS, NUM_DST_MODELS = len(src_model2Box), len(dst_model2Box)
    
    info_pool:Dict[Tuple, torch.Tensor] = {}
    for src_id in src_gaussians.model_id_list:
        _gau = src_gaussians.pop_model(src_id)
        assert _gau is not None and isinstance(_gau, BoundedGaussianModel)
        scr_pkg, scr_gs_scale = _gau.pack_up()
        del _gau    # release the GS model after get package
        
        for dst_id, dst_box in dst_model2Box.items():
            src2dst_pkg = gather_GSParametsers_in_box(scr_pkg, dst_box, scene_3d_grid)
            if len(src2dst_pkg) > 0:
                info_pool[(src_id, dst_id)] = src2dst_pkg
        del scr_pkg, scr_gs_scale   
    
    SEND_AMOUNT_CPU = torch.zeros((NUM_SRC_MODELS, NUM_DST_MODELS), dtype=torch.int, device='cpu')
    for src_dst in info_pool:
        SEND_AMOUNT_CPU[src_dst[0], src_dst[1]] = len(info_pool[src_dst])

    _info_pool = {k:GSParametsers(v) for k,v in info_pool.items()}
    return _info_pool, SEND_AMOUNT_CPU


def blender_render_result(
    local_render_rets:Dict[tuple, RenderResult], 
    extra_render_rets:Dict[tuple, RenderResult], 
    main_rank_task:MainRankTask,
    _relation_vector:torch.Tensor,
    background:torch.Tensor,
    logger: logging.Logger
    ):
    '''
    + tuple here is (int(self.task_id), self.model) of ModelForwardTask
    + _relation_vector is cpu tensor (burning thinks tensors carrying control-infor that is frequently used should be stored on cpu) 
        1) relation_vector (NUM_MODEL, ) int tensor, -1 indicate not-related to task
        2) non-negative value is the blending order of model ouput for task
    '''
    if isinstance(_relation_vector, torch.Tensor):
        relation_vector = _relation_vector.cpu().numpy()
    else:
        relation_vector = _relation_vector
    NUM_RELATED = (relation_vector >= 0).sum()
    related_ret = [] 
       
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
    for _i in range(len(related_ret)):
        pkg = related_ret[_i][2]
        assert isinstance(pkg, RenderResult)
        sorted_image.append(pkg.render)  # (c,H,W)
        sorted_alpha.append(pkg.alpha)   # (1,H,W)
        sorted_depth.append(pkg.depth)   # (1,H,W), this is a depth map not c_depth

    sorted_alpha.insert(0, torch.zeros_like(sorted_alpha[0]))

    _cated_tsprt = torch.cat(sorted_alpha, dim=0)    # (num_img+1,H,W)
    _cated_tsprt = torch.cumprod(1 - _cated_tsprt, dim=0)
    cated_tsprt = _cated_tsprt[0:-1]
    cated_depth = torch.cat(sorted_depth, dim=0)
    cated_image = torch.stack(sorted_image, dim=0)  # (num_img,c,H,W)

    accum_alpha = 1 - _cated_tsprt[-1:]
    accum_depth = (cated_tsprt * cated_depth).sum(dim=0, keepdim=True)
    accum_image = (cated_tsprt.unsqueeze(1) * cated_image).sum(dim=0, keepdim=False)    # (num_img,1,H,W)*(num_img,c,H,W).sum(dim=0, keepdim=False)

    bkg_color = background if background is not None else torch.rand((3), device="cuda")
    accum_image_with_bkg = accum_image + bkg_color.view(-1,1,1) * _cated_tsprt[-1].unsqueeze(0)

    blender_ret = RenderResult({"render":accum_image_with_bkg, "depth":accum_depth, "alpha":accum_alpha})
    return blender_ret


def merge_SharedGSInfo(local_pool:Dict[tuple, SharedGSInfo], extra_pool:Dict[tuple, SharedGSInfo], dst_model_ids:List[int], logger:logging.Logger):
    # tuple key of pool shall be (src_model, dst_model)
    extra_pool = {} if extra_pool is None else extra_pool
    id2infos = {int(_id):[] for _id in dst_model_ids}
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
    ret:Dict[int, SharedGSInfo] = {}    # dst_model_id 2 SharedGSInfo
    for mid, info_list in id2infos.items():
        ret[mid] = SharedGSInfo(
            means3D = torch.cat([info.means3D for info in info_list], dim=0).contiguous(),
            means2D = torch.cat([info.means2D for info in info_list], dim=0).contiguous(),
            opacity = torch.cat([info.opacity for info in info_list], dim=0).contiguous(),
            scales = torch.cat([info.scales for info in info_list], dim=0).contiguous(),
            rotations = torch.cat([info.rotations for info in info_list], dim=0).contiguous(),
            shs = torch.cat([info.shs for info in info_list], dim=0).contiguous(),
        )

    return ret


def merge_GSParametsers(local_pool:Dict[tuple, GSParametsers], extra_pool:Dict[tuple, GSParametsers], dst_model_ids:List[int], CHANNEL:int, logger:logging.Logger):
    # tuple key of pool shall be (src_model, dst_model)
    dst_id2msgs = {dst_id:[torch.zeros((0, CHANNEL), dtype=torch.float, device='cuda')] for dst_id in dst_model_ids}
    for k in local_pool:
        _src, _dst = k
        if _dst in dst_id2msgs:
            dst_id2msgs[_dst].append(local_pool[k].pkg)
    for k in extra_pool:
        _src, _dst = k
        if _dst in dst_id2msgs:
            dst_id2msgs[_dst].append(extra_pool[k].pkg)

    ret:Dict[int, GSParametsers] = {}  # dst_model_id 2 GSParametsers
    for dst in dst_model_ids:
        ret[dst] = GSParametsers(torch.cat(dst_id2msgs[dst], dim=0)) 
        logger.info(f'model {dst} gets pkg of size {ret[dst].pkg.shape}')

    return ret    

