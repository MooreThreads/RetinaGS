import os
import torch
import torch.distributed as dist
import numpy as np

from parallel_utils.communicate_utils.core import send_tensor, recv_tensor, batched_send_recv_v0, batched_send_recv_v1, batched_send_recv_v2, batched_send_recv_v0_profiling 
from parallel_utils.grid_utils.utils import BoxinGrid3D, overlap_BoxinGrid3D
from parallel_utils.schedulers.core import SendTask, RecvTask, RenderTask, MainRankTask, SendExtraGS, RecvExtraGS

import parallel_utils.grid_utils.core as pgc
import parallel_utils.grid_utils.gaussian_grid as pgg
import parallel_utils.grid_utils.utils as ppu

from scene.gaussian_nn_module import BoundedGaussianModel, BoundedGaussianModelGroup
from scene.cameras import Camera, EmptyCamera, ViewMessage
from utils.datasets import CameraListDataset
from scene.scene4bounded_gaussian import SceneV3

import logging, time
from typing import NamedTuple
import traceback

BATCHED_SEND_RECV_DICT = {
    0: batched_send_recv_v0,
    1: batched_send_recv_v1,
    2: batched_send_recv_v2,
    '0': batched_send_recv_v0,
    '1': batched_send_recv_v1,
    '2': batched_send_recv_v2,
    '0+profiling': batched_send_recv_v0_profiling,
}

'''
in this module, we assume the BoundedGS maintain GS locates in (low, high]
and we also assume it would behave as the optical field in (low, high]
thus extra gs shall be collect from other submodels
'''
from parallel_utils.schedulers.dynamic_space import TaskParser
from parallel_utils.schedulers.dynamic_space import SendBoxinGrid, RecvBoxinGrid
from parallel_utils.schedulers.dynamic_space import SpaceTaskMatcher, BasicSchedulerwithDynamicSpace
# as we assume the BoundedGS maintain GS locates in (low, high], SpaceTaskMatcher is just precise

from typing import List, Callable, Dict


class Parser4OpticalFieldSegment(TaskParser):
    def __init__(self, PROCESS_WORLD_SIZE: int, GLOBAL_RANK: int, logger: logging.Logger) -> None:
        super().__init__(PROCESS_WORLD_SIZE, GLOBAL_RANK, logger)

    def parser_extra_gs_task(self, GLOBAL_SEND_GS_AMOUNT:torch.Tensor, modelId2rank:dict, role:int=None):
        '''
        GLOBAL_SEND_GS_AMOUNT: torch.int cpu tensor [num_model, num_model]
        [i, j] is the amount of extra gs from model_i to model_j 
        '''
        ROLE = role if role is not None else self.GLOBAL_RANK
        NUM_ALL_MODELS = len(modelId2rank)
        assert NUM_ALL_MODELS == GLOBAL_SEND_GS_AMOUNT.shape[0] == GLOBAL_SEND_GS_AMOUNT.shape[1]
        rank2modelIds = {i:[] for i in range(self.PROCESS_WORLD_SIZE)}
        for model_id, _r in modelId2rank.items():
            rank2modelIds[_r].append(model_id)
        for _r in rank2modelIds:
            rank2modelIds[_r].sort() 

        send_tasks:List[SendExtraGS] = []
        recv_tasks:List[RecvExtraGS] = []
        # let organize it in (src_model, dst_model) order
        for src_model in rank2modelIds[ROLE]:
            for dst_model in range(NUM_ALL_MODELS):
                src_rank, dst_rank = modelId2rank[src_model], modelId2rank[dst_model]
                if GLOBAL_SEND_GS_AMOUNT[src_model, dst_model] > 0 and (src_rank != dst_rank):
                    send_tasks.append(SendExtraGS(
                        src_model_id=src_model, 
                        dst_model_id=dst_model, 
                        src_rank=src_rank,
                        dst_rank=dst_rank,
                        grid_range=None,
                        length=GLOBAL_SEND_GS_AMOUNT[src_model, dst_model])
                    )

        for src_model in range(NUM_ALL_MODELS):
            for dst_model in rank2modelIds[ROLE]:  
                src_rank, dst_rank = modelId2rank[src_model], modelId2rank[dst_model]
                if GLOBAL_SEND_GS_AMOUNT[src_model, dst_model] > 0 and (src_rank != dst_rank):
                    recv_tasks.append(RecvExtraGS(
                        src_model_id=src_model, 
                        dst_model_id=dst_model, 
                        src_rank=src_rank,
                        dst_rank=dst_rank,
                        grid_range=None,
                        length=GLOBAL_SEND_GS_AMOUNT[src_model, dst_model])
                    )  
        return send_tasks, recv_tasks                


class BasicScheduler4OpticalFieldSegment():
    def __init__(self, 
            zip_extra_gs: Callable[[Dict], torch.Tensor],
            zip_grad_extra_gs: Callable[[Dict], torch.Tensor],
            zip_render_ret: Callable[[Dict], torch.Tensor],
            zip_grad_render_ret: Callable[[Dict], torch.Tensor],
            unzip_extra_gs: Callable[[torch.Tensor], Dict],
            unzip_grad_extra_gs: Callable[[torch.Tensor], Dict],
            unzip_render_ret: Callable[[torch.Tensor], Dict],
            unzip_grad_render_ret: Callable[[torch.Tensor], Dict],
            space_extra_gs: Callable[[RecvExtraGS], torch.Tensor],
            space_grad_extra_gs: Callable[[SendExtraGS], torch.Tensor],
            space_render_ret: Callable[[RecvTask], torch.Tensor],
            space_grad_render_ret: Callable[[SendTask], torch.Tensor],
            logger: logging.Logger, tb_writer = None, 
            batch_isend_irecv_version=0) -> None:
        self.zip_extra_gs = zip_extra_gs
        self.zip_grad_extra_gs = zip_grad_extra_gs
        self.zip_render_ret = zip_render_ret
        self.zip_grad_render_ret = zip_grad_render_ret
        self.unzip_extra_gs = unzip_extra_gs
        self.unzip_grad_extra_gs = unzip_grad_extra_gs
        self.unzip_render_ret = unzip_render_ret
        self.unzip_grad_render_ret = unzip_grad_render_ret
        self.space_extra_gs = space_extra_gs
        self.space_grad_extra_gs = space_grad_extra_gs
        self.space_render_ret = space_render_ret
        self.space_grad_render_ret = space_grad_render_ret
        self.logger = logger
        self.tb_writer = tb_writer
        self.batch_isend_irecv_version = batch_isend_irecv_version
        # task parsing is removed from this sheduler
        # more decoupling more extensible 

        # zip/unzip pair have symmetric input/output list
        # in zip, replace None-grad with all-zero, it's not elegant but effective 
        # in unzip, be careful with requires_grad/is_leaf property of the output tensors, retain_grad() can also be helpful

    def record_info(self):
        pass

    def exchange_GS(self, send_tasks:List[SendBoxinGrid], recv_tasks:List[RecvBoxinGrid], send_dict:dict, recv_dict:dict, batch_isend_irecv_version=None):
        # model can be much larger than render_result
        # thus the memory management can be complex to avoid memory overflow
        # even if the design pattern in forward_pass can still be an alternate
        # Burning insists that user shall design suitable mechanism as this scheduler focuses on communication
        
        # send_tasks:list[SendBoxinGrid], recv_tasks:list[RecvBoxinGrid]
        if batch_isend_irecv_version is None:
            batch_isend_irecv_version = self.batch_isend_irecv_version
        batched_send_recv:callable = BATCHED_SEND_RECV_DICT[batch_isend_irecv_version]

       
        send_ops = [ 
            (send_dict[(s.src_model_id, s.dst_model_id)], s.dst_rank, None) for s in send_tasks
        ]
        recv_ops = [
            (recv_dict[(r.src_model_id, r.dst_model_id)], r.src_rank, None) for r in recv_tasks
        ]
        batched_send_recv(send_tasks=send_ops, recv_tasks=recv_ops, logger=self.logger)
       
        return recv_dict

    def _log_matching_result(self, _send, _recv, _render, _main):
        strs_send_task = ['\t' + str(e) for e in _send]
        strs_send_task.insert(0, 'matched send tasks:')
        strs_send_task = '\n'.join(strs_send_task)

        strs_recv_task = ['\t' + str(e) for e in _recv]
        strs_recv_task.insert(0, 'matched recv tasks:')
        strs_recv_task = '\n'.join(strs_recv_task)

        strs_render_task = ['\t' + str(e) for e in _render]
        strs_render_task.insert(0, 'matched render tasks:')
        strs_render_task = '\n'.join(strs_render_task)

        strs_main_task = ['\t' + str(e) for e in _main]
        strs_main_task.insert(0, 'matched main tasks:')
        strs_main_task = '\n'.join(strs_main_task)

        msg = '\n'.join([strs_send_task, strs_recv_task, strs_render_task, strs_main_task])
        if self.logger is not None:
            self.logger.debug(msg)    

    def _exchange_extra_gs(self, extra_gs_pool:Dict, send_gs_tasks:List[SendExtraGS], recv_gs_tasks:List[RecvExtraGS], batch_isend_irecv_version=None):
        if batch_isend_irecv_version is None:
            batch_isend_irecv_version = self.batch_isend_irecv_version
        batched_send_recv:callable = BATCHED_SEND_RECV_DICT[batch_isend_irecv_version] 
        # assume pre-task-parsing is trustable
        send_gs_tensor = {
            (task.src_model_id, task.dst_model_id): self.zip_extra_gs(extra_gs_pool[(task.src_model_id, task.dst_model_id)]) for task in send_gs_tasks
        }
        recv_gs_tensor = {
            (task.src_model_id, task.dst_model_id): self.space_extra_gs(task) for task in recv_gs_tasks
        }
        # batched ops
        send_ops = [ 
            (send_gs_tensor[(s.src_model_id, s.dst_model_id)], s.dst_rank, None) for s in send_gs_tasks
        ]
        recv_ops = [
            (recv_gs_tensor[(r.src_model_id, r.dst_model_id)], r.src_rank, None) for r in recv_gs_tasks
        ]
        # s&r
        batched_send_recv(send_tasks=send_ops, recv_tasks=recv_ops, logger=self.logger)
        del send_gs_tensor
        # unpack extra_gs
        extra_gs = {
            (r.src_model_id, r.dst_model_id): self.unzip_extra_gs(recv_gs_tensor[(r.src_model_id, r.dst_model_id)])
                for r in recv_gs_tasks
        }
        del recv_gs_tensor
        return extra_gs
    
    def _exchange_render_ret(self, local_render_rets:Dict, send_tasks:List[SendTask], recv_tasks:List[RecvTask], batch_isend_irecv_version=None):
        if batch_isend_irecv_version is None:
            batch_isend_irecv_version = self.batch_isend_irecv_version
        batched_send_recv:callable = BATCHED_SEND_RECV_DICT[batch_isend_irecv_version] 
        # communication_pass:
        # 1) pack_up_render_result and assign storage space for receiving
        send_tensors = {
            (s.task_id, s.model_id): self.zip_render_ret(local_render_rets[(s.task_id, s.model_id)]) for s in send_tasks
        } 
        recv_tensors = {
            (r.task_id, r.model_id): self.space_render_ret(r) for r in recv_tasks
        }
        # 2) batched_p2p
        # TaskParser organizes send/recv in (task_id, model_id) order
        # thus simply using the order of tasks as the order of the isend/irecv can work
        send_ops = [ 
            (send_tensors[(s.task_id, s.model_id)], s.dst_rank, None) for s in send_tasks
        ]
        recv_ops = [
            (recv_tensors[(r.task_id, r.model_id)], r.src_rank, None) for r in recv_tasks
        ]
        
        batched_send_recv(send_tasks=send_ops, recv_tasks=recv_ops, logger=self.logger)
        del send_tensors
        
        # 3) unpack_up extra render_result from other ranks
        extra_render_rets = {
            (r.task_id, r.model_id): self.unzip_render_ret(recv_tensors[(r.task_id, r.model_id)]) 
                for r in recv_tasks
        }
        del recv_tensors

        return extra_render_rets

    def _exchange_gard_render_ret(self, extra_render_rets_grad:Dict, send_tasks:List[SendTask], recv_tasks:List[RecvTask], batch_isend_irecv_version=None):
        if batch_isend_irecv_version is None:
            batch_isend_irecv_version = self.batch_isend_irecv_version
        batched_send_recv:callable = BATCHED_SEND_RECV_DICT[batch_isend_irecv_version]

        # communication_grad_pass
        # send_grad_tasks/recv_grad_tasks are symmetrical with recv_tasks/send_tasks
        send_grad = {} 
        for r in recv_tasks:
            try:
                send_grad[(r.task_id, r.model_id)]=self.zip_grad_render_ret(extra_render_rets_grad[(r.task_id, r.model_id)])
                if self.logger is not None: 
                    self.logger.debug('find gard for recv_render(task_id={}, model_id={})'.format(r.task_id, r.model_id))
            except Exception as e:
                if self.logger is not None: 
                    self.logger.debug('CAN NOT get gard for recv_render(task_id={}, model_id={})'.format(r.task_id, r.model_id))
                    self.logger.warning(traceback.format_exc())
                # raise e

        recv_grad = {
            (s.task_id, s.model_id): self.space_grad_render_ret(s) for s in send_tasks 
        }
        send_ops = [ 
            (send_grad[(r.task_id, r.model_id)], r.src_rank, None) for r in recv_tasks
        ]
        recv_ops = [
            (recv_grad[(s.task_id, s.model_id)], s.dst_rank, None) for s in send_tasks
        ]

        batched_send_recv(send_tasks=send_ops, recv_tasks=recv_ops, logger=self.logger)
        del send_grad
        # unpack_up_grad
        grad_from_other_rank = {
            (s.task_id, s.model_id): self.unzip_grad_render_ret(recv_grad[(s.task_id, s.model_id)])
                for s in send_tasks
        }
        del recv_grad
        # the second time of autograd would backward though local graph 
        # and accumulate the effection of grad_from_other_rank to local model parameters   
        return grad_from_other_rank

    def _exchange_grad_extra_gs(self, extra_gs_grad:Dict, send_gs_tasks:List[SendExtraGS], recv_gs_tasks:List[RecvExtraGS], batch_isend_irecv_version=None):
        if batch_isend_irecv_version is None:
            batch_isend_irecv_version = self.batch_isend_irecv_version
        batched_send_recv:callable = BATCHED_SEND_RECV_DICT[batch_isend_irecv_version] 

        # communication_grad_pass
        # send_grad_tasks/recv_grad_tasks are symmetrical with recv_tasks/send_tasks
        send_grad = {} 
        for r in recv_gs_tasks:
            try:
                send_grad[(r.src_model_id, r.dst_model_id)]=self.zip_grad_extra_gs(extra_gs_grad[(r.src_model_id, r.dst_model_id)])
                if self.logger is not None: 
                    self.logger.debug('find gard for recv_render(task_id={}, model_id={})'.format(r.src_model_id, r.dst_model_id))
            except Exception as e:
                if self.logger is not None: 
                    self.logger.debug('CAN NOT get gard for recv_render(task_id={}, model_id={})'.format(r.src_model_id, r.dst_model_id))
                    self.logger.warning(traceback.format_exc())
                # raise e

        recv_grad = {
            (s.src_model_id, s.dst_model_id): self.space_grad_extra_gs(s) for s in send_gs_tasks 
        }
        send_ops = [ 
            (send_grad[(r.src_model_id, r.dst_model_id)], r.src_rank, None) for r in recv_gs_tasks
        ]
        recv_ops = [
            (recv_grad[(s.src_model_id, s.dst_model_id)], s.dst_rank, None) for s in send_gs_tasks
        ]

        batched_send_recv(send_tasks=send_ops, recv_tasks=recv_ops, logger=self.logger)
        del send_grad
        # unpack_up_grad
        grad_from_other_rank = {
            (s.src_model_id, s.dst_model_id): self.unzip_grad_extra_gs(recv_grad[(s.src_model_id, s.dst_model_id)])
                for s in send_gs_tasks
        }
        del recv_grad
        # the second time of autograd would backward though local graph 
        # and accumulate the effection of grad_from_other_rank to local model parameters   
        return grad_from_other_rank



"""
these two functions provide a simple implementation for exchanging GS, please think twice before using it
    + global_send_size_np is array of [num_src_model, num_dst_model], model_i would sends elemet[i, j] Gaussian-Splattings to model_j
    + every rank writes its own line to tell other ranks about its content to send
    + MAX_GS_CHANNEL:int, as GS model may have different ranks of Spherical Harmonics, this function need a MAX_GS_CHANNEL
"""
def reduce_global_send_size_dist(send_size_tensor:torch.Tensor, src_local_model_ids:list) -> np.ndarray:
    RANK, WORLD_SIZE = dist.get_rank(), dist.get_world_size()
    local_model = src_local_model_ids

    local_send_size = send_size_tensor[local_model, :]
    dist.all_reduce(send_size_tensor, op=dist.ReduceOp.SUM, group=None, async_op=False)
    reduced_local_send_size = send_size_tensor[local_model, :]
    assert torch.all(local_send_size==reduced_local_send_size), 'local rows are written by another rank'
    return send_size_tensor.cpu().detach().numpy()


def minimal_send_recv_GS(
    src_local_model_ids:list, src_model2rank:dict,
    dst_local_model_ids:list, dst_model2rank:dict, 
    global_send_size_np:np.ndarray, send_gs_task2msg:dict,
    scheduler:BasicSchedulerwithDynamicSpace, logger:logging.Logger,
    MAX_GS_CHANNEL:int
    ):
    src_local_model_ids.sort()
    dst_local_model_ids.sort()  
    NUM_GLOBAL_SRC, NUM_GLOBAL_DST = len(src_model2rank), len(dst_model2rank)
    logger.info("global_send_size_np is\n {}".format(global_send_size_np))
    
    min_send_tasks, self_send_buffer = [], {}
    for src_id in src_local_model_ids:
        for dst_id in range(NUM_GLOBAL_DST):
            if global_send_size_np[src_id, dst_id] <= 0:
                continue
            if src_model2rank[src_id] == dst_model2rank[dst_id]:
                # self_send_buffer[(src_id, dst_id)] = send_gs_task2msg[(src_id, dst_id)]
                pass
            else:
                min_send_tasks.append(SendBoxinGrid(
                    src_model_id=src_id,
                    dst_model_id=dst_id,
                    src_rank=src_model2rank[src_id],
                    dst_rank=dst_model2rank[dst_id],
                    grid_range=None,   
                ))
    logger.info("send_tasks is\n {}".format(min_send_tasks))

    recv_gs_task2space, min_recv_tasks, self_recv_buffer = {}, [], {}
    for src_id in range(NUM_GLOBAL_SRC):
        for dst_id in dst_local_model_ids:
            if global_send_size_np[src_id, dst_id] <= 0:
                continue
            if src_model2rank[src_id] == dst_model2rank[dst_id]:
                self_recv_buffer[(src_id, dst_id)] = send_gs_task2msg[(src_id, dst_id)]
            else:
                min_recv_tasks.append(RecvBoxinGrid(
                    src_model_id=src_id,
                    dst_model_id=dst_id,
                    src_rank=src_model2rank[src_id],
                    dst_rank=dst_model2rank[dst_id],
                    grid_range=None, 
                ))   
                recv_size = global_send_size_np[src_id, dst_id]
                recv_gs_task2space[(src_id, dst_id)] = torch.zeros((recv_size, MAX_GS_CHANNEL), dtype=torch.float, device='cuda')
    logger.info("recv_tasks is\n {}".format(min_recv_tasks))

    scheduler.exchange_GS(min_send_tasks, min_recv_tasks, send_gs_task2msg, recv_gs_task2space)
    return self_recv_buffer, recv_gs_task2space


def divide_scene_3d_grid(scene_3d_grid:ppu.Grid3DSpace, bvh_depth, SPLIT_ORDERS, example_path2bvh_nodes={}, logger=None):
    path2bvh_nodes = ppu.build_BvhTree_on_3DGrid(scene_3d_grid, max_depth=bvh_depth, split_orders=SPLIT_ORDERS, example_path2bvh_nodes=example_path2bvh_nodes, logger=logger)
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


def eval_load_and_divide_grid_dist(
        src_gaussians_group:BoundedGaussianModelGroup, scr_path2bvh_nodes:dict,
        src_model2box:dict, src_model2rank:dict, src_local_model_ids:list,
        scene_3d_grid:ppu.Grid3DSpace, 
        space_task_parser:SpaceTaskMatcher,
        scheduler:BasicSchedulerwithDynamicSpace,
        load_dataset:CameraListDataset, 
        BVH_DEPTH:int, SPLIT_ORDERS, MAX_GS_CHANNEL:int, logger:logging.Logger):
    RANK, WORLD_SIZE = dist.get_rank(), dist.get_world_size()
    logger.setLevel(logging.DEBUG)
    with torch.no_grad():
        # evaluation of workload
        scene_3d_grid.clean_load()
        # loader = DataLoader(load_dataset, batch_size=1, num_workers=2, prefetch_factor=2, shuffle=False, drop_last=True, collate_fn=SceneV3.get_batch)
        for _gaussians_name in src_gaussians_group.all_gaussians:
            _gaussians = src_gaussians_group.get_model(_gaussians_name)
            position = _gaussians.get_xyz.cpu().detach().numpy()
            P = position.shape[0]
            tiles_touched = np.ones((P,), dtype=float)
            # for i, data in enumerate(loader):
            #     dist.barrier()
            #     logger.info('evaluate work load on sample {}'.format(i))
            #     empty_camera:Camera = data[0]  # stll need (H,W) for rendering, load full camera
            #     assert isinstance(_gaussians, BoundedGaussianModel)
            #     render_result = _gaussians(empty_camera.to_device('cuda'), pipe, background, need_buffer=True)
            #     tiles_touched += gather_tiles_touched(P, render_result['geomBuffer'], logger)
            scene_3d_grid.accum_load(load_np=tiles_touched, position=position)
            # logger.warning('tiled_touched {}, {}, {}'.format(
            #     tiles_touched.max(), tiles_touched.min(), tiles_touched.mean()
            # ))
            # logger.warning('position {}, {}, {}'.format(
            #     position.max(axis=0), position.min(axis=0), position.mean(axis=0)
            # ))

        logger.info('rank {} has got local load, and is going to all_reduce it'.format(RANK))
        dist.barrier()
        logger.info('after barrier')
        load_tensor = torch.tensor(scene_3d_grid.load_cnt, dtype=torch.float, device='cuda', requires_grad=False)
        logger.info('load_np 2 load_tensor')
        dist.all_reduce(load_tensor, op=dist.ReduceOp.SUM, group=None, async_op=False)
        logger.info(f'reduced load, {load_tensor.max()}, {load_tensor.min()}, {load_tensor.mean()}')

        reduced_load_np = load_tensor.clamp(min=0).cpu().numpy().astype(float)
        scene_3d_grid.load_cnt = reduced_load_np

        # new partition of space  
        if RANK == 0:
            path2bvh_nodes, sorted_leaf_nodes, tree_str = divide_scene_3d_grid(scene_3d_grid, BVH_DEPTH, SPLIT_ORDERS, example_path2bvh_nodes=scr_path2bvh_nodes, logger=logger)
            logger.info(f'get tree\n{tree_str}')
            if len(sorted_leaf_nodes) != 2**BVH_DEPTH:
                logger.warning(f'bad division! expect {2**BVH_DEPTH} leaf-nodes but get {len(sorted_leaf_nodes)}') 
        else:
            path2bvh_nodes, sorted_leaf_nodes, tree_str = None, None, ''

        dst_model2box, dst_model2rank, dst_local_model_ids = pgg.init_GS_model_division_dist(sorted_leaf_nodes, logger)
       
        send_gs_tasks, recv_gs_tasks = space_task_parser.match(src_model2box, src_model2rank, dst_model2box, dst_model2rank)
        # pack up src model and release some memory
        src_id2GS_package = {}
        for src_id in src_gaussians_group.model_id_list:
            _gau = src_gaussians_group.pop_model(src_id)
            assert _gau is not None and isinstance(_gau, BoundedGaussianModel)
            src_id2GS_package[src_id] = _gau.pack_up()
            del _gau    # release the GS model after get package
            torch.cuda.empty_cache()
        
        # gather sub-packages of src_model and all_reduce the send_size
        send_size_tensor = torch.zeros((len(src_model2rank), len(dst_model2rank)), dtype=torch.int, device='cuda', requires_grad=False)
        send_gs_task2msg = {}
        for send_task in send_gs_tasks:
            assert isinstance(send_task, SendBoxinGrid)
            msg = pgc.gather_msg_in_grid(src_id2GS_package[send_task.src_model_id], send_task.grid_range, scene_3d_grid, padding=0.0)
            send_gs_task2msg[(send_task.src_model_id, send_task.dst_model_id)] = msg
            send_size_tensor[send_task.src_model_id, send_task.dst_model_id] = msg.shape[0]
        global_send_size_np = reduce_global_send_size_dist(send_size_tensor, src_local_model_ids)
        del src_id2GS_package   # release complete GS_pkg after gathering sub-packages
        torch.cuda.empty_cache()

        # use gloabl_send_size to organize send/recv instead of relying on send_gs_task, recv_gs_task
        self_recv_buffer, recv_gs_task2space = minimal_send_recv_GS(
            src_local_model_ids, src_model2rank, dst_local_model_ids, dst_model2rank, 
            global_send_size_np, send_gs_task2msg, scheduler, logger, MAX_GS_CHANNEL=MAX_GS_CHANNEL
        )
        del send_gs_task2msg    # release sub-packages after sending
        torch.cuda.empty_cache()

        # collect sub-pkgs for dst_local_model_ids
        dst_id2msgs = {dst_id:[torch.zeros((0, MAX_GS_CHANNEL), dtype=torch.float, device='cuda')] for dst_id in dst_local_model_ids}
        for k in self_recv_buffer:
            _src, _dst = k
            dst_id2msgs[_dst].append(self_recv_buffer[k])
        for k in recv_gs_task2space:
            _src, _dst = k
            dst_id2msgs[_dst].append(recv_gs_task2space[k])
        del self_recv_buffer, recv_gs_task2space
        
        return path2bvh_nodes, sorted_leaf_nodes, tree_str, dst_model2box, dst_model2rank, dst_local_model_ids, dst_id2msgs

