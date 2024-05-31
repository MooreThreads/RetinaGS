import os
import torch
import torch.distributed as dist
import numpy as np

from parallel_utils.communicate_utils.core import send_tensor, recv_tensor, batched_send_recv_v0, batched_send_recv_v1, batched_send_recv_v2, batched_send_recv_v0_profiling 
from parallel_utils.grid_utils.utils import BoxinGrid3D, overlap_BoxinGrid3D
from parallel_utils.schedulers.core import SendTask, RecvTask, RenderTask, MainRankTask

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

"""
this scheduler transmit render_result of models and GS in models between ranks
it has listed features:
1. all half-gs models corresponds to a box in 3D-grid-space
    + they can be different in size and even overlapped
    + it requires prepared _relation_matrix  
2. a view has ONE main_rank and several render_rank
    + render_rank has one/several related models
    + main_rank blender the render result and calculates loss
3. model_ids shall be int just like rank!
4. _relation_matrix shall be (num_view, num_model) cpu Tensor
    + non-negative _relation_matrix[i,j] stands for view_i is related to model_j
"""

BATCHED_SEND_RECV_DICT = {
    0: batched_send_recv_v0,
    1: batched_send_recv_v1,
    2: batched_send_recv_v2,
    '0': batched_send_recv_v0,
    '1': batched_send_recv_v1,
    '2': batched_send_recv_v2,
    '0+profiling': batched_send_recv_v0_profiling,
}


class TaskParser:
    def __init__(self, PROCESS_WORLD_SIZE:int, GLOBAL_RANK:int, logger: logging.Logger) -> None:
        """
        PROCESS_WORLD_SIZE: global information for all rank, provided by dist package
        rank2model_ids: global information, provided by the same initialization strategy in all ranks  
        """
        self.PROCESS_WORLD_SIZE = PROCESS_WORLD_SIZE
        self.GLOBAL_RANK = GLOBAL_RANK
        self.logger = logger 

    def parse_task_tensor(self, modelId2rank:dict, _task_main_rank:torch.Tensor, _relation_matrix:torch.Tensor, tasks_message:list, role:int=None):
        """
        modelId2rank: dict[int:int]
        task_main_rank: (batch_size, 2) int tensor 
        related_matrix: (batch_size, NUM_ALL_MODELS) int tensor, 
            1) -1 indicate not-related to task
            2) non-negative value is the blending order of model ouput for task
        """
        self.modelId2rank = modelId2rank
        self.NUM_ALL_MODELS = len(modelId2rank)
        self.rank2modelIds = {i:[] for i in range(self.PROCESS_WORLD_SIZE)}
        for model_id, _r in modelId2rank.items():
            self.rank2modelIds[_r].append(model_id)
        
        for _r in self.rank2modelIds:
            self.rank2modelIds[_r].sort() 

        task_main_rank, relation_matrix = _task_main_rank.cpu(), _relation_matrix.cpu()
        ROLE = role if role is not None else self.GLOBAL_RANK
        batch_size = task_main_rank.shape[0]

        assert ROLE in self.rank2modelIds
        assert relation_matrix.shape[0] == len(tasks_message) == batch_size and relation_matrix.shape[1] == self.NUM_ALL_MODELS

        send_tasks, recv_tasks, render_tasks, main_rank_tasks = [], [], [], []
        # render_tasks, main_rank_tasks
        for id_in_batch in range(batch_size):
            task_id, main_rank, msg = int(task_main_rank[id_in_batch, 0]), int(task_main_rank[id_in_batch, 1]), tasks_message[id_in_batch]
            for model_id in self.rank2modelIds[ROLE]:
                if relation_matrix[id_in_batch, model_id] >= 0:
                    # render for related task
                    render_tasks.append(RenderTask(task_id=task_id, model_id=model_id, task=msg))
                    # send render result to main_rank if current role is not main_rank 
                    # as self.rank2modelIds had been sorted in __init__, send tasks of the same rank are in model_id order 
                    if ROLE != main_rank:
                        send_tasks.append(SendTask(task_id=task_id, model_id=model_id, src_rank=ROLE, dst_rank=main_rank, task=msg))    

            if main_rank == ROLE:
                main_rank_tasks.append(MainRankTask(task_id=task_id, rank=ROLE, task=msg))
                # organize recv tasks in model_id order as send tasks of the same rank are in model_id order
                for model_id in range(self.NUM_ALL_MODELS):
                    if relation_matrix[id_in_batch, model_id] >= 0:
                        src_rank = self.modelId2rank[model_id]
                        # main_rank has not need to recv from itself (to save resource cost)  
                        if src_rank != ROLE: 
                            recv_tasks.append(RecvTask(task_id=task_id, model_id=model_id, src_rank=src_rank, dst_rank=ROLE, task=msg))
        
        return send_tasks, recv_tasks, render_tasks, main_rank_tasks


class SendBoxinGrid(NamedTuple):
    src_model_id: int
    dst_model_id: int
    src_rank: int
    dst_rank: int
    grid_range: BoxinGrid3D


class RecvBoxinGrid(NamedTuple):
    src_model_id: int
    dst_model_id: int
    src_rank: int
    dst_rank: int
    grid_range: BoxinGrid3D
        

class SpaceTaskMatcher:
    """
    Actually, no need to search overlap in tree, we won't have too many models
    While seralization of bvh tree can bring me more tireness
    Thus, just match boxes in a simpler way
    """
    def __init__(self, PROCESS_WORLD_SIZE:int, GLOBAL_RANK:int, logger: logging.Logger) -> None:
        """
        PROCESS_WORLD_SIZE: global information for all rank, provided by dist package
        rank2model_ids: global information, provided by the same initialization strategy in all ranks  
        """
        self.PROCESS_WORLD_SIZE = PROCESS_WORLD_SIZE
        self.GLOBAL_RANK = GLOBAL_RANK
        self.logger = logger
        
    def match(
            self, 
            id2box:dict,    
            id2rank:dict, 
            new_id2box:dict,
            new_id2rank: dict,  
            role:int=None,  # analysis for rank-th process
            ):
        send_gs_task, recv_gs_task = [], []
        ROLE = role if role is not None else self.GLOBAL_RANK

        rank2ids = {i:[] for i in range(self.PROCESS_WORLD_SIZE)} 
        new_rank2ids = {i:[] for i in range(self.PROCESS_WORLD_SIZE)}

        for model_id,_r in id2rank.items():
            rank2ids[_r].append(model_id)
        for model_id,_r in new_id2rank.items():
            new_rank2ids[_r].append(model_id)

        for _r in rank2ids:
            rank2ids[_r].sort()     
            new_rank2ids[_r].sort() 

        # send/recv in (src_model_id, dst_model_id) order
        for src_model_id in rank2ids[ROLE]: 
            for dst_model_id in range(len(new_id2rank)):
                if overlap_BoxinGrid3D(id2box[src_model_id], new_id2box[dst_model_id]) is not None:
                    send_gs_task.append(
                        SendBoxinGrid(
                            src_model_id=src_model_id,
                            dst_model_id=dst_model_id,
                            src_rank=ROLE,
                            dst_rank=new_id2rank[dst_model_id],
                            grid_range=new_id2box[dst_model_id],   # new range of model is enough to find GS 
                        )
                    )

        for src_model_id in range(len(id2rank)) : 
            for dst_model_id in new_rank2ids[ROLE]:
                if overlap_BoxinGrid3D(id2box[src_model_id], new_id2box[dst_model_id]) is not None:
                    recv_gs_task.append(
                        RecvBoxinGrid(
                            src_model_id=src_model_id,
                            dst_model_id=dst_model_id,
                            src_rank=id2rank[src_model_id],
                            dst_rank=ROLE,
                            grid_range=None,   # just recv GS for dst_model, no more information is needed
                        )
                    )
        return send_gs_task, recv_gs_task


class BasicSchedulerwithDynamicSpace:
    def __init__(
            self, 
            task_parser: TaskParser, 
            logger: logging.Logger, 
            func_pack_up: callable,
            func_grad_pack_up: callable,
            func_unpack_up: callable,
            func_grad_unpack_up: callable,
            func_space: callable,
            func_grad_space: callable,
            tb_writer: None,
            batch_isend_irecv_version=0
            ) -> None:
        self.task_parser = task_parser
        self.logger = logger
        self.tb_writer = tb_writer
        self._cnt = 0
        self.saved_for_backward_dict = {}
        self.batch_isend_irecv_version = batch_isend_irecv_version

        # accept render_result and pack up it to float tensor
        self.func_pack_up, self.func_grad_pack_up = func_pack_up, func_grad_pack_up 
        # accept packed tensor delivered by other rank and read render_result from it
        self.func_unpack_up, self.func_grad_unpack_up = func_unpack_up, func_grad_unpack_up
        # accept recv_task and return the empty tensor to store the delivered tensor  
        self.func_space, self.func_grad_space = func_space, func_grad_space

        self.render_pass_cost:float = 0.0
        self.send_recv_forward_cost:float = 0.0
        self.send_recv_backward_cost:float = 0.0

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

    def forward_pass(self, func_render:callable, modelId2rank:dict, _task_main_rank:torch.tensor, _relation_matrix:torch.tensor, views:list, batch_isend_irecv_version=None):
        if batch_isend_irecv_version is None:
            batch_isend_irecv_version = self.batch_isend_irecv_version
        batched_send_recv:callable = BATCHED_SEND_RECV_DICT[batch_isend_irecv_version]

        # task_matching_pass:
        self._cnt += 1 
        # self.logger.debug(f'{modelId2rank}, {_task_main_rank}, {views}')
        send_tasks, recv_tasks, render_tasks, main_rank_tasks = self.task_parser.parse_task_tensor(
            modelId2rank=modelId2rank,
            _task_main_rank = _task_main_rank,
            _relation_matrix = _relation_matrix,
            tasks_message=views
        )
        if self.logger is not None:
            self._log_matching_result(send_tasks, recv_tasks, render_tasks, main_rank_tasks)
        
        # local render_pass:  
        t0 = time.time()
        render_rets = {(t.task_id, t.model_id): None for t in render_tasks}
        for t in render_tasks:
            self.logger.debug('render {}'.format((t.task_id, t.model_id)))
            render_rets[(t.task_id, t.model_id)] = func_render(t)
        t1 = time.time() 
        self.render_pass_cost += (t1-t0)  
        if self.tb_writer is not None:
            self.tb_writer.add_scalar('cnt/render_pass', t1-t0, self._cnt)
            
        # communication_pass:
        # 1) pack_up_render_result and assign storage space for receiving
        send_tensors = {
            (s.task_id, s.model_id): self.func_pack_up(render_rets[(s.task_id, s.model_id)]) 
                for s in send_tasks
        } 
        recv_tensors = {
            (r.task_id, r.model_id): self.func_space(r) for r in recv_tasks
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
        t0 = time.time()
        batched_send_recv(send_tasks=send_ops, recv_tasks=recv_ops, logger=self.logger)
        t1 = time.time()
        del send_tensors

        self.send_recv_forward_cost += (t1-t0)
        if self.tb_writer is not None:
            self.tb_writer.add_scalar('cnt/send_recv_forward', t1-t0, self._cnt)
            self.logger.debug('cnt/send_recv_forward {}, {}'.format(t1-t0, self._cnt))
        
        # 3) unpack_up extra render_result from other ranks
        extra_render_rets = {
            (r.task_id, r.model_id): self.func_unpack_up(recv_tensors[(r.task_id, r.model_id)]) 
                for r in recv_tasks
        }
        del recv_tensors

        self.saved_for_backward_dict['send_tasks'] = send_tasks
        self.saved_for_backward_dict['recv_tasks'] = recv_tasks
        self.saved_for_backward_dict['render_tasks'] = render_tasks
        self.saved_for_backward_dict['main_rank_tasks'] = main_rank_tasks

        # scheduler is mainly design for communication, let user control the blending 
        return main_rank_tasks, render_rets, extra_render_rets
    
    def render_pass(self, func_render:callable, modelId2rank:dict, _task_main_rank:torch.tensor, _relation_matrix:torch.tensor, views:list):
        send_tasks, recv_tasks, render_tasks, main_rank_tasks = self.task_parser.parse_task_tensor(
            modelId2rank=modelId2rank,
            _task_main_rank = _task_main_rank,
            _relation_matrix = _relation_matrix,
            tasks_message=views
        )
        if self.logger is not None:
            self._log_matching_result(send_tasks, recv_tasks, render_tasks, main_rank_tasks)
        
        # local render_pass:  
        t0 = time.time()
        render_rets = {(t.task_id, t.model_id): None for t in render_tasks}
        for t in render_tasks:
            self.logger.debug('render {}'.format((t.task_id, t.model_id)))
            render_rets[(t.task_id, t.model_id)] = func_render(t)
        t1 = time.time() 
        self.render_pass_cost += (t1-t0)  
        if self.tb_writer is not None:
            self.tb_writer.add_scalar('cnt/render_pass', t1-t0, self._cnt)
            
        # scheduler is mainly design for communication, let user control the blending 
        return render_rets, send_tasks, recv_tasks, render_tasks, main_rank_tasks
    
    def render_batch_pass(self, func_render:callable, modelId2rank:dict, _task_main_rank:torch.tensor, _relation_matrix:torch.tensor, views:list):
        # func_render shall accept: a model_id, a list of render_task with the same model_id
        send_tasks, recv_tasks, render_tasks, main_rank_tasks = self.task_parser.parse_task_tensor(
            modelId2rank=modelId2rank,
            _task_main_rank = _task_main_rank,
            _relation_matrix = _relation_matrix,
            tasks_message=views
        )
        if self.logger is not None:
            self._log_matching_result(send_tasks, recv_tasks, render_tasks, main_rank_tasks)
        
        # local render_pass:  
        t0 = time.time()
        model_id2render_tasks = {}
        for _i in range(len(render_tasks)):
            t:RenderTask = render_tasks[_i]
            if t.model_id not in model_id2render_tasks:
                model_id2render_tasks[t.model_id] = []
            model_id2render_tasks[t.model_id].append(t)   

        model_id2render_batch = {}
        for model_id in model_id2render_tasks:
            model_id2render_batch[model_id] = func_render(model_id, model_id2render_tasks[model_id])

        render_rets = {}     
        for model_id in model_id2render_tasks:
            render_tasks:list = model_id2render_tasks[model_id]
            render_batch:list = model_id2render_batch[model_id]
            for _i in range(len(render_tasks)):
                t:RenderTask = render_tasks[_i]
                ret:dict = render_batch[_i]
                render_rets[(t.task_id, t.model_id)] = ret
   
        t1 = time.time() 
        self.render_pass_cost += (t1-t0)  
        if self.tb_writer is not None:
            self.tb_writer.add_scalar('cnt/render_pass', t1-t0, self._cnt)
            
        # scheduler is mainly design for communication, let user control the blending 
        return render_rets, send_tasks, recv_tasks, render_tasks, main_rank_tasks

    def comm_pass(self, render_rets, send_tasks, recv_tasks, render_tasks, main_rank_tasks, batch_isend_irecv_version=None):
        if batch_isend_irecv_version is None:
            batch_isend_irecv_version = self.batch_isend_irecv_version
        batched_send_recv:callable = BATCHED_SEND_RECV_DICT[batch_isend_irecv_version] 
        # communication_pass:
        # 1) pack_up_render_result and assign storage space for receiving
        send_tensors = {
            (s.task_id, s.model_id): self.func_pack_up(render_rets[(s.task_id, s.model_id)]) 
                for s in send_tasks
        } 
        recv_tensors = {
            (r.task_id, r.model_id): self.func_space(r) for r in recv_tasks
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
        t0 = time.time()
        batched_send_recv(send_tasks=send_ops, recv_tasks=recv_ops, logger=self.logger)
        t1 = time.time()
        del send_tensors

        self.send_recv_forward_cost += (t1-t0)
        if self.tb_writer is not None:
            self.tb_writer.add_scalar('cnt/send_recv_forward', t1-t0, self._cnt)
            self.logger.debug('cnt/send_recv_forward {}, {}'.format(t1-t0, self._cnt))
        
        # 3) unpack_up extra render_result from other ranks
        extra_render_rets = {
            (r.task_id, r.model_id): self.func_unpack_up(recv_tensors[(r.task_id, r.model_id)]) 
                for r in recv_tasks
        }
        del recv_tensors

        self.saved_for_backward_dict['send_tasks'] = send_tasks
        self.saved_for_backward_dict['recv_tasks'] = recv_tasks
        self.saved_for_backward_dict['render_tasks'] = render_tasks
        self.saved_for_backward_dict['main_rank_tasks'] = main_rank_tasks

        return extra_render_rets

    def __loss_pass(self, *args, **kwargs):
        # I hope this joke can relax the pepole reading these codes
        raise NotImplementedError('this scheduler is mainly design for communication :(, please calculate loss by yourself :)')

    def backward_pass(self, extra_render_rets_grad, batch_isend_irecv_version=None):
        '''
        1.1) the extra_render_rets_grad shall be sent back to corresponding rank 
        1.2) as extra_render_rets are received from other ranks, hence they are leaf-nodes in local graph, 
        and their grad is supposed to be created and kept after loss.backward()        

        2.1) current rank would receive grad of its sent-out render rets
        2.2) keep in mind: if the sent-out render rets participate in the loss.backward() called before backward_pass,
        their grad received from other ranks would be passed through local graph a second time in this function
        '''
        if batch_isend_irecv_version is None:
            batch_isend_irecv_version = self.batch_isend_irecv_version
        batched_send_recv:callable = BATCHED_SEND_RECV_DICT[batch_isend_irecv_version]

        send_tasks = self.saved_for_backward_dict['send_tasks']
        recv_tasks = self.saved_for_backward_dict['recv_tasks']
        render_tasks = self.saved_for_backward_dict['render_tasks']
        main_rank_tasks = self.saved_for_backward_dict['main_rank_tasks']

        # communication_grad_pass
        # send_grad_tasks/recv_grad_tasks are symmetrical with recv_tasks/send_tasks
        send_grad = {} 
        for r in recv_tasks:
            try:
                send_grad[(r.task_id, r.model_id)]=self.func_grad_pack_up(
                    extra_render_rets_grad[(r.task_id, r.model_id)])
                if self.logger is not None: 
                    self.logger.debug('find gard for recv_render(task_id={}, model_id={})'.format(r.task_id, r.model_id))
            except Exception as e:
                if self.logger is not None: 
                    self.logger.debug('CAN NOT get gard for recv_render(task_id={}, model_id={})'.format(r.task_id, r.model_id))
                    self.logger.warning(traceback.format_exc())
                # raise e

        recv_grad = {
            (s.task_id, s.model_id): self.func_grad_space(s) for s in send_tasks 
        }
        send_ops = [ 
            (send_grad[(r.task_id, r.model_id)], r.src_rank, None) for r in recv_tasks
        ]
        recv_ops = [
            (recv_grad[(s.task_id, s.model_id)], s.dst_rank, None) for s in send_tasks
        ]

        t0 = time.time()
        batched_send_recv(send_tasks=send_ops, recv_tasks=recv_ops, logger=self.logger)
        t1 = time.time()
        del send_grad

        self.send_recv_backward_cost += (t1-t0)
        if self.tb_writer is not None:
            self.tb_writer.add_scalar('cnt/send_recv_backward', t1-t0, self._cnt)
            self.logger.debug('cnt/send_recv_backward {}, {}'.format(t1-t0, self._cnt))
        # unpack_up_grad
        grad_from_other_rank = {
            (s.task_id, s.model_id): self.func_grad_unpack_up(recv_grad[(s.task_id, s.model_id)])
                for s in send_tasks
        }
        del recv_grad
        # the second time of autograd would backward though local graph 
        # and accumulate the effection of grad_from_other_rank to local model parameters   
        return grad_from_other_rank

    def exchange_GS(self, send_tasks:list, recv_tasks:list, send_dict:dict, recv_dict:dict, batch_isend_irecv_version=None):
        # model can be much larger than render_result
        # thus the memory management can be complex to avoid memory overflow
        # even if the design pattern in forward_pass can still be an alternate
        # Burning insists that user shall design suitable mechanism as this scheduler focuses on communication
        
        # send_tasks:list[SendBoxinGrid], recv_tasks:list[RecvBoxinGrid]
        if batch_isend_irecv_version is None:
            batch_isend_irecv_version = self.batch_isend_irecv_version
        batched_send_recv:callable = BATCHED_SEND_RECV_DICT[batch_isend_irecv_version]

        t0 = time.time()
        send_ops = [ 
            (send_dict[(s.src_model_id, s.dst_model_id)], s.dst_rank, None) for s in send_tasks
        ]
        recv_ops = [
            (recv_dict[(r.src_model_id, r.dst_model_id)], r.src_rank, None) for r in recv_tasks
        ]
        batched_send_recv(send_tasks=send_ops, recv_tasks=recv_ops, logger=self.logger)
        t1 = time.time()
        if self.tb_writer is not None:
            self.tb_writer.add_scalar('resplit/exchange_gs', t1-t0, self._cnt)
        return recv_dict

    def record_info(self):
        self.logger.info('cnt:{}, render:{}, send_recv_forward:{}, send_recv_backward {}'.format(
            self._cnt, self.render_pass_cost, self.send_recv_forward_cost, self.send_recv_backward_cost
        ))


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

