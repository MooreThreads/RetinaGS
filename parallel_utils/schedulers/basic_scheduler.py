'''
RankModelInfo:
    + provide type-hint of rank2model, model2rank

TaskParser:
    + find kinds of tasks assigned to models/process
    + give a poper sequence (valid sequence is not unique) for the send/recv tasks

Scheduler:
    + provide logic that organizes the exchanging of messages, rarely maintain status
    + yes, it can be broken into some functions 
'''

import os
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
import numpy as np

import logging, time
from typing import NamedTuple, Callable, List, Dict, Any
import traceback

from scene.cameras import Camera
from parallel_utils.grid_utils.utils import BoxinGrid3D
from parallel_utils.schedulers.core import Model2ModelRecv, Model2ModelSend, ModelForwardTask

from parallel_utils.communicate_utils.core import get_batched_send_recv_function


class RankModelInfo:
    # it records the location of models, but not initilaize/assign/call models (let trainer do these) 
    def __init__(self, num_ranks:int, num_models:int, model2rank:Dict[int, int]) -> None:
        self.num_ranks = num_ranks
        self.num_models = num_models
        self.model2rank = model2rank
        self.rank2models:Dict[int, List[int]] = {i:[] for i in range(self.num_ranks)}
        for model_id, _r in model2rank.items():
            self.rank2models[_r].append(model_id)
        for _r in self.rank2models:
            self.rank2models[_r].sort() 


class BasicScheduler:
    def __init__(self, logger: logging.Logger, tb_writer = None, batch_isend_irecv_version=0, group:ProcessGroup=None) -> None:
        self.logger = logger
        self.tb_writer = tb_writer
        self.batch_isend_irecv_version = batch_isend_irecv_version
        self.group = group

    def model2model_exchange_forward(
        self, 
        content_pool: Dict[Any, Any], # hash(Model2ModelSend|Model2ModelRecv): ContentType
        space_pool: Dict[Any, torch.Tensor], # hash(Model2ModelSend|Model2ModelRecv): Tensor
        sorted_send_tasks: List[Model2ModelSend], 
        sorted_recv_tasks: List[Model2ModelRecv],
        func_zip: Callable[[Any], torch.Tensor],
        func_unzip: Callable[[torch.Tensor], Any],
        batch_isend_irecv_version=None
    ):
        '''
        + caller shall provide content-tensor and space-tensor for exchanging    
        + this function can be considered as a forward operation: content, new_content = f(content, blank_space, *config_args)
        '''    
        if batch_isend_irecv_version is None:
            batch_isend_irecv_version = self.batch_isend_irecv_version

        # prepare tenors 
        send_tensors = {task.key(): func_zip(content_pool[task.key()]) for task in sorted_send_tasks} 
        recv_tensors = space_pool
        # send&recv
        self.__model2model_tensor_exchange(send_tensors, recv_tensors, sorted_send_tasks, sorted_recv_tasks, batch_isend_irecv_version)
        # unzip tensor
        new_content_dict = {task.key(): func_unzip(recv_tensors[task.key()]) for task in sorted_recv_tasks}
        # release send_tensors, let called handle space_pool
        del send_tensors
        return new_content_dict
    
    def model2model_exchange_backward(
        self,
        content_grad_pool: Dict[Any, Any], # hash(Model2ModelSend|Model2ModelRecv): ContentType
        space_grad_pool: Dict[Any, torch.Tensor], # hash(Model2ModelSend|Model2ModelRecv): Tensor
        sorted_send_tasks: List[Model2ModelSend], 
        sorted_recv_tasks: List[Model2ModelRecv],
        func_zip: Callable[[Any], torch.Tensor],
        func_unzip: Callable[[torch.Tensor], Any],
        batch_isend_irecv_version=None
    ):
        '''
        + backward of model2model_exchange_forward
        + the model2model_exchange operation pair involve nccl, 
                burning is not sure about whether they cause bugs if registed as torch function,
                thus they are manually called
        + burning is a lazy guy, he usually chooses a simple and verified way      
        '''
        if batch_isend_irecv_version is None:
            batch_isend_irecv_version = self.batch_isend_irecv_version
        batched_send_recv:Callable = get_batched_send_recv_function(batch_isend_irecv_version)
        # collect send-grad corresponding to recv-new-content 
        send_grad = {} 
        for r in sorted_recv_tasks:
            try:
                send_grad[r.key()]=func_zip(content_grad_pool[r.key()])
                if self.logger is not None: 
                    self.logger.debug('find gard for recv_render(key={})'.format(r.key()))
            except Exception as e:
                if self.logger is not None: 
                    self.logger.debug('CAN NOT get gard for recv_render(key={})'.format(r.key()))
                    self.logger.warning(traceback.format_exc())
                # raise e
        recv_grad = space_grad_pool
        # send&recv
        # send_grad_tasks/recv_grad_tasks are symmetrical with send_tasks/recv_tasks
        # a-send_content->b : a<-recv_grad-b
        
        send_ops = [(send_grad[r.key()], r.src_rank, self.group) for r in sorted_recv_tasks]
        recv_ops = [(recv_grad[s.key()], s.dst_rank, self.group) for s in sorted_send_tasks]
        batched_send_recv(send_tasks=send_ops, recv_tasks=recv_ops, logger=self.logger)
        
        # unzip tensor
        new_grad_dict = {task.key(): func_unzip(recv_grad[task.key()]) for task in sorted_send_tasks}
        del send_grad
        return new_grad_dict

    def __model2model_tensor_exchange(
        self,
        content_tensors: Dict[Any, torch.Tensor], 
        space_tensors: Dict[Any, torch.Tensor], 
        sorted_send_tasks: List[Model2ModelSend], 
        sorted_recv_tasks: List[Model2ModelRecv],
        batch_isend_irecv_version=None
    ):
        '''
        + exchange tensor data between ranks/models
        + gradients of tensors are not transmitted, this operation treats tensors as C-like array  
        '''
        send_ops = [(content_tensors[s.key()], s.dst_rank, self.group) for s in sorted_send_tasks]
        recv_ops = [(space_tensors[r.key()],   r.src_rank, self.group) for r in sorted_recv_tasks]
        
        if batch_isend_irecv_version is None:
            batch_isend_irecv_version = self.batch_isend_irecv_version

        batched_send_recv:Callable = get_batched_send_recv_function(batch_isend_irecv_version)
        batched_send_recv(send_tasks=send_ops, recv_tasks=recv_ops, logger=self.logger)
        return None
    

    def assign_space4recv(self, recv_tasks:List[Model2ModelRecv], get_space:Callable[[Model2ModelRecv], torch.Tensor]):
        '''
            For some recv_task, process can predict space size with the .info property
            recv_tasks:List[Model2ModelRecv|Model2ModelSend], get_space:Callable[[Model2ModelRecv|Model2ModelSend], torch.Tensor]
        '''
        ret = {task.key():get_space(task) for task in recv_tasks}
        return ret
    
    def broadcast_global_model2model_info(self, src_rkmd:RankModelInfo, SEND_AMOUNT_CPU:torch.Tensor):    
        '''
            For some Model2Model like(RecvSharedGSInfo/RecvGSParameters), recv-process need extra shape information (like num_gs) from send_process
            this function synchroizes model2model_send_info in a broadcast pattern
            + global_send_size_np is array of [num_src_model, num_dst_model], model_i would sends elemet[i, j] Gaussian-Splattings to model_j
            + every rank writes its own lines to tell other ranks about its content to send
        '''
        RANK, WORLD_SIZE = dist.get_rank(), dist.get_world_size()
        NUM_SRC_MODEL, NUM_DST_MODEL = SEND_AMOUNT_CPU.shape[0], SEND_AMOUNT_CPU.shape[1]
        self.logger.debug('broadcast_global_model2model_info, rank {}, num_src {}, num_dst {}'.format(RANK, NUM_SRC_MODEL, NUM_DST_MODEL))

        local_line_idx = src_rkmd.rank2models[RANK]
        GLOBAL_SEND_AMOUNT_CPU = torch.zeros((NUM_SRC_MODEL, NUM_DST_MODEL), dtype=torch.int, device='cpu')
        GLOBAL_SEND_AMOUNT_CPU[local_line_idx] = SEND_AMOUNT_CPU[local_line_idx]
        # broadcast by allreduce
        GLOBAL_SEND_AMOUNT = GLOBAL_SEND_AMOUNT_CPU.cuda()
        dist.all_reduce(GLOBAL_SEND_AMOUNT, op=dist.ReduceOp.SUM, group=self.group, async_op=False)
        GLOBAL_SEND_AMOUNT_CPU = GLOBAL_SEND_AMOUNT.cpu()
        # check whether local_lines are corrupted
        assert torch.all(SEND_AMOUNT_CPU[local_line_idx]==GLOBAL_SEND_AMOUNT_CPU[local_line_idx]), 'local rows are written by another rank'
        self.logger.debug('GLOBAL_SEND_AMOUNT_CPU is \n{}'.format(GLOBAL_SEND_AMOUNT_CPU))
        return GLOBAL_SEND_AMOUNT_CPU
    
