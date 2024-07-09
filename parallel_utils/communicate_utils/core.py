import os
import torch
import torch.distributed
from typing import NamedTuple
import numpy as np
import logging
from torch.profiler import record_function
from typing import List, Callable, Dict
from enum import Enum

# assume all shape is of size 3
def _send_shape(tensor_send: torch.Tensor, dst: int, group: torch.distributed.ProcessGroup):
    shape_tensor = torch.tensor(
        tensor_send.size(), device=torch.cuda.current_device(), dtype=torch.int64
    )
    torch.distributed.send(shape_tensor, dst=dst, group=group)

def send_tensor(tensor_send: torch.Tensor, dst: int, group: torch.distributed.ProcessGroup=None):   
    _send_shape(tensor_send, dst, group)
    torch.distributed.send(tensor_send, dst=dst, group=group)

def _recv_shape(src: int, group: torch.distributed.ProcessGroup):
    shape_tensor = torch.empty(
        (3), device=torch.cuda.current_device(), dtype=torch.int64
    )
    torch.distributed.recv(shape_tensor, src=src, group=group)
    return shape_tensor

def recv_tensor(src: int, group: torch.distributed.ProcessGroup, dtype=torch.float32):
    shape_tensor = _recv_shape()
    tensor_recv = torch.empty(
        shape_tensor, requires_grad=True, device=torch.cuda.current_device(), dtype=dtype,
    )
    return tensor_recv

# use asynchronous function to send/recv batch of tensors
def batched_send_recv_v0(send_tasks, recv_tasks, logger:logging.Logger):
    '''
    assumption: user has arranged the order of tasks 
    '''
    for op in send_tasks:
        logger.debug('send {} to {}'.format(op[0].shape, op[1]))
    for op in recv_tasks:
        logger.debug('recv {} from {}'.format(op[0].shape, op[1]))

    ops = []
    for send_task in send_tasks:
        tensor, dst_peer, group = send_task
        ops.append(torch.distributed.P2POp(torch.distributed.isend, tensor, dst_peer, group))

    for recv_task in recv_tasks:
        tensor, src_peer, group = recv_task
        ops.append(torch.distributed.P2POp(torch.distributed.irecv, tensor, src_peer, group))

    if len(ops) > 0:
        reqs = torch.distributed.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()
    else:
        # "empty operation list can cause error"
        pass 
    # To protect against race condition when using batch_isend_irecv().
    # User should assert that we have a modern enough PyTorch to not need this
    torch.cuda.synchronize()
    logger.debug('batched_send_recv v0')

def batched_send_recv_v0_profiling(send_tasks, recv_tasks, logger:logging.Logger):
    '''
    assumption: user has arranged the order of tasks 
    '''
    for op in send_tasks:
        logger.debug('send {} to {}'.format(op[0].shape, op[1]))
    for op in recv_tasks:
        logger.debug('recv {} from {}'.format(op[0].shape, op[1]))

    with record_function("batched_send_recv_v0"):
        ops = []
        for send_task in send_tasks:
            tensor, dst_peer, group = send_task
            ops.append(torch.distributed.P2POp(torch.distributed.isend, tensor, dst_peer, group))

        for recv_task in recv_tasks:
            tensor, src_peer, group = recv_task
            ops.append(torch.distributed.P2POp(torch.distributed.irecv, tensor, src_peer, group))

        if len(ops) > 0:
            reqs = torch.distributed.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()
        else:
            # "empty operation list can cause error"
            pass 
        # To protect against race condition when using batch_isend_irecv().
        # User should assert that we have a modern enough PyTorch to not need this
        torch.cuda.synchronize()
    logger.debug('batched_send_recv v0')

# use synchronous function to send/recv batch of tensors
def batched_send_recv_v1(send_tasks, recv_tasks, logger:logging.Logger):
    '''
    assumption: user has arranged the order of tasks 
    '''   
    RANK, WORLD_SIZE = torch.distributed.get_rank(), torch.distributed.get_world_size()

    sorted_send_tasks = [[] for r in range(WORLD_SIZE)]
    for s_task in send_tasks:
        tensor, dst_peer, group = s_task
        sorted_send_tasks[dst_peer].append(s_task)

    sorted_recv_tasks = [[] for r in range(WORLD_SIZE)]
    for r_task in recv_tasks:
        tensor, src_peer, group = r_task
        sorted_recv_tasks[src_peer].append(r_task)

    # recv from 0:RANK
    for src in range(0, RANK):
        for r_task in sorted_recv_tasks[src]:
            tensor, src_peer, group = r_task
            torch.distributed.recv(tensor, src_peer, group)
            logger.debug('recv {} from {}'.format(tensor.shape, src_peer))
    logger.debug('recv from 0:RANK')        

    # send to RANK:WORLD_SIZE
    for dst in range(RANK+1, WORLD_SIZE):
        for s_task in sorted_send_tasks[dst]:
            tensor, dst_peer, group = s_task
            torch.distributed.send(tensor, dst_peer, group)
            logger.debug('send {} to {}'.format(tensor.shape, dst_peer))
    logger.debug('send to RANK:WORLD_SIZE')         

    # recv from WORLD_SIZE-1, RANK
    for src in range(WORLD_SIZE-1, RANK, -1):
        for r_task in sorted_recv_tasks[src]:
            tensor, src_peer, group = r_task
            torch.distributed.recv(tensor, src_peer, group)
            logger.debug('recv {} from {}'.format(tensor.shape, src_peer))
    logger.debug('recv from WORLD_SIZE-1, RANK')        

    # send to RANK-1:-1
    for dst in range(RANK-1, -1, -1):
        for s_task in sorted_send_tasks[dst]:
            tensor, dst_peer, group = s_task
            torch.distributed.send(tensor, dst_peer, group)
            logger.debug('send {} to {}'.format(tensor.shape, dst_peer))
    logger.debug('send to RANK-1:-1')         

    # To protect against race condition when using batch_isend_irecv().
    # User should assert that we have a modern enough PyTorch to not need this
    torch.cuda.synchronize()
    logger.debug('batched_send_recv v1')        

# use asynchronous function to send/recv batch of tensors
def batched_send_recv_v2(send_tasks, recv_tasks, logger:logging.Logger):
    RANK, WORLD_SIZE = torch.distributed.get_rank(), torch.distributed.get_world_size()

    sorted_send_tasks = [[] for r in range(WORLD_SIZE)]
    for s_task in send_tasks:
        tensor, dst_peer, group = s_task
        sorted_send_tasks[dst_peer].append(s_task)

    sorted_recv_tasks = [[] for r in range(WORLD_SIZE)]
    for r_task in recv_tasks:
        tensor, src_peer, group = r_task
        sorted_recv_tasks[src_peer].append(r_task)

    reqs = []
    # recv from 0:RANK
    for src in range(0, RANK):
        for r_task in sorted_recv_tasks[src]:
            tensor, src_peer, group = r_task
            req = torch.distributed.irecv(tensor, src_peer, group)
            reqs.append(req)
            logger.debug('recv {} from {}'.format(tensor.shape, src_peer))
    logger.debug('recv from 0:RANK')         

    # send to RANK:WORLD_SIZE
    for dst in range(RANK+1, WORLD_SIZE):
        for s_task in sorted_send_tasks[dst]:
            tensor, dst_peer, group = s_task
            req = torch.distributed.isend(tensor, dst_peer, group)
            reqs.append(req)
            logger.debug('send {} to {}'.format(tensor.shape, dst_peer))
    logger.debug('send to RANK:WORLD_SIZE')   


    if len(reqs) > 0:
        for req in reqs:
            req.wait()

    torch.cuda.synchronize()      
    logger.debug('left 2 right pass')

    reqs = []
    # recv from WORLD_SIZE-1, RANK
    for src in range(WORLD_SIZE-1, RANK, -1):
        for r_task in sorted_recv_tasks[src]:
            tensor, src_peer, group = r_task
            req = torch.distributed.irecv(tensor, src_peer, group)
            reqs.append(req)
            logger.debug('recv {} from {}'.format(tensor.shape, src_peer))
    logger.debug('recv from WORLD_SIZE-1, RANK')         

    # send to RANK-1:-1
    for dst in range(RANK-1, -1, -1):
        for s_task in sorted_send_tasks[dst]:
            tensor, dst_peer, group = s_task
            req = torch.distributed.isend(tensor, dst_peer, group)
            reqs.append(req)
            logger.debug('send {} to {}'.format(tensor.shape, dst_peer))
    logger.debug('send to RANK-1:-1')         

    # To protect against race condition when using batch_isend_irecv().
    # User should assert that we have a modern enough PyTorch to not need this
    if len(reqs) > 0:
        for req in reqs:
            req.wait()

    torch.cuda.synchronize()
    logger.debug('right 2 left pass')
    logger.debug('batched_send_recv v2')    

BATCHED_SEND_RECV_DICT = {
    0: batched_send_recv_v0,
    1: batched_send_recv_v1,
    2: batched_send_recv_v2,
    '0': batched_send_recv_v0,
    '1': batched_send_recv_v1,
    '2': batched_send_recv_v2,
    '0+profiling': batched_send_recv_v0_profiling,
}

def get_batched_send_recv_function(version):
    return BATCHED_SEND_RECV_DICT[version]