import os
import torch.distributed as dist
from abc import ABC, abstractmethod
from scene.cameras import Camera, ViewMessage, Patch
from typing import NamedTuple
from parallel_utils.grid_utils.utils import BoxinGrid3D


class SendTask(NamedTuple):
    task_id: int
    model_id: int
    src_rank: int
    dst_rank: int
    task: ViewMessage 


class RecvTask(NamedTuple):
    task_id: int
    model_id: int
    src_rank: int
    dst_rank: int
    task: ViewMessage   


class RenderTask(NamedTuple):
    task_id: int
    model_id: int
    task: ViewMessage 
    
    
# A data sample might meed a main rank to assemble data from all related rank
class MainRankTask(NamedTuple):
    task_id: int
    rank: int
    task: ViewMessage 


# send extra gs info
class SendExtraGS(NamedTuple):
    src_model_id: int
    dst_model_id: int
    src_rank: int
    dst_rank: int
    grid_range: BoxinGrid3D # box of dst_model
    length: int


class RecvExtraGS(NamedTuple):
    src_model_id: int
    dst_model_id: int
    src_rank: int
    dst_rank: int
    grid_range: BoxinGrid3D
    length: int