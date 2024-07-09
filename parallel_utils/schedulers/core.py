from abc import ABC, abstractmethod

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup


class NcclSendTask():
    def __init__(self, task_id, src_rank:int, dst_rank:int, group:dist.ProcessGroup) -> None:
        self.task_id = task_id
        self.src_rank = src_rank
        self.dst_rank = dst_rank
        self.group = group
    
class NcclRecvTask():
    def __init__(self, task_id, src_rank:int, dst_rank:int, group:ProcessGroup) -> None:
        self.task_id = task_id
        self.src_rank = src_rank
        self.dst_rank = dst_rank
        self.group = group

class TransportableData(ABC):
    @staticmethod
    @abstractmethod
    def zip_content(obj) -> torch.Tensor:
        pass

    @staticmethod
    @abstractmethod
    def space_for_content(msg, device, only_shape):
        pass

    @staticmethod
    @abstractmethod
    def unzip_content(pkg:torch.Tensor):
        pass

    @staticmethod
    @abstractmethod
    def zip_grad(obj) -> torch.Tensor:
        pass

    @staticmethod
    @abstractmethod
    def space_for_grad(msg, device, only_shape):
        pass

    @staticmethod
    @abstractmethod
    def unzip_grad(pkg: torch.Tensor):
        pass

class Model2ModelSend(NcclSendTask):
    def __init__(self, info, task_id, src_rank: int, dst_rank: int, src_model: int, dst_model: int, group: dist.ProcessGroup) -> None:
        super().__init__(task_id, src_rank, dst_rank, group)
        self.src_model = src_model
        self.dst_model = dst_model
        self.info = info
        self._key = (int(self.task_id), self.src_rank, self.dst_rank, self.src_model, self.dst_model)

    def key(self) -> tuple:
        return self._key

    def __str__(self) -> str:
        return  '{}(taskId:{}, src_rank:{}, dst_rank:{}, src_model:{}, dst_model:{})'.format(
            type(self).__name__, self.task_id, self.src_rank, self.dst_rank, self.src_model, self.dst_model
        )    

class Model2ModelRecv(NcclRecvTask):
    def __init__(self, info, task_id, src_rank: int, dst_rank: int, src_model: int, dst_model: int, group: dist.ProcessGroup) -> None:
        super().__init__(task_id, src_rank, dst_rank, group)
        self.src_model = src_model
        self.dst_model = dst_model
        self.info = info
        self._key = (int(self.task_id), self.src_rank, self.dst_rank, self.src_model, self.dst_model)

    def key(self) -> tuple:
        return self._key   
    
    def __str__(self) -> str:
        return  '{}(taskId:{}, src_rank:{}, dst_rank:{}, src_model:{}, dst_model:{})'.format(
            type(self).__name__, self.task_id, self.src_rank, self.dst_rank, self.src_model, self.dst_model
        )  

class ModelForwardTask():
    def __init__(self, info, task_id, model: int, rank: int) -> None:
        self.info = info
        self.task_id = task_id
        self.model = model
        self.rank = rank
        self._key = (int(self.task_id), self.model, self.rank)

    def key(self) -> tuple:
        return self._key

    def __str__(self) -> str:
        return  '{}(taskId:{}, rank:{},model:{})'.format(
            type(self).__name__, self.task_id, self.rank, self.model
        )  
