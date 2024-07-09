import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
import logging, time
from typing import NamedTuple, Callable, List, Dict, Any, Tuple
import traceback

from scene.cameras import Camera
from parallel_utils.schedulers.core import Model2ModelRecv, Model2ModelSend, ModelForwardTask
from parallel_utils.schedulers.basic_scheduler import RankModelInfo

class SendRenderResult(Model2ModelSend):
    def __init__(self, info:Camera, task_id, src_rank: int, dst_rank: int, src_model: int=-1, dst_model: int=-1, group: ProcessGroup=None) -> None:
        super().__init__(info, task_id, src_rank, dst_rank, src_model, dst_model, group)

class RecvRenderResult(Model2ModelRecv):
    def __init__(self, info:Camera, task_id, src_rank: int, dst_rank: int, src_model: int=-1, dst_model: int=-1, group: ProcessGroup=None) -> None:
        super().__init__(info, task_id, src_rank, dst_rank, src_model, dst_model, group)

class SendSharedGSInfo(Model2ModelSend):
    def __init__(self, info:int, task_id, src_rank: int, dst_rank: int, src_model: int, dst_model: int, group: ProcessGroup) -> None:
        super().__init__(info, task_id, src_rank, dst_rank, src_model, dst_model, group)
        # self.info is num of gs primitives

class RecvSharedGSInfo(Model2ModelRecv):
    def __init__(self, info:int, task_id, src_rank: int, dst_rank: int, src_model: int, dst_model: int, group: ProcessGroup) -> None:
        super().__init__(info, task_id, src_rank, dst_rank, src_model, dst_model, group) 
        # self.info is num of gs primitives

class SendGSParameters(Model2ModelSend):
    def __init__(self, info:int, task_id, src_rank: int, dst_rank: int, src_model: int, dst_model: int, group: ProcessGroup) -> None:
        super().__init__(info, task_id, src_rank, dst_rank, src_model, dst_model, group)
        # self.info is num of gs primitives

class RecvGSParameters(Model2ModelRecv):
    def __init__(self, info:int, task_id, src_rank: int, dst_rank: int, src_model: int, dst_model: int, group: ProcessGroup) -> None:
        super().__init__(info, task_id, src_rank, dst_rank, src_model, dst_model, group) 
        # self.info is num of gs primitives

class RenderTask(ModelForwardTask):
    def __init__(self, info:Camera, task_id, model: int, rank: int) -> None:
        super().__init__(info, task_id, model, rank)

class MainRankTask(ModelForwardTask):
    def __init__(self, info:Camera, task_id, model: int=-1, rank: int=-1) -> None:
        super().__init__(info, task_id, model, rank)

class BasicTaskParser:
    def __init__(self, PROCESS_WORLD_SIZE:int, GLOBAL_RANK:int, logger: logging.Logger) -> None:
        """
        PROCESS_WORLD_SIZE: global information for all rank, provided by dist package 
        """
        self.PROCESS_WORLD_SIZE = PROCESS_WORLD_SIZE
        self.GLOBAL_RANK = GLOBAL_RANK
        self.logger = logger 

    def parse_render_relation(self, rkmd:RankModelInfo, _task_main_rank:torch.Tensor, _relation_matrix:torch.Tensor, tasks_message:List[Camera], role:int=None):
        rank2modelIds, model2rank = rkmd.rank2models, rkmd.model2rank
        task_main_rank, relation_matrix = _task_main_rank.cpu(), _relation_matrix.cpu()
        ROLE = role if role is not None else self.GLOBAL_RANK
        NUM_ALL_MODELS = len(model2rank)    # _relation_matrix.shape[1]
        batch_size = task_main_rank.shape[0]

        assert ROLE in rank2modelIds
        assert relation_matrix.shape[0] == len(tasks_message) == batch_size and relation_matrix.shape[1] == NUM_ALL_MODELS

        send_tasks: List[SendRenderResult] = []
        recv_tasks: List[RecvRenderResult] = [] 
        render_tasks: List[RenderTask] = [] 
        main_rank_tasks: List[MainRankTask] = []
        # render_tasks, main_rank_tasks
        for id_in_batch in range(batch_size):
            task_id, main_rank, msg = int(task_main_rank[id_in_batch, 0]), int(task_main_rank[id_in_batch, 1]), tasks_message[id_in_batch]
            for model_id in rank2modelIds[ROLE]:
                if relation_matrix[id_in_batch, model_id] >= 0:
                    # render for related task
                    render_tasks.append(RenderTask(info=msg, task_id=task_id, model=model_id, rank=model2rank[model_id]))
                    # send render result to main_rank if current role is not main_rank 
                    # as rank2modelIds had been sorted in __init__, send tasks of the same rank are in model_id order 
                    if ROLE != main_rank:
                        send_tasks.append(SendRenderResult(info=msg, task_id=task_id, src_rank=ROLE, dst_rank=main_rank, src_model=model_id))    

            if main_rank == ROLE:
                main_rank_tasks.append(MainRankTask(info=msg, task_id=task_id, rank=ROLE))
                # organize recv tasks in model_id order as send tasks of the same rank are in model_id order
                for model_id in range(NUM_ALL_MODELS):
                    if relation_matrix[id_in_batch, model_id] >= 0:
                        src_rank = model2rank[model_id]
                        # main_rank has not need to recv from itself (to save resource cost)  
                        if src_rank != ROLE: 
                            recv_tasks.append(RecvRenderResult(info=msg, task_id=task_id, src_rank=src_rank, dst_rank=ROLE, src_model=model_id))
        
        return send_tasks, recv_tasks, render_tasks, main_rank_tasks
    
    def parse_model2model_task(self, src_rkmd:RankModelInfo, dst_rkmd:RankModelInfo, GLOBAL_SEND_AMOUNT:torch.Tensor, role:int=None):
        src_modelId2rank, rank2src_modelIds = src_rkmd.model2rank, src_rkmd.rank2models
        dst_modelId2rank, rank2dst_modelIds = dst_rkmd.model2rank, dst_rkmd.rank2models

        ROLE = role if role is not None else self.GLOBAL_RANK
        NUM_SRC_MODELS, NUM_DST_MODELS = len(src_modelId2rank), len(dst_modelId2rank)
        assert NUM_SRC_MODELS==GLOBAL_SEND_AMOUNT.shape[0] and NUM_DST_MODELS==GLOBAL_SEND_AMOUNT.shape[1]
        # lets just use Model2ModelSend, Model2ModelRecv

        send_tasks:List[Model2ModelSend] = []
        recv_tasks:List[Model2ModelRecv] = []
        # let organize it in (src_model, dst_model) order
        for src_model in rank2src_modelIds[ROLE]:
            for dst_model in range(NUM_DST_MODELS):
                src_rank, dst_rank = src_modelId2rank[src_model], dst_modelId2rank[dst_model]
                if GLOBAL_SEND_AMOUNT[src_model, dst_model] > 0 and (src_rank != dst_rank):
                    send_tasks.append(Model2ModelSend(
                        info=GLOBAL_SEND_AMOUNT[src_model, dst_model], task_id=-1, src_rank=src_rank, dst_rank=dst_rank, src_model=src_model, dst_model=dst_model, group=None
                       ))

        for src_model in range(NUM_SRC_MODELS):
            for dst_model in rank2dst_modelIds[ROLE]:  
                src_rank, dst_rank = src_modelId2rank[src_model], dst_modelId2rank[dst_model]
                if GLOBAL_SEND_AMOUNT[src_model, dst_model] > 0 and (src_rank != dst_rank):
                    recv_tasks.append(Model2ModelRecv(
                        info=GLOBAL_SEND_AMOUNT[src_model, dst_model], task_id=-1, src_rank=src_rank, dst_rank=dst_rank, src_model=src_model, dst_model=dst_model, group=None
                       ))  
        return send_tasks, recv_tasks 
    

'''
classes with zip/unzip methods
    + serve as the output of ModelForwardTask
    + are transportable to BasicScheduler as zip/space/unzip methods are implemented
'''
def fetch_grad(t: torch.Tensor):
    RANK = dist.get_rank()
    if t.grad is None:
        # logging.getLogger('rank_{}'.format(RANK)).warn('no grad for tesnor of shape {}'.format(t.shape))
        return torch.zeros_like(t, dtype=torch.float32)
    else:
        # logging.getLogger('rank_{}'.format(RANK)).warn('find grad for tesnor of shape {}'.format(t.shape))
        return t.grad


class RenderResult:
    def __init__(self, raw_dict:Dict[str, torch.Tensor]) -> None:
        assert {'render', 'alpha', 'depth'}.issubset(raw_dict.keys()), 'key missing'
        self.render = raw_dict['render']
        self.alpha = raw_dict['alpha']
        self.depth = raw_dict['depth']
        self.raw = raw_dict

    def make_copy(self):
        return RenderResult({
            "render": self.render.clone().detach().requires_grad_(True),
            "depth": self.depth.clone().detach().requires_grad_(True),
            "alpha": self.alpha.clone().detach().requires_grad_(True)
        })

    @staticmethod
    def zip_content(render_result):
        render_ret:RenderResult = render_result
        with torch.no_grad():
            # (5, H, W) tensor
            pkg = torch.cat(
                [render_ret.render, render_ret.depth, render_ret.alpha], dim=0
            ).to(dtype=torch.float32) 
        return pkg

    @staticmethod
    def space_for_content(rrr:RecvRenderResult, device='cuda', only_shape=False):
        task:Camera = rrr.info
        H, W = task.image_height, task.image_width
        if only_shape:
            return (5, H, W)
        else:
            return torch.zeros((5, H, W), dtype=torch.float32, device=device, requires_grad=False)

    @staticmethod
    def unzip_content(pkg:torch.Tensor):
        # inversion of zip_content
        # Do not forget to set requires_grad, space-allocating function may not do it 
        assert pkg.shape[0] == 5 and len(pkg.shape) == 3
        return RenderResult({
            "render": pkg[:3].clone().detach().requires_grad_(True),
            "depth": pkg[3:4].clone().detach().requires_grad_(True),
            "alpha": pkg[4:5].clone().detach().requires_grad_(True)
        })

    @staticmethod
    def zip_grad(render_result):
        render_ret:RenderResult = render_result
        with torch.no_grad():
            pkg = torch.cat(
                [fetch_grad(render_ret.render), fetch_grad(render_ret.depth), fetch_grad(render_ret.alpha)], 
                dim=0
            )
        return pkg
    
    @staticmethod
    def space_for_grad(rrr:SendRenderResult, device='cuda', only_shape=False):
        task:Camera = rrr.info
        H, W = task.image_height, task.image_width
        if only_shape:
            return (5, H, W)
        else:
            return torch.zeros((5, H, W), dtype=torch.float32, device=device, requires_grad=False)
    
    @staticmethod
    def unzip_grad(pkg:torch.Tensor):
        assert pkg.shape[0] == 5 and len(pkg.shape) == 3
        return RenderResult({
            "render": pkg[:3],
            "depth": pkg[3:4],
            "alpha": pkg[4:5]
        })

    @staticmethod
    def gather_paired_tensor_grad_list(ret, grad=None):
        if grad is not None:
            assert isinstance(ret, RenderResult) and isinstance(grad, RenderResult)
            return [ret.render, ret.alpha, ret.depth], [grad.render, grad.alpha, grad.depth]
        else:
            assert isinstance(ret, RenderResult)
            pairs = [(ret.render, ret.render.grad), (ret.alpha, ret.alpha.grad), (ret.depth, ret.depth.grad)]
            return [t for t,g in pairs if g is not None], [g for t,g in pairs if g is not None]
         
    @staticmethod
    def gather_paired_tensor_grad_list_from_copy(ret, copy):
        assert isinstance(ret, RenderResult) and isinstance(copy, RenderResult)
        pairs = [(ret.render, copy.render.grad), (ret.alpha, copy.alpha.grad), (ret.depth, copy.depth.grad)]
        return [t for t,g in pairs if g is not None], [g for t,g in pairs if g is not None]
       

class SharedGSInfo:
    def __init__(
        self, 
        means3D:torch.Tensor, 
        means2D:torch.Tensor, 
        opacity:torch.Tensor,
        scales:torch.Tensor, 
        rotations:torch.Tensor, 
        shs:torch.Tensor,
    ) -> None:
        self.means3D = means3D
        self.means2D = means2D
        self.opacity = opacity
        self.scales = scales
        self.rotations = rotations
        self.shs = shs  # (num, channel, 3)

    @staticmethod
    def check_dim(num_channel):
        c_mean3d, c_mean2d = 3, 3
        c_sh_dc = 1
        c_sh = (c_sh_rest + c_sh_dc)*3
        c_opacity, c_scales, c_rotations = 1, 3, 4

        channel2degree = {}
        for sh_degree in range(4):
            c_sh_rest = (sh_degree + 1) ** 2 - 1
            all_channel = (c_mean3d + c_mean2d + c_opacity + c_scales + c_rotations + c_sh)
            channel2degree[all_channel] = sh_degree 

        if num_channel not in channel2degree:
            return None
        else:
            return channel2degree[num_channel]    
        
    @staticmethod
    def zip_content(info):
        assert isinstance(info, SharedGSInfo)
        NUM_GS = info.shs.shape[0]
        ret = torch.cat(
            [info.means3D, info.means2D, info.opacity, info.scales, info.rotations, info.shs.view(NUM_GS, -1)],
            dim=-1).to(dtype=torch.float32).contiguous()
        return ret

    @staticmethod
    def space_for_content(msg:list, device='cuda', only_shape=False):
        rsg:RecvSharedGSInfo = msg[0]
        sh_degree:int = msg[1]

        c_mean3d, c_mean2d = 3, 3
        c_sh_rest = (sh_degree + 1) ** 2 - 1
        c_sh_dc = 1
        c_sh = (c_sh_rest + c_sh_dc)*3
        c_opacity, c_scales, c_rotations = 1, 3, 4
        all_channel = (c_mean3d + c_mean2d + c_opacity + c_scales + c_rotations + c_sh)
       
        length:int = rsg.info
        if only_shape:
            return (length, all_channel)
        else:
            return torch.zeros((length, all_channel), dtype=torch.float32, device=device, requires_grad=False)
    
    @staticmethod
    def unzip_content(pkg:torch.Tensor):
        NUM_GS, all_channel = pkg.shape[0], pkg.shape[1]
        c_mean3d, c_mean2d = 3, 3
        c_opacity, c_scales, c_rotations = 1, 3, 4
        c_sh = all_channel - (c_mean3d + c_mean2d + c_opacity + c_scales + c_rotations)
        return SharedGSInfo(
            means3D=pkg[:, 0:3].clone().detach().requires_grad_(True).contiguous(),
            means2D=pkg[:, 3:6].clone().detach().requires_grad_(True).contiguous(),
            opacity=pkg[:, 6:7].clone().detach().requires_grad_(True).contiguous(),
            scales=pkg[:, 7:10].clone().detach().requires_grad_(True).contiguous(),
            rotations=pkg[:, 10:14].clone().detach().requires_grad_(True).contiguous(),
            shs=pkg[:, 14:].reshape(NUM_GS, c_sh//3, 3).contiguous().clone().detach().requires_grad_(True).contiguous()
        )

    @staticmethod
    def zip_grad(info):
        assert isinstance(info, SharedGSInfo)
        NUM_GS = info.shs.shape[0]
        ret = torch.cat(
            [fetch_grad(info.means3D), fetch_grad(info.means2D), fetch_grad(info.opacity), fetch_grad(info.scales), fetch_grad(info.rotations), fetch_grad(info.shs).view(NUM_GS, -1)],
            dim=-1).to(dtype=torch.float32).contiguous()
        return ret

    @staticmethod
    def space_for_grad(msg:list, device='cuda', only_shape=False):
        rsg:SendSharedGSInfo = msg[0]
        sh_degree:int = msg[1]
        c_mean3d, c_mean2d = 3, 3
        c_sh_rest = (sh_degree + 1) ** 2 - 1
        c_sh_dc = 1
        c_sh = (c_sh_rest + c_sh_dc)*3
        c_opacity, c_scales, c_rotations = 1, 3, 4
        all_channel = (c_mean3d + c_mean2d + c_opacity + c_scales + c_rotations + c_sh)
       
        length:int = rsg.info
        if only_shape:
            return (length, all_channel)
        else:
            return torch.zeros((length, all_channel), dtype=torch.float32, device=device, requires_grad=False)

    @staticmethod
    def unzip_grad(pkg:torch.Tensor):
        NUM_GS, all_channel = pkg.shape[0], pkg.shape[1]
        c_mean3d, c_mean2d = 3, 3
        c_opacity, c_scales, c_rotations = 1, 3, 4
        c_sh = all_channel - (c_mean3d + c_mean2d + c_opacity + c_scales + c_rotations)
        return SharedGSInfo(
            means3D=pkg[:, 0:3].contiguous(),
            means2D=pkg[:, 3:6].contiguous(),
            opacity=pkg[:, 6:7].contiguous(),
            scales=pkg[:, 7:10].contiguous(),
            rotations=pkg[:, 10:14].contiguous(),
            shs=pkg[:, 14:].reshape(NUM_GS, c_sh//3, 3).contiguous()
        )

    @staticmethod
    def gather_paired_tensor_grad_list(ret, grad=None):
        if grad is not None:
            assert isinstance(ret, SharedGSInfo) and isinstance(grad, SharedGSInfo)
            return [ret.means3D, ret.means2D, ret.opacity, ret.scales, ret.rotations, ret.shs], [grad.means3D, grad.means2D, grad.opacity, grad.scales, grad.rotations, grad.shs]
        else:
            assert isinstance(ret, SharedGSInfo)
            pairs = [
                (ret.means3D, ret.means3D.grad), 
                (ret.means2D, ret.means2D.grad), 
                (ret.opacity, ret.opacity.grad), 
                (ret.scales, ret.scales.grad), 
                (ret.rotations, ret.rotations.grad), 
                (ret.shs, ret.shs.grad)]
            return [t for t,g in pairs if g is not None], [g for t,g in pairs if g is not None]


class GSParametsers:
    def __init__(self, pkg:torch.Tensor) -> None:
        self.pkg = pkg

    @staticmethod
    def get_channel_size(sh_degree:int):
        rest_channel = (sh_degree + 1) ** 2 - 1
        all_channel = 14 + 3*rest_channel
        return 3*all_channel

    @staticmethod
    def zip_content(info):   
        assert isinstance(info, GSParametsers)
        return info.pkg

    @staticmethod
    def space_for_content(msg:list, device='cuda', only_shape=False):
        recv:RecvGSParameters = msg[0]
        sh_degree:int = msg[1]

        rest_channel = (sh_degree + 1) ** 2 - 1
        all_channel = 14 + 3*rest_channel

        length:int = recv.info
        if only_shape:
            return (length, 3*all_channel)
        else:
            return torch.zeros((length, 3*all_channel), dtype=torch.float32, device=device, requires_grad=False)
        
    @staticmethod  
    def unzip_content(pkg:torch.Tensor):  
        return GSParametsers(pkg)

