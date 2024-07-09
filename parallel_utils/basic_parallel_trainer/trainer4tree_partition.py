import os, sys
import traceback, uuid, logging, time, shutil, glob
from tqdm import tqdm
import numpy as np
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import torch, torchvision, cv2, datetime
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from typing import List, Dict, Tuple, Union

os.environ["NCCL_SOCKET_TIMEOUT"] = "60000"

from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from utils.general_utils import safe_state, is_interval_in_batch, is_point_in_batch, build_rotation
from utils.workload_utils import NaiveWorkloadBalancer

import parallel_utils.basic_parallel_trainer.gaussian_utils as gs_utils
import parallel_utils.basic_parallel_trainer.result_aggregator as merger
import parallel_utils.basic_parallel_trainer.scene_utils as scene_utils
import parallel_utils.basic_parallel_trainer.task_utils as task_utils
import parallel_utils.schedulers.basic_scheduler as psb
import parallel_utils.grid_utils.utils as pgu

from scene.gaussian_nn_module import BoundedGaussianModel, BoundedGaussianModelGroup
from scene.cameras import Camera, EmptyCamera
from utils.datasets import CameraListDataset, DatasetRepeater, GroupedItems
from scene.scene4bounded_gaussian import SceneV3
from lpipsPyTorch import LPIPS

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

class Trainer4TreePartition:
    '''
        the initialization of Trainer4TreePartition involves nccl works. 
        DO NOT create Trainer4TreePartition in multiple-threads way
        + tensors for controling are usually cpu-tensors while most of others are gpu-tensors
        + format of some dict in methods:
            + shared_info_dict: dict[(src_model, dst_model): SharedGSInfo]
            + render_dict: dict[(taskid, src_model): RenderResult]
            + GS_Parameters_dict: dict[(src_model, dst_model): GSParameters]
    '''
    def __init__(self, nccl_group:dist.ProcessGroup, mdp:ModelParams, opt:OptimizationParams, pipe:PipelineParams, args:Namespace, ply_iteration:int) -> None:
        self.mdp = mdp
        self.opt = opt
        self.pipe = pipe
        self.args = args
        self.nccl_group = nccl_group
        self.TENSORBOARD_FOUND:bool = not args.DISABLE_TENSORBOARD

        self.SCENE_GRID_SIZE:np.ndarray = args.SCENE_GRID_SIZE
        self.SPLIT_ORDERS: List[int] = args.SPLIT_ORDERS
        self.ENABLE_REPARTITION:bool = args.ENABLE_REPARTITION
        self.REPARTITION_START_EPOCH:int = args.REPARTITION_START_EPOCH
        self.REPARTITION_END_EPOCH:int = args.REPARTITION_END_EPOCH
        self.REPARTITION_INTERVAL_EPOCH:int = args.REPARTITION_INTERVAL_EPOCH 
        self.SHRAE_GS_INFO:bool = args.SHRAE_GS_INFO

        self.Z_NEAR:float = args.Z_NEAR
        self.Z_FAR:float = args.Z_FAR
        self.EVAL_PSNR_INTERVAL:int = args.EVAL_PSNR_INTERVAL
        self.EVAL_INTERVAL_EPOCH:int = args.EVAL_INTERVAL_EPOCH
        self.SAVE_INTERVAL_EPOCH:int = args.SAVE_INTERVAL_EPOCH
        self.SAVE_INTERVAL_ITER:int = args.SAVE_INTERVAL_ITER
    
        self.MAX_SIZE_SINGLE_GS:int = args.MAX_SIZE_SINGLE_GS
        self.MAX_LOAD:int = args.MAX_LOAD
        self.MAX_BATCH_SIZE:int = args.MAX_BATCH_SIZE
        self.SKIP_PRUNE_AFTER_RESET:int = args.SKIP_PRUNE_AFTER_RESET
        self.SKIP_SPLIT:int = args.SKIP_SPLIT
        self.SKIP_CLONE:int = args.SKIP_CLONE
        self.DATALOADER_FIX_SEED:bool = args.DATALOADER_FIX_SEED
        self.bvh_depth:int = args.bvh_depth 
        self.sh_degree = self.mdp.sh_degree
        bg_color = [1, 1, 1] if self.mdp.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda") if not opt.random_background else None 

        self.RANK, self.WORLD_SIZE = dist.get_rank(), dist.get_world_size()

        self.prepare_output_and_logger()
        self.checkpoint_cleaner = scene_utils.CkptCleaner(max_len=args.CKPT_MAX_NUM)
        self.task_parser = task_utils.BasicTaskParser(self.WORLD_SIZE, self.RANK, self.logger)
        self.scheduler = psb.BasicScheduler(self.logger, self.tb_writer, batch_isend_irecv_version=0, group=self.nccl_group)
        self.scene:SceneV3 = SceneV3(self.mdp, None, shuffle=False)
        self.train_dataset:CameraListDataset = self.scene.getTrainCameras()
        self.test_dataset:CameraListDataset = self.scene.getTestCameras()

        # after set dataset, need it to set some opt
        self.init_grid_bvhTree_gsmodels(ply_iteration)
        # self.start_iteration = load_iteration
        # self.scene_3d_grid, self.path2bvh_nodes, self.sorted_leaf_nodes = scene_3d_grid, path2bvh_nodes, sorted_leaf_nodes
        # self.gaussians_group = gaussians_group
        # self.model_id2box, self.model_id2rank, self.local_model_ids = model_id2box, model_id2rank, local_model_ids
        # self.rkmd = task_utils.RankModelInfo(NUM_RANKS, NUM_MODELS, self.model_id2rank)
        self.train_rlt:torch.Tensor = scene_utils.get_camera_model_relation_dist(os.path.join(self.scene.model_path, 'trainset_relation.pt'), 
            self.train_dataset, len(self.model_id2box), self.path2bvh_nodes, self.sorted_leaf_nodes, self.Z_NEAR, self.Z_FAR, self.logger)
        self.test_rlt:torch.Tensor = scene_utils.get_camera_model_relation_dist(os.path.join(self.scene.model_path, 'testset_relation.pt'),
            self.test_dataset, len(self.model_id2box), self.path2bvh_nodes, self.sorted_leaf_nodes, self.Z_NEAR, self.Z_FAR, self.logger)
        if self.RANK == 0:
            torch.save(self.train_rlt, os.path.join(self.scene.model_path, 'trainset_relation.pt'))
            torch.save(self.test_rlt, os.path.join(self.scene.model_path, 'testset_relation.pt'))

    def prepare_output_and_logger(self):
        RANK = dist.get_rank()
        self.model_root = self.mdp.model_path
        self.mdp.model_path = os.path.join(self.mdp.model_path, 'rank_{}'.format(RANK))

        print("Output folder: {}".format(self.mdp.model_path))
        os.makedirs(self.mdp.model_path, exist_ok = True)
        logdir4rank = self.mdp.model_path
        if len(self.args.logdir) > 0:
            logdir4rank = os.path.join(self.args.logdir, 'rank_{}'.format(RANK))
            os.makedirs(logdir4rank, exist_ok=True)

        self.tb_writer = None
        if TENSORBOARD_FOUND and self.TENSORBOARD_FOUND:
            self.tb_writer = SummaryWriter(logdir4rank)
        else:
            print("Tensorboard not available: not logging progress")
        
        current_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())     
        logging.basicConfig(
            format='%(asctime)s-%(filename)s[line:%(lineno)d]-%(levelname)s: %(message)s',
            filemode='w',
            filename=os.path.join(logdir4rank, 'rank_{}_{}.txt'.format(RANK, current_time))
        )
        self.logger:logging.Logger = logging.getLogger('rank_{}'.format(RANK))
        self.logger.setLevel(logging.INFO)

    def init_grid_bvhTree_gsmodels(self, ply_iteration:int):
        # load or init scene_3d_grid, path2bvh_nodes, sorted_leaf_nodes
        load_iteration = scene_utils.find_ply_iteration(self.scene, self.logger) if ply_iteration<=0 else ply_iteration  
        if load_iteration > 0:
            # load exisiting log
            scene_3d_grid, path2bvh_nodes, sorted_leaf_nodes = scene_utils.load_BvhTree_on_3DGrid_dist(self.scene, load_iteration, self.SCENE_GRID_SIZE, self.logger)
        else: 
            scene_3d_grid, path2bvh_nodes, sorted_leaf_nodes = scene_utils.init_BvhTree_on_3DGrid_dist(
                self.scene, self.SCENE_GRID_SIZE, self.bvh_depth, load=None, position=self.scene.point_cloud.points, 
                SPLIT_ORDERS=self.SPLIT_ORDERS, pre_load_grid=None, conflict={}, logger=self.logger)
        
        model_id2box, model_id2rank, local_model_ids = scene_utils.get_model_assignment_dist(sorted_leaf_nodes, self.logger)
        local_model_ids.sort()

        gaussians_group = BoundedGaussianModelGroup(
            sh_degree_list=[self.mdp.sh_degree] * len(local_model_ids),
            range_low_list=[(model_id2box[m].range_low * scene_3d_grid.voxel_size + scene_3d_grid.range_low) for m in local_model_ids],
            range_up_list=[(model_id2box[m].range_up * scene_3d_grid.voxel_size + scene_3d_grid.range_low) for m in local_model_ids],
            device_list=["cuda"] * len(local_model_ids), 
            model_id_list=local_model_ids,
            padding_width=0.0,
            max_size=self.MAX_SIZE_SINGLE_GS,
        )
        self.logger.info(gaussians_group.get_info())

        if load_iteration <= 0:
            for mid in local_model_ids:
                _gau:BoundedGaussianModel = gaussians_group.get_model(mid)
                self.scene.loadPointCloud2Gaussians(
                    _gau,
                    range_low=_gau.range_low.cpu().detach().numpy(),
                    range_up=_gau.range_up.cpu().detach().numpy(),
                    padding_width=0.0
                )
            # init optimizer
            gaussians_group.training_setup(self.opt)   
        else:
            # load gs and optimizer
            gs_utils.load_gs_from_ply(self.opt, gaussians_group, local_model_ids, self.scene, load_iteration, self.logger)
        del self.scene.point_cloud
        self.logger.info('models are initialized:' + gaussians_group.get_info())

        # set up training args
        self.opt.iterations = len(self.train_dataset) *self.opt.epochs
        self.opt.scaling_lr_max_steps = len(self.train_dataset) *self.opt.epochs
        self.logger.info('set scaling_lr_max_steps {}'.format(self.opt.scaling_lr_max_steps))
        # self.gaussians_group.training_setup(self.opt) 
        gaussians_group.update_learning_rate(max(0, load_iteration))

        NUM_RANKS, NUM_MODELS = dist.get_world_size(), len(model_id2box)
        self.start_iteration = load_iteration
        self.scene_3d_grid, self.path2bvh_nodes, self.sorted_leaf_nodes = scene_3d_grid, path2bvh_nodes, sorted_leaf_nodes
        
        self.gaussians_group = gaussians_group
        self.model_id2box, self.model_id2rank, self.local_model_ids = model_id2box, model_id2rank, local_model_ids
        self.rkmd = task_utils.RankModelInfo(NUM_RANKS, NUM_MODELS, self.model_id2rank)

    def resplit_grid_bvhTree_gsmodels(self, iteration:int):
        with torch.no_grad():
            # evaluation of workload distribution on 3D space
            self.scene_3d_grid.clean_load()
            step = None
            for _gaussians_name in self.gaussians_group.all_gaussians:
                _gaussians:BoundedGaussianModel = self.gaussians_group.get_model(_gaussians_name)
                position = _gaussians.get_xyz.cpu().detach().numpy()
                P = position.shape[0]
                load = np.ones((P,), dtype=float)
                self.scene_3d_grid.accum_load(load_np=load, position=position)
                if step is None:
                    _group = _gaussians.optimizer.param_groups[0]
                    _torchParameter = _group["params"][0]
                    step = int(_gaussians.optimizer.state[_torchParameter]['step'])
                    self.logger.info('read step {} of {}'.format(step, _torchParameter))

            self.logger.info('rank {} has got local load, and is going to all_reduce it'.format(self.RANK))
            dist.barrier()
            load_tensor = torch.tensor(self.scene_3d_grid.load_cnt, dtype=torch.float, device='cuda', requires_grad=False)
            dist.all_reduce(load_tensor, op=dist.ReduceOp.SUM, group=None, async_op=False)
            self.logger.info(f'reduced load, {load_tensor.max()}, {load_tensor.min()}, {load_tensor.mean()}')
            reduced_load_np = load_tensor.clamp(min=0).cpu().numpy().astype(float)
        
            # force a new tree
            scene_3d_grid, path2bvh_nodes, sorted_leaf_nodes = scene_utils.resplit_BvhTree_on_3DGrid_dist(
                self.scene_3d_grid, self.bvh_depth, load=None, position=None, 
                SPLIT_ORDERS=self.SPLIT_ORDERS, pre_load_grid=reduced_load_np, conflict=self.path2bvh_nodes, logger=self.logger)
            
            model_id2box, model_id2rank, local_model_ids = scene_utils.get_model_assignment_dist(sorted_leaf_nodes, self.logger)
            local_model_ids.sort()
            NUM_RANKS, NUM_MODELS = dist.get_world_size(), len(model_id2box)
            rkmd = task_utils.RankModelInfo(NUM_RANKS, NUM_MODELS, model_id2rank)

            gaussians_group = BoundedGaussianModelGroup(
                sh_degree_list=[self.mdp.sh_degree] * len(local_model_ids),
                range_low_list=[(model_id2box[m].range_low * scene_3d_grid.voxel_size + scene_3d_grid.range_low) for m in local_model_ids],
                range_up_list=[(model_id2box[m].range_up * scene_3d_grid.voxel_size + scene_3d_grid.range_low) for m in local_model_ids],
                device_list=["cuda"] * len(local_model_ids), 
                model_id_list=local_model_ids,
                padding_width=0.0,
                max_size=self.MAX_SIZE_SINGLE_GS,
            )
            self.logger.info('new module groups' + gaussians_group.get_info())
            # gather GSParameters
            dst_model2GSParameters = self.forward_only_GSParameters_dist(
                dst_model2Box=model_id2box, dst_rkmd=rkmd, dst_local_models=local_model_ids
            )
            # update gaussians_group
            for mid in local_model_ids:
                gau:BoundedGaussianModel = gaussians_group.get_model(mid)
                pkg = dst_model2GSParameters.pop(mid)
                gau.un_pack_up(
                    pkg.pkg, spatial_lr_scale=self.scene.cameras_extent, iteration=iteration, 
                    step=step if step is not None else 1, opt=self.opt)
            torch.cuda.empty_cache()
            # update state
            self.scene_3d_grid, self.path2bvh_nodes, self.sorted_leaf_nodes = scene_3d_grid, path2bvh_nodes, sorted_leaf_nodes
            self.gaussians_group = gaussians_group
            self.model_id2box, self.model_id2rank, self.local_model_ids = model_id2box, model_id2rank, local_model_ids
            self.rkmd = task_utils.RankModelInfo(NUM_RANKS, NUM_MODELS, self.model_id2rank)
            # update camera_models relation
            self.train_rlt:torch.Tensor = scene_utils.get_camera_model_relation_dist('', 
                self.train_dataset, len(self.model_id2box), self.path2bvh_nodes, self.sorted_leaf_nodes, self.Z_NEAR, self.Z_FAR, self.logger)
            self.test_rlt:torch.Tensor = scene_utils.get_camera_model_relation_dist('',
                self.test_dataset, len(self.model_id2box), self.path2bvh_nodes, self.sorted_leaf_nodes, self.Z_NEAR, self.Z_FAR, self.logger)
            if self.RANK == 0:
                torch.save(self.train_rlt, os.path.join(self.scene.model_path, 'trainset_relation.pt'))
                torch.save(self.test_rlt, os.path.join(self.scene.model_path, 'testset_relation.pt'))

    def forward_only_GSParameters_dist(self, dst_model2Box:Dict[int, pgu.BoxinGrid3D], dst_rkmd:task_utils.RankModelInfo, dst_local_models:List[int]): 
        local_content_pool, send_amount_cpu = merger.gather_GSParametsers_pool_and_release_srcGS(
            self.gaussians_group, self.model_id2box, dst_model2Box, self.scene_3d_grid, self.logger
        )
        global_send_amount = self.scheduler.broadcast_global_model2model_info(src_rkmd=self.rkmd, SEND_AMOUNT_CPU=send_amount_cpu)
        send_tasks, recv_tasks = self.task_parser.parse_model2model_task(src_rkmd=self.rkmd, dst_rkmd=dst_rkmd, GLOBAL_SEND_AMOUNT=global_send_amount)
        extra_space_pool = {
            recv.key(): task_utils.GSParametsers.space_for_content(msg=(recv, self.sh_degree))
                for recv in recv_tasks
        }
        send2content = {send.key():local_content_pool[(send.src_model, send.dst_model)] for send in send_tasks}
        recv2space = extra_space_pool
        _extra_content_pool = self.scheduler.model2model_exchange_forward(
            content_pool=send2content, space_pool=recv2space,
            sorted_send_tasks=send_tasks, sorted_recv_tasks=recv_tasks,
            func_zip=task_utils.GSParametsers.zip_content, func_unzip=task_utils.GSParametsers.unzip_content
        )
        extra_content_pool = {}
        for task_key, content in _extra_content_pool.items():
            src_dst = (task_key[3], task_key[4])
            extra_content_pool[src_dst] = content

        model2GSParametsers= merger.merge_GSParametsers(
            local_content_pool, extra_content_pool, dst_local_models, 
            CHANNEL=task_utils.GSParametsers.get_channel_size(self.sh_degree), 
            logger=self.logger
        )    
        return model2GSParametsers

    def mini_batch_dist_sync(self, iteration, data:List[Union[Camera, EmptyCamera]]):
        batch_size = len(data)
        cameraUid_taskId_mainRank = torch.zeros((batch_size, 3), dtype=torch.int, device='cpu') 
        packages = torch.zeros((batch_size, *Camera.package_shape()), dtype=torch.float32, device='cpu') 
        if self.RANK == 0:  
            for i, camera in enumerate(data):
                cameraUid_taskId_mainRank[i, 0] = camera.uid
                cameraUid_taskId_mainRank[i, 1] = iteration + i
                cameraUid_taskId_mainRank[i, 2] = i % self.WORLD_SIZE
                # cameraUid_taskId_mainRank[i, 2] = assign_task2rank_dist(train_rlt[camera.uid, :], i)
                packages[i] = camera.pack_up(device='cpu')
        data_gpu = [_cmr.to_device('cuda') for _cmr in data]        
        cameraUid_taskId_mainRank_gpu = cameraUid_taskId_mainRank.cuda()
        dist.broadcast(cameraUid_taskId_mainRank_gpu, src=0, async_op=False, group=None)
        dist.barrier()  
        cameraUid_taskId_mainRank = cameraUid_taskId_mainRank_gpu.cpu()
        self.logger.info('broadcast cameraUid_taskId_mainRank {}, iteration {}'.format(cameraUid_taskId_mainRank, iteration))
         
        task_id2cameraUid: Dict[int, int] = {}
        uid2camera: Dict[int, Union[Camera, EmptyCamera]] = {}
        task_id2camera: Dict[int, Union[Camera, EmptyCamera]] = {}
        for camera in data_gpu:
            uid2camera[camera.uid] = camera 
        self.logger.info(uid2camera)    
        for row in cameraUid_taskId_mainRank:
            uid, tid, mid = int(row[0]), int(row[1]), int(row[2])
            task_id2cameraUid[tid] = uid
            task_id2camera[tid] = uid2camera[uid]

        mini_message = ' '.join(['(id={}, H={}, W={})'.format(e.id, e.image_height, e.image_width) for e in data_gpu])
        _uids = cameraUid_taskId_mainRank[:,0].to(torch.long)
        self.logger.info("rank {} get task {}, relation \n{}".format(self.RANK, mini_message, self.train_rlt[_uids, :]))  
            
        return data_gpu, cameraUid_taskId_mainRank, task_id2cameraUid, uid2camera, task_id2camera
    
    def forward_SharedGSInfo_dist(self):
        local_content_pool, send_amount_cpu = merger.gather_SharedGSInfo_pool(
            self.gaussians_group, self.model_id2box, self.scene_3d_grid, self.logger
        )
        if self.SHRAE_GS_INFO:
            global_send_amount = self.scheduler.broadcast_global_model2model_info(src_rkmd=self.rkmd, SEND_AMOUNT_CPU=send_amount_cpu)
            send_tasks, recv_tasks = self.task_parser.parse_model2model_task(src_rkmd=self.rkmd, dst_rkmd=self.rkmd, GLOBAL_SEND_AMOUNT=global_send_amount)
            extra_space_pool = {
                recv.key(): task_utils.SharedGSInfo.space_for_content(msg=(recv, self.sh_degree))
                    for recv in recv_tasks
            }
            # how stupid I am to set these java-like template-functions/abc-classes
            # I still have to add details for different kinds of data
            send2content = {send.key():local_content_pool[(send.src_model, send.dst_model)] for send in send_tasks}
            recv2space = extra_space_pool
            _extra_content_pool = self.scheduler.model2model_exchange_forward(
                content_pool=send2content, space_pool=recv2space,
                sorted_send_tasks=send_tasks, sorted_recv_tasks=recv_tasks,
                func_zip=task_utils.SharedGSInfo.zip_content, func_unzip=task_utils.SharedGSInfo.unzip_content
            )
            extra_content_pool = {}
            for task_key, content in _extra_content_pool.items():
                src_dst = (task_key[3], task_key[4])
                extra_content_pool[src_dst] = content

            model2GSInfo = merger.merge_SharedGSInfo(
                local_content_pool, extra_content_pool, self.local_model_ids, self.logger
            )    
        else:
            send_tasks:List[task_utils.SendSharedGSInfo] = []
            recv_tasks:List[task_utils.RecvSharedGSInfo] = []
            extra_content_pool:Dict[Tuple[int, int], task_utils.SharedGSInfo] = {}
            model2GSInfo = merger.gather_local_SharedGSInfo(self.gaussians_group, self.local_model_ids)

        return send_tasks, recv_tasks, local_content_pool, extra_content_pool, model2GSInfo

    def forward_RenderResult_dist(self, model2GSinfo:Dict[int, task_utils.SharedGSInfo], cameraUid_taskId_mainRank, relation, data_gpu):
        send_task:List[task_utils.SendRenderResult]; recv_tasks:List[task_utils.RecvRenderResult]; render_tasks:List[task_utils.RenderTask]; main_rank_tasks:List[task_utils.MainRankTask]
        send_task, recv_tasks, render_tasks, main_rank_tasks = self.task_parser.parse_render_relation(
            rkmd=self.rkmd, _task_main_rank=cameraUid_taskId_mainRank[:, [1,2]], _relation_matrix=relation, tasks_message=data_gpu
        )
        local_content_pool = merger.gather_RenderResult_pool(
            gaussians=self.gaussians_group, render_info_pool=model2GSinfo, render_tasks=render_tasks, pipe=self.pipe, 
            background=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"), logger=self.logger
        )   # for render_result of a optical field segment, background must be all-zeros
        extra_space_pool = {
            recv.key(): task_utils.RenderResult.space_for_content(recv)
                for recv in recv_tasks
        }

        send2content = {send.key():local_content_pool[(send.task_id, send.src_model)] for send in send_task}
        recv2space = extra_space_pool
        _extra_content_pool = self.scheduler.model2model_exchange_forward(
            content_pool=send2content, space_pool=recv2space,
            sorted_send_tasks=send_task, sorted_recv_tasks=recv_tasks,
            func_zip=task_utils.RenderResult.zip_content, func_unzip=task_utils.RenderResult.unzip_content
        )
        extra_content_pool = {}
        for task_key, content in _extra_content_pool.items():
            task_srcModel = (task_key[0], task_key[3])
            extra_content_pool[task_srcModel] = content
        return send_task, recv_tasks, render_tasks, main_rank_tasks, local_content_pool, extra_content_pool
    
    def forward_blender_local(
            self, 
            local_render:Dict[Tuple, task_utils.RenderResult], 
            extra_render:Dict[Tuple, task_utils.RenderResult], 
            main_rank_tasks:List[task_utils.MainRankTask], 
            cameraUid_taskId_mainRank:torch.Tensor,
            relation_matrix:torch.Tensor):
        # use copy of local render_ret to bledner, so that first auto_grad can be faster
        local_copy = {key: v.make_copy() for key,v in local_render.items()}
        self.logger.debug("copy {} local render_rets for main rank task".format(len(local_copy)))
        taskid2relation = {int(cameraUid_taskId_mainRank[i, 1]):relation_matrix[i] for i in range(len(cameraUid_taskId_mainRank))}
        image:Dict[Tuple(int, int), task_utils.RenderResult] = {}
        for main_task in main_rank_tasks:
            _relation_vector = taskid2relation[main_task.task_id]
            image[(main_task.task_id, main_task.rank)] = merger.blender_render_result(
                local_copy, extra_render, main_task, _relation_vector, self.background, self.logger
            )
        return image, local_copy    

    def loss_local(self, blender_result:Dict[Tuple[int,int], task_utils.RenderResult], main_tasks:List[task_utils.MainRankTask], task_id2camera:Dict[int, Camera]):
        if len(blender_result) <= 0:
            return None, None
        
        loss_main_rank, Ll1_main_rank = torch.tensor(0.0, device='cuda'), torch.tensor(0.0, device='cuda') 
        for task_rank in blender_result:
            render = blender_result[task_rank]
            ori_camera:Camera = task_id2camera[task_rank[0]]

            image = render.render
            gt_image = ori_camera.original_image.to('cuda')

            self.logger.debug('gt info {}'.format(gt_image.mean()))

            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            self.logger.debug('taskid_mainRank {}, loss {}, l1 {}'.format(task_rank, loss, Ll1))
            loss_main_rank += loss
            Ll1_main_rank += Ll1
        return loss_main_rank, Ll1_main_rank

    def check_gaussian_model_grad(self):
        for mid in self.gaussians_group.model_id_list:
            _gau:BoundedGaussianModel = self.gaussians_group.get_model(mid)
            self.logger.debug('none grad of xyz {}'.format(_gau._xyz.grad is None))
            try:
                self.logger.debug('mean grad of xyz {} {} {}'.format(_gau._xyz.grad.mean(), _gau._xyz.grad.max(), _gau._xyz.grad.min()))
            except:
                pass    

    def backward_RenderResult_dist(
            self, 
            local_content:Dict[Tuple, task_utils.RenderResult], 
            local_copy:Dict[Tuple, task_utils.RenderResult], 
            extra_content:Dict[Tuple, task_utils.RenderResult], 
            send_task:List[task_utils.SendRenderResult], 
            recv_task:List[task_utils.RecvRenderResult]):
        
        # self.check_gaussian_model_grad()
    
        gard_of_extra_content = {
            recv.key(): extra_content[(recv.task_id, recv.src_model)]
                for recv in recv_task
        }
        space_for_extra_grad_of_local_content = {
            send.key(): task_utils.RenderResult.space_for_grad(send) 
                for send in send_task
        }
        _extra_grad_pool = self.scheduler.model2model_exchange_backward(
            content_grad_pool=gard_of_extra_content, 
            space_grad_pool=space_for_extra_grad_of_local_content,
            sorted_send_tasks=send_task,
            sorted_recv_tasks=recv_task,
            func_zip=task_utils.RenderResult.zip_grad,
            func_unzip=task_utils.RenderResult.unzip_grad
        )
        extra_grad_pool = {}
        for task_key, content in _extra_grad_pool.items():
            task_srcModel = (task_key[0], task_key[3])
            extra_grad_pool[task_srcModel] = content

        # gather tensor/grad pair
        send_tensors, recv_grads = [], []
        for send in send_task:
            task_model = (send.task_id, send.src_model)
            _t, _g = task_utils.RenderResult.gather_paired_tensor_grad_list(local_content[task_model], extra_grad_pool[task_model])
            send_tensors += _t;  recv_grads += _g
        self.logger.debug('find {} pairs  (send_tensors, recv_grads)'.format(len(send_tensors)))

        copy_tensors, copy_grads = [], []
        for task_model in local_content:
            _t, _g = task_utils.RenderResult.gather_paired_tensor_grad_list_from_copy(local_content[task_model], local_copy[task_model])
            copy_tensors += _t; copy_grads += _g
        self.logger.debug('find {} pairs  (copy_tensors, copy_grads)'.format(len(copy_tensors)))

        tensors, grads = send_tensors + copy_tensors, recv_grads + copy_grads
        if len(tensors) > 0:
            self.logger.debug('backward for {} pairs in backward_RenderResult_dist'.format(len(tensors)))
            torch.autograd.backward(tensors, grads, retain_graph=True) 
        else:
            self.logger.warning('find no main grad nor extra grad in backward_RenderResult_dist') 

        # check model grad
        # self.check_gaussian_model_grad()

    def __backward_SharedGSInfo_dist(
            self,
            local_content:Dict[Tuple, task_utils.SharedGSInfo], 
            extra_content:Dict[Tuple, task_utils.SharedGSInfo], 
            send_task:List[task_utils.SendSharedGSInfo], 
            recv_task:List[task_utils.RecvSharedGSInfo]):
        gard_of_extra_content = {
            recv.key(): extra_content[(recv.src_model, recv.dst_model)]
                for recv in recv_task
        }
        space_for_extra_grad_of_local_content = {
            send.key(): task_utils.SharedGSInfo.space_for_grad(msg=(send, self.sh_degree)) 
                for send in send_task
        }
        _extra_grad_pool = self.scheduler.model2model_exchange_backward(
            content_grad_pool=gard_of_extra_content, 
            space_grad_pool=space_for_extra_grad_of_local_content,
            sorted_send_tasks=send_task,
            sorted_recv_tasks=recv_task,
            func_zip=task_utils.SharedGSInfo.zip_grad,
            func_unzip=task_utils.SharedGSInfo.unzip_grad
        )
        extra_grad_pool = {}
        for task_key, content in _extra_grad_pool.items():
            srcModel_dstModel = (task_key[3], task_key[4])
            extra_grad_pool[srcModel_dstModel] = content

        # gather tensor/grad pair
        tensors, grads = [], []
        for send in send_task:
            srcModel_dstModel = (send.src_model, send.dst_model)
            _t, _g = task_utils.SharedGSInfo.gather_paired_tensor_grad_list(local_content[srcModel_dstModel], extra_grad_pool[srcModel_dstModel])
            tensors += _t;  grads += _g
   
        if len(tensors) > 0:
            self.logger.debug('find {} pairs in __backward_SharedGSInfo_dist'.format(len(tensors)))
            torch.autograd.backward(tensors, grads, retain_graph=True) 
        else:
            self.logger.warning('find no main grad nor extra grad in __backward_SharedGSInfo_dist') 

    def backward_SharedGSInfo_dist(
            self,
            local_content:Dict[Tuple, task_utils.SharedGSInfo], 
            extra_content:Dict[Tuple, task_utils.SharedGSInfo], 
            send_task:List[task_utils.SendSharedGSInfo], 
            recv_task:List[task_utils.RecvSharedGSInfo]):
        if self.SHRAE_GS_INFO:
            self.__backward_SharedGSInfo_dist(local_content, extra_content, send_task, recv_task)
        else:
            # nothing to do as SharedGSInfos are all local tensor
            self.logger.debug("backward_SharedGSInfo_dist: nothing to do as SharedGSInfos are all local tensor")

    def training_report(self, iteration, Ll1_main_rank, loss_main_rank, elapsed):
        if self.tb_writer:
            if Ll1_main_rank is not None:
                self.tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1_main_rank.item(), iteration)
            if loss_main_rank is not None:
                self.tb_writer.add_scalar('train_loss_patches/total_loss', loss_main_rank.item(), iteration)
            self.tb_writer.add_scalar('iter_time', elapsed, iteration)
            self.tb_writer.add_scalar('cnt/memory', torch.cuda.memory_allocated()/(1024**3), iteration)

    def train(self):
        RANK, WORLD_SIZE = dist.get_rank(), dist.get_world_size()
        start_epoch = int(round(self.start_iteration/len(self.train_dataset)))
        self.logger.info('start from epoch {}'.format(start_epoch))
        NUM_EPOCH = self.opt.epochs - start_epoch
       
        iter_start, iter_end = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
        progress_bar = tqdm(range(self.start_iteration, NUM_EPOCH*len(self.train_dataset)), desc="Training progress") if RANK == 0 else None 
        iteration, ema_loss_for_log = self.start_iteration, None
        for _i_epoch in range(NUM_EPOCH):
            i_epoch = _i_epoch + start_epoch
            seed = 0 if self.DATALOADER_FIX_SEED else i_epoch
            # predict balanced mini-batches, update lr at the start of an epoch
            _indices:list = scene_utils.get_sampler_indices_dist(self.train_dataset, seed=seed)
            _grouped_data_idx = scene_utils.get_grouped_indices_dist(self.model_id2rank, self.train_rlt, _indices, self.MAX_LOAD, self.MAX_BATCH_SIZE)   
            _grouped_train_dataset = GroupedItems(self.train_dataset, _grouped_data_idx) 
            train_loader = DataLoader(_grouped_train_dataset, batch_size=1, num_workers=8, prefetch_factor=2, drop_last=True,
                shuffle=False, collate_fn=SceneV3.get_batch, pin_memory=True, pin_memory_device='cuda') # DO NOT shuffle
            self.logger.info("build groups of data items")
            self.gaussians_group.update_learning_rate(iteration) 
            self.gaussians_group.set_SHdegree(iteration//1000)
            # self.pipe.debug = True # never debug in render kernel
            for ids_data in train_loader:
                # sync data
                # actually this step is not necessary for this trainer, as it holds synchronized sample-indces whichever rank it locates
                ids, data = ids_data[0] # [(tuple(int), tuple(camera))]
                batch_size = len(data)  # list of Camera/None, batchsize can be dynamic in the future    
                assert batch_size > 0, "get empty group"
                self.logger.info('start of iteration {}'.format(iteration))
                iter_start.record()
                data_gpu, cameraUid_taskId_mainRank, task_id2cameraUid, uid2camera, task_id2camera = self.mini_batch_dist_sync(iteration, data)

                # forward: step 1, gather gs_info and exchange them if needed 
                SEND_GSINFO_TASKS, RECV_GSINFO_TASKS, local_GSinfo, extra_GSinfo, model2GSInfo = self.forward_SharedGSInfo_dist()
                
                # forward: step 2, render some image and exchange them
                _uids = cameraUid_taskId_mainRank[:,0].to(torch.long)
                _relation = self.train_rlt[_uids, :].clone().detach().contiguous()
                SEND_RENDER, RECV_RENDER, RENDER_TASK, MAIN_RANK_TASK, local_render, extra_render = self.forward_RenderResult_dist(model2GSInfo, cameraUid_taskId_mainRank, _relation, data_gpu)

                # forward: step 3, blender images, this step involves no nn.Module
                blender_result, local_render_copy = self.forward_blender_local(
                    local_render, extra_render, MAIN_RANK_TASK, cameraUid_taskId_mainRank, _relation)
                # loss
                loss_main_rank, Ll1_main_rank = self.loss_local(blender_result, MAIN_RANK_TASK, task_id2camera)
                # backward: step -3, from loss to (dL_dextraRender, dL_dlocalCopy)
                if loss_main_rank is not None:
                    loss_main_rank.backward(retain_graph=True)  
                    self.logger.debug('{} main_rank image rets'.format(len(blender_result))) 
                else:
                    self.logger.debug('no main_rank image ret, also no need to backward') 

                # backward, step -2, recv extra_gard, from dL_drender to (dL_dextraGSinfo, dL_dlocalGSpara)
                self.backward_RenderResult_dist(local_render, local_render_copy, extra_render, SEND_RENDER, RECV_RENDER)
                # backward, step -1, recv extra_grad, from dL_GSinfo to GSPara
                self.backward_SharedGSInfo_dist(local_GSinfo, extra_GSinfo, SEND_GSINFO_TASKS, RECV_GSINFO_TASKS)

                iter_end.record()
                torch.cuda.synchronize()
                iter_time = iter_start.elapsed_time(iter_end)
                with torch.no_grad():
                # torch.cuda.empty_cache()
                # Progress bar
                    if dist.get_rank() == 0:
                        if ema_loss_for_log is None:
                            ema_loss_for_log = loss_main_rank.item() 
                        ema_loss_for_log = 0.4 * loss_main_rank.item() + 0.6 * ema_loss_for_log
                        progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                        progress_bar.update(batch_size)
                        if iteration >= self.opt.iterations:
                            progress_bar.close()
                    # Log 
                    self.training_report(iteration, Ll1_main_rank, loss_main_rank, iter_time)
                    gs_utils.update_densification_stat(iteration, self.opt, self.gaussians_group, local_render, self.tb_writer, self.logger)
                    gs_utils.densification(iteration, batch_size, False, self.SKIP_CLONE, self.SKIP_SPLIT,
                        self.opt, self.scene, self.gaussians_group, self.tb_writer, self.logger)
                    gs_utils.reset_opacity(iteration, batch_size, self.opt, self.mdp, self.gaussians_group, self.tb_writer, self.logger)
                    # Optimizer
                    if True: # iteration < self.opt.iterations:
                        for _gau_name in self.gaussians_group.all_gaussians:
                            _gaussians:BoundedGaussianModel = self.gaussians_group.get_model(_gau_name)
                            _gaussians.optimizer.step()
                            _gaussians.optimizer.zero_grad(set_to_none = True)
                            # self.logger.warning('skip optimizer step')

                iteration += batch_size
    
            # end of epoch
            with torch.no_grad():
                if (i_epoch % self.SAVE_INTERVAL_EPOCH == 0) or (i_epoch == (NUM_EPOCH-1)):
                    scene_utils.save_BvhTree_on_3DGrid(self.checkpoint_cleaner, iteration, self.scene.model_path, self.gaussians_group, self.path2bvh_nodes)
                if self.ENABLE_REPARTITION and (i_epoch%self.REPARTITION_INTERVAL_EPOCH==0) and (self.REPARTITION_START_EPOCH<=i_epoch <=self.REPARTITION_END_EPOCH):
                    # burning: I don't think resplit scene is cool, it can easily cause OOM
                    # this naive implementation works if memeory is always enough
                    self.resplit_grid_bvhTree_gsmodels(iteration=iteration)
                if (i_epoch % self.EVAL_INTERVAL_EPOCH == 0) or (i_epoch == (NUM_EPOCH-1)):  
                    torch.cuda.empty_cache()  
                    self.logger.info('eval after epoch {}'.format(i_epoch))
                    self.eval(train_iteration=iteration)
                    torch.cuda.empty_cache()     

        print('complete')
        self.logger.info('complete')

    def eval(self, train_iteration:int):
        with torch.no_grad():
            RANK, WORLD_SIZE = dist.get_rank(), dist.get_world_size()
            # gather SharedGSInfo only once
            _, _, _, _, model2GSInfo = self.forward_SharedGSInfo_dist()
            # dataloader
            eval_train_loader = DataLoader(
                DatasetRepeater(self.train_dataset, len(self.train_dataset)//self.EVAL_PSNR_INTERVAL, False, self.EVAL_PSNR_INTERVAL),
                batch_size=self.MAX_BATCH_SIZE, num_workers=8, prefetch_factor=2, drop_last=False,
                shuffle=False, collate_fn=SceneV3.get_batch, pin_memory=True, pin_memory_device='cuda'
            )
            eval_test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.MAX_BATCH_SIZE, num_workers=8, prefetch_factor=2, drop_last=False,
                shuffle=False, collate_fn=SceneV3.get_batch, pin_memory=True, pin_memory_device='cuda'
            )
            name:str; dataloader:DataLoader; rlt:torch.Tensor
            for name, dataloader, rlt in zip(('train', 'test'), (eval_train_loader, eval_test_loader), (self.train_rlt, self.test_rlt)):
                iteration, num_samples, psnr_total, l1_total = 0, 0.0, 0.0, 0.0
                for idx, data in enumerate(dataloader):
                    # sync data
                    data_gpu, cameraUid_taskId_mainRank, task_id2cameraUid, uid2camera, task_id2camera = self.mini_batch_dist_sync(iteration, data)
                    # forward: render some image and exchange them
                    _uids = cameraUid_taskId_mainRank[:,0].to(torch.long)
                    _relation = rlt[_uids, :].clone().detach().contiguous()
                    _, _, RENDER_TASK, MAIN_RANK_TASK, local_render, extra_render = self.forward_RenderResult_dist(model2GSInfo, cameraUid_taskId_mainRank, _relation, data_gpu)
                    # forward:  blender images, this step involves no nn.Module
                    blender_result, local_render_copy = self.forward_blender_local(
                        local_render, extra_render, MAIN_RANK_TASK, cameraUid_taskId_mainRank, _relation)
                    # loss
                    for task_rank in blender_result:
                        render, ori_camera = blender_result[task_rank], task_id2camera[task_rank[0]]
                        image, gt_image = render.render, ori_camera.original_image.to('cuda')
                        Ll1, PSNR = l1_loss(image, gt_image).mean().item(), psnr(image, gt_image).mean().item()
                        self.logger.info('{}, image {}, psnr {}, l1 {}'.format(name, ori_camera.image_name, PSNR, Ll1))
                        if self.tb_writer and (ori_camera.uid < 1000) and (ori_camera.uid % 5 == 0):
                            self.tb_writer.add_images(name + "_view_{}/render".format(ori_camera.image_name), image[None], global_step=train_iteration)
                            self.tb_writer.add_images(name + "_view_{}/ground_truth".format(ori_camera.image_name), gt_image[None], global_step=train_iteration)
                        num_samples += 1; psnr_total += PSNR; l1_total += Ll1
                        if self.args.SAVE_EVAL_IMAGE:
                            torchvision.utils.save_image(image, os.path.join(self.scene.save_img_path, '{}_{}_{}'.format(ori_camera.image_name, iteration, name) + ".png"))
                            torchvision.utils.save_image(gt_image, os.path.join(self.scene.save_gt_path, '{}'.format(ori_camera.image_name) + ".png"))  
                        
                    iteration += len(data)
                    # save image
                    if self.args.SAVE_EVAL_SUB_IMAGE:
                        for k in local_render:
                            task_id, model_id = k
                            ori_camera:Camera = task_id2camera[task_id]
                            idx = ori_camera.uid
                            torchvision.utils.save_image(
                                local_render[k].render, 
                                os.path.join(self.scene.save_img_path, '{}_sub_{}'.format(ori_camera.image_name, model_id) + ".png")
                                )
                        for k in extra_render:
                            task_id, model_id = k
                            ori_camera:Camera = task_id2camera[task_id]
                            idx = ori_camera.uid
                            torchvision.utils.save_image(
                                extra_render[k].render, 
                                os.path.join(self.scene.save_img_path, '{}_sub_{}'.format(ori_camera.image_name, model_id) + ".png")
                            )
                statistical = torch.tensor([num_samples, psnr_total, l1_total], dtype=torch.float32, device='cuda', requires_grad=False)
                dist.all_reduce(statistical, op=dist.ReduceOp.SUM, group=None, async_op=False)
                torch.cuda.synchronize()
                _samples, psnr_test, l1_test = statistical.cpu().numpy()

                self.logger.info("\n[ITER {}] Evaluating {}: samples {} L1 {} PSNR {}".format(train_iteration, name, _samples, l1_test/_samples, psnr_test/_samples))      
                print("\n[ITER {}] Evaluating {}: samples{} L1 {} PSNR {}".format(train_iteration, name, _samples, l1_test/_samples, psnr_test/_samples))
                if self.tb_writer and RANK==0:
                    self.tb_writer.add_scalar(name + '/loss_viewpoint - l1_loss', l1_test/_samples, train_iteration)
                    self.tb_writer.add_scalar(name + '/loss_viewpoint - psnr', psnr_test/_samples, train_iteration)
                    
        if self.tb_writer:
            for name in self.gaussians_group.all_gaussians:
                model_id = name
                _gaussians:BoundedGaussianModel = self.gaussians_group.get_model(name)
                self.tb_writer.add_scalar('total_points_{}'.format(model_id), _gaussians.get_xyz.shape[0], train_iteration)

