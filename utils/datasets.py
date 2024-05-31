import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks, readColmapSceneAndEmptyCameraInfo, readColmapOnlyEmptyCameraInfo, readNerfSyntheticAndEmptyCameraInfo, storePly, fetchPly
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import BasicPointCloud
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON, loadCam, loadEmptyCam
from PIL import Image
from scene.dataset_readers import CameraInfo, SceneInfo
from scene.cameras import Camera, EmptyCamera, ViewMessage, Patch
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2


def getCameraListDataset(camera_infos:list, resolution_scale:float, args:ModelParams):
    """
    camera_infos: list of CameraInfo
    """
    return CameraListDataset(cameras_infos=camera_infos, resolution_scale=resolution_scale, args=args)    


class CameraListDataset(Dataset):
    '''
    torch.utils.data.Dataset for DDP and the convenience of debug
    '''
    def __init__(self, cameras_infos, resolution_scale, args:ModelParams) -> None:
        super().__init__()
        self.cameras_infos = cameras_infos
        self.resolution_scale = resolution_scale
        self.args = args
        self.args.data_device = 'cpu'   # make sure the minimal batch is loaded on cpu
       
    def __len__(self):
        return len(self.cameras_infos)

    def __getitem__(self, idx):     
        info: CameraInfo = self.cameras_infos[idx]
        image = Image.open(info.image_path)
        full_info = CameraInfo(uid=info.uid, R=info.R, T=info.T, FovY=info.FovY, FovX=info.FovX, image=image,
                              image_path=info.image_path, image_name=info.image_name, width=info.width, height=info.height)

        camera:Camera = loadCam(self.args, id=idx, cam_info=full_info, resolution_scale=self.resolution_scale)   
        # return camera.to_device('cuda')
        return camera

    def get_empty_item(self, idx):
        info: CameraInfo = self.cameras_infos[idx]
        image = None
        part_of_info = CameraInfo(uid=info.uid, R=info.R, T=info.T, FovY=info.FovY, FovX=info.FovX, image=None,
                              image_path=info.image_path, image_name=info.image_name, width=info.width, height=info.height)
        return loadEmptyCam(self.args, id=idx, cam_info=part_of_info, resolution_scale=self.resolution_scale)


class EmptyCameraListDataset(Dataset):
    def __init__(self, empty_cameras:list) -> None:
        super().__init__()
        self.empty_cameras:list = empty_cameras

    def __len__(self):
        return len(self.empty_cameras)

    def __getitem__(self, idx):     
        _c:EmptyCamera = self.empty_cameras[idx]
        return EmptyCamera(
            colmap_id=0, 
            R=_c.R+0, T = _c.T+0, 
            FoVx=_c.FoVx+0, FoVy=_c.FoVy+0, 
            width_height=(_c.image_width, _c.image_height),
            gt_alpha_mask=None,
            image_name='',
            uid=idx,
            data_device='cpu'
        )

    def get_empty_item(self, idx):
        _c:EmptyCamera = self.empty_cameras[idx]
        return EmptyCamera(
            colmap_id=0, 
            R=_c.R+0, T = _c.T+0, 
            FoVx=_c.FoVx+0, FoVy=_c.FoVy+0, 
            width_height=(_c.image_width, _c.image_height),
            gt_alpha_mask=None,
            image_name='',
            uid=idx,
            data_device='cpu'
        )
    

class PatchListDataset(Dataset):
    def __init__(self, cameras_list:CameraListDataset, h_division:int=2, w_division:int=2) -> None:
        super().__init__()
        assert h_division >= 1 and w_division >= 1
        self.cameras_list = cameras_list
        self.h_division = h_division
        self.w_division = w_division

    def __len__(self):
        return len(self.cameras_list) * self.h_division * self.w_division 
    
    def __getitem__(self, idx):
        camera_idx = idx // (self.h_division * self.w_division)
        patch_idx = idx % (self.h_division * self.w_division)

        camera: Camera = self.cameras_list[camera_idx]
        h_patch = camera.image_height // self.h_division
        w_patch = camera.image_width // self.w_division

        v_start = h_patch * (patch_idx // self.w_division)
        in_last_row = (patch_idx // self.w_division) == (self.h_division - 1)
        v_end = camera.image_height if in_last_row else (v_start + h_patch)

        u_start = w_patch * (patch_idx % self.w_division)
        in_last_column = (patch_idx % self.w_division) == (self.w_division - 1)
        u_end = camera.image_width if in_last_column else (u_start + w_patch) 

        return Patch(camera=camera, uid=idx, v_start=v_start, v_end=v_end, u_start=u_start, u_end=u_end)


class DatasetRepeater(Dataset):
    '''
    torch.utils.data.Dataset for DDP and the convenience of debug
    '''
    def __init__(self, origin:Dataset, repeat_utill:int, empty:bool, step:int=1) -> None:
        super().__init__()
        self.origin = origin
        self.repeat_utill = repeat_utill
        self.empty = empty
        self.step = step
        
    def __len__(self):
        return self.repeat_utill

    def __getitem__(self, idx):     
        if self.empty:
            return None
        else:
            _i = (idx * self.step) % len(self.origin)
            return self.origin.__getitem__(_i)


class PartOfDataset(Dataset):
    '''
    torch.utils.data.Dataset for DDP and the convenience of debug
    '''
    def __init__(self, origin:Dataset, empty:bool, start:int=0, end:int=None) -> None:
        super().__init__()
        self.origin = origin
        self.start = start
        self.empty = empty
        if end is None:
            self.end = len(self.origin)
        else:
            self.end = end    
        self.len:int = self.end - self.start    
        
    def __len__(self):
        return self.len

    def __getitem__(self, idx):     
        if self.empty:
            return None
        else:
            _i = (idx + self.start) % len(self.origin)
            return self.origin.__getitem__(_i)


class GroupedItems(Dataset):
    '''
    using tuple[tuple[int]] to grup the items
    '''
    def __init__(self, origin:Dataset, groups:tuple) -> None:
        super().__init__()
        self.origin: Dataset = origin
        self.groups: tuple = groups
        
    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):     
        _group = self.groups[idx]
        _data = tuple(self.origin[_i] for _i in _group)
        
        return _group, _data

