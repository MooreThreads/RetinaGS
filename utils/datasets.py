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
from scene.cameras import Camera, EmptyCamera, ViewMessage
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

        return loadCam(self.args, id=idx, cam_info=full_info, resolution_scale=self.resolution_scale)   

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