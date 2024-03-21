#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getView2World, getWorld2View2, getProjectionMatrix


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).to(data_device)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).to(data_device)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.projection_matrix_inv = torch.tensor(getView2World(R, T, trans, scale), dtype=torch.float32).transpose(0, 1).to(data_device)


    def to_device(self, device):
        self.data_device = device
        self.original_image = self.original_image.to(device)
        self.world_view_transform = self.world_view_transform.to(device)
        self.projection_matrix = self.projection_matrix.to(device)
        self.projection_matrix_inv = self.projection_matrix_inv.to(device)
        self.full_proj_transform = self.full_proj_transform.to(device)
        self.camera_center = self.camera_center.to(device)
        return self

    @staticmethod
    def package_shape():
        return (16, 4)
    
    def pack_up(self, device=None):
        if device is None:
            device = self.data_device
        ret = torch.zeros((16, 4), dtype=torch.float32, device=self.data_device, requires_grad=False)
        ret[0:4, :] = self.world_view_transform
        ret[4:8, :] = self.full_proj_transform
        ret[8:12,:] = self.projection_matrix_inv
        ret[12, 0] = self.image_width
        ret[12, 1] = self.image_height
        ret[12, 2] = self.FoVx
        ret[12, 3] = self.FoVy
        ret[13, 0:3] = self.camera_center
        
        return ret.to(device)
    
    def get_depth(self, point3d:list) -> float:
        point3d_homo = torch.ones((1,4), dtype=torch.float32, device=self.data_device)
        point3d_homo[0, :3] = torch.tensor(point3d, dtype=torch.float32, device=self.data_device)
        point3d_view = torch.matmul(point3d_homo, self.world_view_transform)
        return float(point3d_view[0, 2])


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]


class ViewMessage(nn.Module):
    def __init__(self, package: torch.Tensor, id: int) -> None:
        """
        this class is built for the transport of Camera among GPUs
        package: package of Camera
        id: uuid
        """
        super().__init__()
        assert package.size()[0] == 16 and package.size()[1] == 4
        self.data_device = package.device
        self.id = id

        self.world_view_transform = package[0:4, :]
        self.full_proj_transform = package[4:8, :]
        self.projection_matrix_inv = package[8:12,:] 
        self.image_width = int(package[12, 0])
        self.image_height = int(package[12, 1])
        self.FoVx = package[12, 2]
        self.FoVy = package[12, 3]
        self.camera_center = package[13, 0:3]

    def to_device(self, device):
        self.data_device = device

        self.world_view_transform = self.world_view_transform.to(device)
        self.projection_matrix_inv = self.projection_matrix_inv.to(device)
        self.full_proj_transform = self.full_proj_transform.to(device)
        self.camera_center = self.camera_center.to(device)    

    def get_depth(self, point3d:list) -> float:
        point3d_homo = torch.ones((1,4), dtype=torch.float32, device=self.data_device)
        point3d_homo[0, :3] = torch.tensor(point3d, dtype=torch.float32, device=self.data_device)
        point3d_view = torch.matmul(point3d_homo, self.world_view_transform)
        return float(point3d_view[0, 2])
    
    def __str__(self) -> str:
        return 'ViewMessage(uuid={}, H={}, W={})'.format(
            self.id, self.image_height, self.image_width
        )
    

class CamerawithDepth(Camera):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, depth, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(CamerawithDepth, self).__init__(
            colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
            image_name, uid,
            trans, scale, data_device)

        if depth is not None:
            self.depth = depth.unsqueeze(0)
        else:
            self.depth = torch.zeros((1, self.image_height, self.image_width)).to(data_device)

        self.full_proj_transform = self.world_view_transform@self.projection_matrix
        self.camera_center = torch.from_numpy(np.linalg.inv(self.world_view_transform.cpu().numpy())[3, :3]).to(data_device)
        self.projection_matrix_inv = torch.tensor(getView2World(R, T, trans, scale), dtype=torch.float32).transpose(0, 1).to(data_device)

    def to_device(self, device):
        self.data_device = device
        self.original_image = self.original_image.to(device)
        self.depth = self.depth.to(device)
        self.world_view_transform = self.world_view_transform.to(device)
        self.projection_matrix = self.projection_matrix.to(device)
        self.projection_matrix_inv = self.projection_matrix_inv.to(device)
        self.full_proj_transform = self.full_proj_transform.to(device)
        self.camera_center = self.camera_center.to(device)
        return self

    def get_mini_clone(self, device='cpu'):
        ret = CamerawithDepth(
            self.colmap_id, self.R, self.T, self.FoVx, self.FoVy, torch.zeros((0,0)), torch.zeros((0,0)), None, 
            self.image_name, self.uid, 
            self.trans, self.scale, data_device=device)     
        ret.image_height = self.image_height
        ret.image_width = self.image_width
        return ret

    def pack_up(self, device=None):
        if device is None:
            device = self.data_device
        ret = torch.zeros((16, 4), dtype=torch.float32, device=self.data_device, requires_grad=False)
        ret[0:4, :] = self.world_view_transform
        ret[4:8, :] = self.full_proj_transform
        ret[8:12,:] = self.projection_matrix_inv
        ret[12, 0] = self.image_width
        ret[12, 1] = self.image_height
        ret[12, 2] = self.FoVx
        ret[12, 3] = self.FoVy
        ret[13, 0:3] = self.camera_center
        
        return ret.to(device)
    
    @staticmethod
    def package_shape():
        return (16, 4)

    def get_depth(self, point3d:list) -> float:
        """
        get depth of a point in camera space
        """
        point3d_homo = torch.ones((1,4), dtype=torch.float32, device=self.data_device)
        point3d_homo[0, :3] = torch.tensor(point3d, dtype=torch.float32, device=self.data_device)
        point3d_view = torch.matmul(point3d_homo, self.world_view_transform)
        return float(point3d_view[0, 2])


class EmptyCamera(nn.Module):
    '''
    This class is used in render new views
    It's obvious that new view has not gt_image nor gt_alpha_mask
    '''
    def __init__(self, colmap_id, R, T, FoVx, FoVy, width_height, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(EmptyCamera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.image_width = width_height[0]
        self.image_height = width_height[1]

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).to(data_device)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).to(data_device)
        self.full_proj_transform = self.world_view_transform@self.projection_matrix #(self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        # self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.camera_center = torch.from_numpy(np.linalg.inv(self.world_view_transform.cpu().numpy())[3, :3]).to(data_device)
        self.projection_matrix_inv = torch.tensor(getView2World(R, T, trans, scale), dtype=torch.float32).transpose(0, 1).to(data_device)

    def to_device(self, device):
        self.data_device = device
        # self.original_image = self.original_image.to(device)
        self.world_view_transform = self.world_view_transform.to(device)
        self.projection_matrix = self.projection_matrix.to(device)
        self.projection_matrix_inv = self.projection_matrix_inv.to(device)
        self.full_proj_transform = self.full_proj_transform.to(device)
        self.camera_center = self.camera_center.to(device)
        return self
    
    def pack_up(self, device=None):
        if device is None:
            device = self.data_device
        ret = torch.zeros((16, 4), dtype=torch.float32, device=self.data_device, requires_grad=False)
        ret[0:4, :] = self.world_view_transform
        ret[4:8, :] = self.full_proj_transform
        ret[8:12,:] = self.projection_matrix_inv
        ret[12, 0] = self.image_width
        ret[12, 1] = self.image_height
        ret[12, 2] = self.FoVx
        ret[12, 3] = self.FoVy
        ret[13, 0:3] = self.camera_center
        
        return ret.to(device)
    
    def get_depth(self, point3d:list) -> float:
        point3d_homo = torch.ones((1,4), dtype=torch.float32, device=self.data_device)
        point3d_homo[0, :3] = torch.tensor(point3d, dtype=torch.float32, device=self.data_device)
        point3d_view = torch.matmul(point3d_homo, self.world_view_transform)
        return float(point3d_view[0, 2])    