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
import math
from utils.graphics_utils import getView2World, getWorld2View2, getProjectionMatrix


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.id = uid
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


class Patch(nn.Module):
    """
    -----------> u-axis
    | image|
    |------|
    |
    v v-axis
    
    """
    def __init__(self, camera:Camera, uid, v_start:int, v_end:int, u_start:int, u_end:int):
        super(Patch, self).__init__()

        self.uid = uid
        self.id = uid
        self.parent_uid = camera.uid
        self.v_start = v_start
        self.v_end = v_end
        self.u_start = u_start
        self.u_end = u_end
        # copy from parent
        self.colmap_id = camera.colmap_id
        self.R = camera.R
        self.T = camera.T
        self.FoVx = camera.FoVx
        self.FoVy = camera.FoVy
        self.image_name = camera.image_name
        self.data_device = camera.data_device
    
        assert 0 <= v_start < v_end <= camera.image_height
        assert 0 <= u_start < u_end <= camera.image_width

        self.original_image = camera.original_image[:, v_start:v_end, u_start:u_end]   # (3, H, W)

        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        self.complete_width = camera.image_width
        self.complete_height = camera.image_height

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = camera.trans
        self.scale = camera.scale

        self.world_view_transform = camera.world_view_transform
        self.projection_matrix = camera.projection_matrix
        self.full_proj_transform = camera.full_proj_transform
        self.camera_center = camera.camera_center
        self.projection_matrix_inv = camera.projection_matrix_inv

        self.all_tiles = self.get_tile_map(tile_size=16)

    def to_device(self, device):
        self.data_device = device
        self.original_image = self.original_image.to(device)
        self.world_view_transform = self.world_view_transform.to(device)
        self.projection_matrix = self.projection_matrix.to(device)
        self.projection_matrix_inv = self.projection_matrix_inv.to(device)
        self.full_proj_transform = self.full_proj_transform.to(device)
        self.camera_center = self.camera_center.to(device)

        self.all_tiles = self.all_tiles.to(device)
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
        ret[14, 0] = self.v_start
        ret[14, 1] = self.v_end
        ret[14, 2] = self.u_start
        ret[14, 3] = self.u_end
        
        return ret.to(device)
    
    def get_depth(self, point3d:list) -> float:
        point3d_homo = torch.ones((1,4), dtype=torch.float32, device=self.data_device)
        point3d_homo[0, :3] = torch.tensor(point3d, dtype=torch.float32, device=self.data_device)
        point3d_view = torch.matmul(point3d_homo, self.world_view_transform)
        return float(point3d_view[0, 2])
    
    def get_padding_range(self, tile_size=16):
        # [tile_start, tile_end]
        v_tile_start = self.v_start // tile_size
        v_tile_end = (self.v_end - 1) // tile_size
        u_tile_start = self.u_start // tile_size
        u_tile_end = (self.u_end - 1) // tile_size
        return v_tile_start*tile_size, (v_tile_end+1)*tile_size, u_tile_start*tile_size, (u_tile_end+1)*tile_size

    def get_tile_map(self, tile_size=16):
        v_tile_start = self.v_start // tile_size
        v_tile_end = (self.v_end - 1) // tile_size
        u_tile_start = self.u_start // tile_size
        u_tile_end = (self.u_end - 1) // tile_size

        v_range, u_range = math.ceil(self.complete_height / tile_size), math.ceil(self.complete_width / tile_size)  
        complete_map = 1 + torch.arange(0, v_range * u_range, requires_grad=False).int().reshape(v_range, u_range)
        tile_map = complete_map[v_tile_start:(v_tile_end + 1), u_tile_start:(u_tile_end + 1)]
        return tile_map.reshape(-1).to(self.data_device)

    def tiles2patch(self, tiles:torch.Tensor):
        # tiles shall be shape of (num_tile, channel, h_tile, w_tile)
        num_tile, channel, h_tile, w_tile = tiles.shape
        v_tile_start = self.v_start // h_tile
        v_tile_end = (self.v_end - 1) // h_tile
        u_tile_start = self.u_start // w_tile
        u_tile_end = (self.u_end - 1) // w_tile

        tilesNumY = v_tile_end - v_tile_start + 1
        tilesNumX = u_tile_end - u_tile_start + 1
        tile_grid = tiles.transpose(0,1).reshape(channel, tilesNumY, tilesNumX, h_tile, w_tile).transpose(-2,-3)
        patch_padding = tile_grid.reshape(channel, tilesNumY*h_tile,tilesNumX*w_tile)
        
        v_start_in_tile, u_start_in_tile = self.v_start % h_tile, self.u_start % w_tile
        patch = patch_padding[:, v_start_in_tile:(v_start_in_tile + self.image_height) , u_start_in_tile:(u_start_in_tile + self.image_width)]
        return patch
    

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
        1. this class is built for the transport of Camera among GPUs, as some GPUs may not load dataset from hard disk
        2. make sure that a Camera instance can always work as an alternate ViewMessage 
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