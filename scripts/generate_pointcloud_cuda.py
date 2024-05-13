from load_data import load_depth
import json
import numpy as np
import os
from PIL import Image
import math
from plyfile import PlyData, PlyElement
from tqdm import tqdm
# import open3d as o3d
import torch
from torch.utils.data import Dataset, DataLoader
import time

class MatrixCity_Dataset(Dataset):
    def __init__(self, type:str, unit:str, pose_path:str) -> None:
        super().__init__()
        with open(pose_path, "r") as json_file:
            self.contents = json.load(json_file)
            self.angle_x = self.contents["camera_angle_x"]
            self.extension = ""
            self.frames = self.contents["frames"]
            self.H_interval, self.W_interval = interval, interval
            self.pose_path = pose_path
            
            if type == 'aerial':
                self.H, self.W = 1080, 1920
            elif type == 'street':
                self.H, self.W = 1000, 1000
            self.unit = unit
        
    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        
        # 初始化地址
        frame = self.frames[idx]
        cam_name = os.path.join(os.path.dirname(self.pose_path), frame["file_path"] + self.extension)
        normalized_path = os.path.normpath(cam_name)
        parts = normalized_path.split(os.sep)
        aerial_street, train_test, block_order, filename = parts[-4:]
        pure_filename = os.path.splitext(filename)[0]
        depth_folder = '/root/Nerf/Data/MatrixCity/bdaibdai___MatrixCity/small_city_depth'
        depth_name = os.path.join(depth_folder, aerial_street, train_test, block_order + '_depth', pure_filename + '.exr')

        # 读RGB和Depth
        image = Image.open(cam_name)
        im_data = np.array(image.convert("RGBA"))
        norm_data = im_data / 255.0
        bg = np.array([0, 0, 0])
        arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
        image_array = torch.tensor(np.asarray(image)).cuda()
        depth = load_depth(depth_name)
        depth = torch.tensor(np.asarray(depth)).cuda()
        depth_mask_all = (depth <= 2.00)
        if self.unit == '100m':
            depth = depth * 1                
        elif self.unit == '1m':
            depth = depth * 100
        
        # Pose和内参
        RT_C2W = torch.tensor(frame["transform_matrix"]).cuda()
        RT_C2W[:3, 1:3] *= -1       
        f_x = float(.5 * self.W / np.tan(.5 * self.angle_x))
        f_y = f_x
        
        return RT_C2W, f_x, f_y, image_array, depth, depth_mask_all
    
    @staticmethod
    def get_batch(cameras):
        return list(cameras)


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

from typing import NamedTuple
class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array
    
def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T
    return positions, colors

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    
    
    for i, name in tqdm(enumerate(dtype)):
        elements[name[0]] = attributes[:, i]
    print('transport done!')
    # elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    print('ply data done!')
    ply_data.write(path) 
    print('save at ' + path)
    

    
def getWorld2View(R, t):
    Rt = torch.zeros((4, 4)).cuda()
    Rt[:3, :3] = R # debug
    Rt[:3, 3] = t # t存的时候没有做相应变换
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def generate_pointcloud_cuda(pose_path, depth_folder, save_path='/root/Nerf/Code/MatrixCity-main/ply', interval=10, type='aerial', debug=False, unit='100m'):
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    
    _dataset = MatrixCity_Dataset(type, unit, pose_path)
    dataloader = DataLoader(_dataset, batch_size=1, num_workers=32, prefetch_factor=4, shuffle=False, collate_fn=MatrixCity_Dataset.get_batch)
    # dataloader = DataLoader(_dataset, batch_size=1, num_workers=0, shuffle=False, collate_fn=MatrixCity_Dataset.get_batch)
    H_interval, W_interval = interval, interval
    if type == 'aerial': 
        H, W = 1080, 1920
        f_x, f_y = 2317.6449482429634, 2317.6449482429634
    elif type == 'street':
        H, W = 1000, 1000
        f_x, f_y = 499.9999781443055, 499.9999781443055
    
    if debug:
        N_image = 32
        xyzs = torch.zeros((N_image, int((H / H_interval) * (W / W_interval)), 3), dtype=torch.float32).cuda()
        rgbs = torch.zeros((N_image, int((H / H_interval) * (W / W_interval)), 3), dtype=torch.uint8).cuda()
        depth_masks = torch.full((N_image, int((H / H_interval) * (W / W_interval))), False, dtype=torch.bool).cuda()
    else:
        xyzs = torch.zeros((len(dataloader), int((H / H_interval) * (W / W_interval)), 3), dtype=torch.float32).cuda()
        rgbs = torch.zeros((len(dataloader), int((H / H_interval) * (W / W_interval)), 3), dtype=torch.uint8).cuda()
        depth_masks = torch.full((len(dataloader), int((H / H_interval) * (W / W_interval))), False, dtype=torch.bool).cuda()
    
    N_H, N_W = int(H / H_interval), int(W / W_interval)
    depth_mask = torch.full((N_H * N_W, ), False, dtype=torch.bool).cuda()
    v, u = torch.meshgrid(torch.arange(0,H,H_interval), torch.arange(0,W,W_interval))
    v = v.reshape(-1).cuda()
    u = u.reshape(-1).cuda()
    uv_reshape = torch.stack([u, v, torch.ones_like(u)], dim=-1).cuda().float()
    uv_reshape = uv_reshape.transpose(0, 1)
    project_matrix = getProjectionMatrix(f_x, f_y, H=H, W=W)
    project_matrix_inv = torch.inverse(project_matrix)
        
    progress_bar = tqdm(range(0, len(dataloader)+1), desc="progress")    
    for _i, data in enumerate(dataloader):   
        
        progress_bar.update(1)
        
        if debug:
            if _i >= N_image:
                break
            
        RT_C2W, f_x, f_y, image_array, depth, depth_mask_all  =  data[0]       
        xyz, rgb, depth_mask = gain_ply_cur_pose_v2(RT_C2W, image_array, depth, depth_mask_all, project_matrix_inv, u, v, uv_reshape, depth_mask)
        depth_masks[_i, :] = depth_mask
        xyzs[_i, :, :] = xyz
        rgbs[_i, :, :] = rgb

    return depth_masks, xyzs, rgbs

def getProjectionMatrix(f_x, f_y, H, W):
    cx = W / 2
    cy = H / 2
    
    P = torch.zeros((3, 3)).cuda()
    P[0, 0] = f_x
    P[1, 1] = f_y
    P[0, 2] = cx
    P[1, 2] = cy    
    P[2, 2] = 1.0
    
    return P

def gain_ply_cur_pose_v2(RT_C2W, image, depth, depth_mask_all, project_matrix_inv, u, v, uv_reshape, depth_mask):
    rgb = image[v, u]
    depth_mask = depth_mask_all[v, u]
    xyz = compute_3d_point_v2(RT_C2W, project_matrix_inv, u, v, depth, uv_reshape)
    
    return xyz, rgb, depth_mask

def compute_3d_point_v2(RT_C2W, project_matrix_inv, u, v, z, uv_reshape):
    xyz_camera = z[v, u] * torch.matmul(project_matrix_inv, uv_reshape)
    R_C2W = RT_C2W[0:3, 0:3]
    # T_C2W = RT_C2W[0:3, 3].T
    T_C2W = RT_C2W[0:3, 3].reshape(3, 1)
    xyz_world = torch.matmul(R_C2W, xyz_camera) + T_C2W
    return xyz_world.transpose(0, 1)

def random_sample_pointcloud(ply_path, save_ply_path, num_samples):
    xyzs, rgbs = fetchPly(ply_path)
    print("Load Done!")
    print(xyzs.shape)
    sampled_indices = np.random.choice(xyzs.shape[0], num_samples, replace=False)
    xyzs = xyzs[sampled_indices]
    rgbs = rgbs[sampled_indices]
    print("Random Sample Done!")
    
    start_time = time.time()
    storePly(save_ply_path, xyzs, rgbs)
    end_time = time.time()    
    print(f"The function took {end_time - start_time} seconds to complete.")
    
    print(xyzs.shape)
    print("Save Done!")

def random_sample_pointcloud_division(ply_path, save_ply_path, sample_rate):
    xyzs, rgbs = fetchPly(ply_path)
    print("Load Done!")
    print(xyzs.shape)
    num_samples = int(xyzs.shape[0]/sample_rate)
    sampled_indices = np.random.choice(xyzs.shape[0], num_samples, replace=False)
    xyzs = xyzs[sampled_indices]
    rgbs = rgbs[sampled_indices]
    print("Random Sample Done!")
    
    start_time = time.time()
    storePly(save_ply_path, xyzs, rgbs)
    end_time = time.time()    
    print(f"The function took {end_time - start_time} seconds to complete.")
    
    print(xyzs.shape)
    print("Save Done!")

def sky_ball_addition(ply_path, save_ply_path, sky_ball_rate):
    xyzs, rgbs = fetchPly(ply_path)
    print("Load Done!")
    print(xyzs.shape)
    num_sky_ball = int(xyzs.shape[0]*sky_ball_rate)
    
    # 计算点云的中心，xy轴以全部的平均，z轴以最小值
    center = np.zeros(3)
    center[0:2] = np.mean(xyzs, axis=0)[0:2]
    center[2] = np.min(xyzs, axis=0)[2]
    
    print("center is ", center)
    
    # 计算点云中到平均点的最大距离，作为天空球半径
    radius_min = np.max(np.sqrt(np.sum((xyzs - center)**2, axis=1)))
    radius = radius_min * 1
    print("max distance is ", radius)
    
    # 生成sky_ball，颜色设置为白色
    theta = 2 * np.pi * np.random.rand(num_sky_ball)
    phi = np.arccos(2 * np.random.rand(num_sky_ball) - 1)
    sky_ball_x = center[0] + radius * np.sin(phi) * np.cos(theta)
    sky_ball_y = center[1] + radius * np.sin(phi) * np.sin(theta)
    sky_ball_z = center[2] + np.abs(radius * np.cos(phi))
    sky_ball_xyz = np.stack((sky_ball_x, sky_ball_y, sky_ball_z), axis=-1)
    sky_ball_rgb = np.full((num_sky_ball, 3), 255)
    
    # 合并为新的点云    
    xyzs = np.concatenate((xyzs, sky_ball_xyz), axis=0)
    rgbs = np.concatenate((rgbs, sky_ball_rgb), axis=0)
    
    start_time = time.time()
    storePly(save_ply_path, xyzs, rgbs)
    end_time = time.time()    
    print(f"The function took {end_time - start_time} seconds to complete.")
    
    print(xyzs.shape)
    print("Save Done!")


if __name__ == '__main__':
    # # aerial
    # pose_path = '/root/Nerf/Data/MatrixCity/bdaibdai___MatrixCity/small_city/aerial/pose/block_all_unit-1m_val/transforms_test.json'
    # depth_folder = '/root/Nerf/Data/MatrixCity/bdaibdai___MatrixCity/small_city_depth'
    # save_path ='/root/Nerf/Code/MatrixCity-main/ply'
    # interval = 8
    # type ='aerial'
    # _, xyzs_aerial, rgbs_aerial = generate_pointcloud_cuda(pose_path, depth_folder, save_path, interval, type, False, unit='1m')
    # xyzs_aerial = xyzs_aerial.detach().cpu().contiguous().clone().numpy()
    # rgbs_aerial = rgbs_aerial.detach().cpu().contiguous().clone().numpy()
    # storePly(os.path.join(save_path,'aerial_block_A_unit-1m_sub-1_zoom-out-interval_8.ply'), np.concatenate(xyzs_aerial, axis=0), np.concatenate(rgbs_aerial, axis=0))
    
    # downsample('/jfs/shengyi.chen/HT/Code/MatrixCity-main/ply/aerial_block-all_interval-4_2.ply', voxel_size=0.01)

    
    # ply_path = '/root/Nerf/Code/MatrixCity-main/ply/block_A_unit-1m_sub-1/aerial_block_A_unit-1m_sub-1_zoom-out-interval_8.ply'
    # save_ply_path = '/root/Nerf/Code/MatrixCity-main/ply/block_A_unit-1m_sub-1/aerial_block_A_unit-1m_sub-1_zoom-out-interval_8_3Mi.ply'
    # random_sample_pointcloud(ply_path, save_ply_path, 3_000_000)
    
    ply_path = '/root/Nerf/Code/MatrixCity-main/ply/block_A_unit-1m_sub-1/aerial-3Mi_street-dense-3Mi.ply'
    save_ply_path = '/root/Nerf/Code/MatrixCity-main/ply/block_A_unit-1m_sub-1/aerial-3Mi_street-dense-3Mi_with-skyball-percentage-1.ply'
    sky_ball_addition(ply_path, save_ply_path, 0.01)
    
    # ply_path = '/root/Nerf/Code/MatrixCity-main/ply/street_test/Block_all_unit-1m_choice-100_intelval-1.ply'
    # save_ply_path = '/root/Nerf/Code/MatrixCity-main/ply/street_test/Block_all_unit-1m_choice-100_intelval-1_random_sample-8.ply'
    # random_sample_pointcloud_division(ply_path, save_ply_path, 8)


    # # street
    # pose_path = '/root/Nerf/Data/MatrixCity/bdaibdai___MatrixCity/small_city/street/pose/Block_all_unit-1m/transforms_dense_all.json'
    # depth_folder = '/root/Nerf/Data/MatrixCity/bdaibdai___MatrixCity/small_city_depth'
    # save_path ='/root/Nerf/Code/MatrixCity-main/ply/street_test'
    # interval = 4
    # type ='street'
    # depth_masks, xyzs_street, rgbs_street = generate_pointcloud_cuda(pose_path, depth_folder, save_path, interval, type, False, unit='1m')
    # depth_masks = depth_masks.view(-1)
    # xyzs_street = xyzs_street.view(-1, 3)
    # rgbs_street = rgbs_street.view(-1, 3)
    # xyzs_masked = xyzs_street[depth_masks].detach().cpu().contiguous().clone().numpy()
    # rgbs_masked = rgbs_street[depth_masks].detach().cpu().contiguous().clone().numpy()
    # storePly(os.path.join(save_path,'street_block_A_unit-1m_sub-1-interval_4.ply'), xyzs_masked, rgbs_masked)
    
    # # 合并
    # aerial_ply = '/root/Nerf/Code/MatrixCity-main/ply/block_A_unit-1m_sub-1/street_block_A_unit-1m_sub-1-interval_4-3Mi.ply'
    # street_ply = '/root/Nerf/Code/MatrixCity-main/ply/block_A_unit-1m_sub-1/aerial_block_A_unit-1m_sub-1_zoom-out-interval_8_3Mi.ply'
    # aerial_xyzs, aerial_rgbs = fetchPly(aerial_ply)
    # print("Number of aerial pointcloud:", aerial_xyzs.shape[0])        
    # street_xyzs, street_rgbs = fetchPly(street_ply)
    # print("Number of street pointcloud:", street_xyzs.shape[0])    
    # # 保存
    # xyzs = np.concatenate((aerial_xyzs, street_xyzs), axis=0)
    # rgbs = np.concatenate((aerial_rgbs, street_rgbs), axis=0)
    # print("Number of merge pointcloud:", xyzs.shape[0])        
    # ply_path = '/root/Nerf/Code/MatrixCity-main/ply/block_A_unit-1m_sub-1/aerial-3Mi_street-dense-3Mi.ply'    
    # storePly(ply_path, xyzs, rgbs)
    # print(ply_path, 'done!')
    
    # test
    # ply_path = '/root/ply/aerial_street-dense_block_A.ply'
    # xyzs, rgbs = fetchPly(ply_path)
    # print("Number of pointcloud:", xyzs.shape[0])
    
    # # 降采样
    # ply_path = '/root/Nerf/Code/MatrixCity-main/ply/street_block-A_dense_unit-1m-all_interval-10_voxel-donwsample-1.ply'
    # downsample(ply_path, voxel_size=0.3)
        
        
