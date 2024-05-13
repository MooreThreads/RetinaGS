import json
import os
import numpy as np
from plyfile import PlyData, PlyElement

# 读取json文件，修改unit后存到另一个目录
def change_unit(json_file, save_json_file):
    with open(os.path.join(json_file), "r") as f:
        mate = json.load(f)
    
    for i, frame in enumerate(mate['frames']):
        c2w = np.array(mate['frames'][i]['transform_matrix'])
        # 单位从100m改到1m
        c2w[:3,3] *= 100.0
        mate['frames'][i]['transform_matrix'] = c2w.tolist()
    
    with open(save_json_file, 'w') as outfile:
        json.dump(mate, outfile, indent=2)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T
    return positions, colors

def change_unit_pointcloud(ply_path, save_ply_path):
    xyzs, rgbs = fetchPly(ply_path)
    xyzs = xyzs * 100.0
    storePly(save_ply_path, xyzs, rgbs)

def random_sample_pointcloud(ply_path, save_ply_path, num_samples):
    xyzs, rgbs = fetchPly(ply_path)
    sampled_indices = np.random.choice(xyzs.shape[0], num_samples, replace=False)
    xyzs = xyzs[sampled_indices]
    rgbs = rgbs[sampled_indices]
    storePly(save_ply_path, xyzs, rgbs)
    
if __name__ == '__main__':
    # json文件单位
    json_dir = '/root/Nerf/Data/MatrixCity/bdaibdai___MatrixCity/small_city/aerial/pose/block_all'
    save_json_dir = json_dir + '_unit-1m_val'
    os.makedirs(save_json_dir, exist_ok=True)
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    # json_files = ["transforms_all.json"]
    for json_file in json_files:
        save_json_file = os.path.join(save_json_dir, json_file)
        change_unit(os.path.join(json_dir, json_file), save_json_file)
    
    # ply文件单位
    # ply_path = '/root/Nerf/Code/MatrixCity-main/ply/street_train_block_A_interval-20.ply'
    # pure_name = os.path.splitext(os.path.basename(ply_path))[0]
    # save_ply_path = os.path.join(os.path.dirname(ply_path), pure_name+'_uint-1m.ply')
    # change_unit_pointcloud(ply_path, save_ply_path)
    