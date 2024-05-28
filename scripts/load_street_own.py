import argparse
import os
import json
import imageio
import numpy as np
from tqdm import tqdm

def load_street():
    STREET_BLOCK_NAME = "Block_all"
    
    os.makedirs(os.path.join('/root/Nerf/Data/MatrixCity/bdaibdai___MatrixCity/small_city/street/pose', STREET_BLOCK_NAME),exist_ok=True)
    
    street_dirs=[]
    base_path = '/root/Nerf/Data/MatrixCity/bdaibdai___MatrixCity/small_city/street/train_dense'
    street_files = os.listdir(base_path)
    for street_file in street_files:
        if not os.path.isfile(os.path.join(base_path,street_file)):
            street_dirs.append(street_file)
    
    all_frames=[]
    for street_dir in street_dirs:
        if os.path.exists(os.path.join(base_path,street_dir,"transforms.json")):
            with open(os.path.join(base_path,street_dir,"transforms.json"), "r") as f:
                tj = json.load(f)
            
            for _i, frame in tqdm(enumerate(tj['frames']), total=len(tj['frames'])):
                file_path = os.path.join("/root/Nerf/Data/MatrixCity/bdaibdai___MatrixCity/small_city/street/train_dense", street_dir, str(frame['frame_index']).zfill(4)+'.png')
                c2w = np.array(frame['rot_mat'])
                c2w[:3,:3] *= 100
                all_frames.append({'file_path':file_path,'transform_matrix':c2w.tolist()})
    
    angle_x = tj['camera_angle_x']
    w = float(1000)
    h = float(1000)
    fl_x = float(.5 * w / np.tan(.5 * angle_x))
    fl_y = fl_x
    k1 = 0
    k2 = 0
    k3 = 0
    k4 = 0
    p1 = 0
    p2 = 0
    cx = w / 2
    cy = h / 2
    
    pose = {
            "camera_angle_x": angle_x,
            "fl_x": fl_x,
            "fl_y": fl_y,
            "k1": k1,
            "k2": k2,
            "k3": k3,
            "k4": k4,
            "p1": p1,
            "p2": p2,
            "cx": cx,
            "cy": cy,
            "w": w,
            "h": h,
            "frames": all_frames
        }
    
    save_dir = os.path.join('/root/Nerf/Data/MatrixCity/bdaibdai___MatrixCity/small_city/street/pose', STREET_BLOCK_NAME)
    with open(os.path.join(save_dir, 'transforms_dense_all.json'),"w") as outfile:
        json.dump(pose, outfile, indent=2)

if __name__ == '__main__':
    load_street()