import argparse
import os
import json
import imageio
import numpy as np
from tqdm import tqdm

def split_train_test(json_floder_path, val_interval):
    # 读取transforms_all.json
    with open(os.path.join(json_floder_path, "transforms_all.json"), "r") as f:
        meta_all = json.load(f)
    
    angle_x = meta_all['camera_angle_x']
    w = meta_all['w']
    h = meta_all['h']
    fl_x = meta_all['fl_x']
    fl_y = meta_all['fl_y']
    k1 = 0
    k2 = 0
    k3 = 0
    k4 = 0
    p1 = 0
    p2 = 0
    cx = meta_all['cx']
    cy = meta_all['cy']
    all_frames = meta_all['frames']
    
    train_frames = [c for idx, c in enumerate(all_frames) if idx % val_interval != 0]
    test_frames = [c for idx, c in enumerate(all_frames) if idx % val_interval == 0]           
    
    train_pose = {
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
            "frames": train_frames
        }

    test_pose = {
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
            "frames": test_frames
        }
    
    with open(os.path.join(json_floder_path, 'transforms_train.json'),"w") as outfile_train:
        json.dump(train_pose, outfile_train, indent=2)
    with open(os.path.join(json_floder_path, 'transforms_test.json'),"w") as outfile_test:
        json.dump(test_pose, outfile_test, indent=2)

if __name__ == '__main__':
    json_floder_path = "/root/Nerf/Data/MatrixCity/bdaibdai___MatrixCity/small_city/street/pose/block_A_unit-1m_dense_sub-1_val"
    val_interval = 8
    split_train_test(json_floder_path, val_interval)