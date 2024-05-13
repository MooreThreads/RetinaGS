import argparse
import os
import json
import imageio
import numpy as np

# TODO: fuse the train and test
if __name__ == "__main__":
    HIGH_NAME = '/root/Nerf/Data/MatrixCity/bdaibdai___MatrixCity/small_city/aerial/pose/block_all_unit-1m'
    ROAD_NAME = '/root/Nerf/Data/MatrixCity/bdaibdai___MatrixCity/small_city/street/pose/Block_all_unit-1m'
    OUT_NAME = '/root/Nerf/Data/MatrixCity/bdaibdai___MatrixCity/small_city/aerial_street/pose/Block-all_aerial-1_street-dense-1_unit-1m'

    metas = {}
    # road_metas={}
    # assume all situations share the same intri
    with open(os.path.join(HIGH_NAME,f"transforms_train.json"), "r") as f:
        meta_high_train = json.load(f)
    high_angle_x = meta_high_train['camera_angle_x']
    high_fl_x = meta_high_train['fl_x']
    high_fl_y = meta_high_train['fl_y']
    high_cx = meta_high_train['cx']
    high_cy = meta_high_train['cy']
    high_w = meta_high_train['w']
    high_h = meta_high_train['h']
    
    with open(os.path.join(ROAD_NAME,f"transforms_train.json"), "r") as f:
        meta_road_train = json.load(f)
    road_angle_x = meta_road_train['camera_angle_x']
    road_fl_x = meta_road_train['fl_x']
    road_fl_y = meta_road_train['fl_y']
    road_cx = meta_road_train['cx']
    road_cy = meta_road_train['cy']
    road_w = meta_road_train['w']
    road_h = meta_road_train['h']
    
    train_json = {
        "camera_model": "SIMPLE_PINHOLE",
        "frames": []
        }

    test_json = {
        "camera_model": "SIMPLE_PINHOLE",
        "frames": []
    }
        
    
    split = ['train', 'test']
        
    data_type = ['high', 'road']

    for data in data_type:
        metas[data]={}
        for s in split:
            name = HIGH_NAME if data == 'high' else ROAD_NAME
            with open(
                    os.path.join(name,
                                 f"transforms_{s}.json"), "r") as f:
                metas[data][s] = json.load(f)

    for data in data_type:

        basedir = os.path.join("../", HIGH_NAME) if data == 'high' else os.path.join(
                                   "../", ROAD_NAME)
        camera_angle_x = high_angle_x if data=='high' else road_angle_x
        fl_x = high_fl_x if data=='high' else road_fl_x
        fl_y = high_fl_y if data=='high' else road_fl_y
        cx = high_cx if data=='high' else road_cx
        cy = high_cy if data=='high' else road_cy
        w = high_w if data=='high' else road_w
        h = high_h if data=='high' else road_h

        for s in split:
            meta = metas[data][s]
            for i, frame in enumerate(meta['frames']):
                fname = os.path.join(basedir, frame['file_path'])
                # 空中重复五倍，地面不变
                TIME = 1
                times = TIME if data=='high' else 1
                for _time in range(times):
                    if s == "train":
                        train_json['frames'].append({
                            'camera_angle_x': camera_angle_x, 
                            'fl_x': fl_x,
                            'fl_y': fl_y,
                            'cx': cx,
                            'cy': cy,
                            'w': w,
                            'h': h,
                            'file_path':fname,
                            'transform_matrix':frame['transform_matrix']
                        })
                    elif s == "test":
                        test_json['frames'].append({
                            'camera_angle_x': camera_angle_x, 
                            'fl_x': fl_x,
                            'fl_y': fl_y,
                            'cx': cx,
                            'cy': cy,
                            'w': w,
                            'h': h,
                            'file_path':fname,
                            'transform_matrix':frame['transform_matrix']
                        })
            
    # save
    save_dir = os.path.join(OUT_NAME)
    os.makedirs(save_dir,exist_ok=True)
    with open(os.path.join(save_dir, 'transforms_train.json'),"w") as outfile:
        json.dump(train_json, outfile, indent=2)
    with open(os.path.join(save_dir, 'transforms_test.json'),"w") as outfile:
        json.dump(test_json, outfile, indent=2)
  