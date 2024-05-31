# we need to get sfm points with colmap using known camera/image

import argparse
import collections
import os, sys, json

import struct
from argparse import ArgumentParser, Namespace
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from tqdm import tqdm
import glob
from torch.utils.data import Dataset, DataLoader

import zipfile
from zipfile import ZipFile

RDF_TO_DRB = torch.FloatTensor([[0, 1, 0],
                                [1, 0, 0],
                                [0, 0, -1]])
DRB_TO_RDF = torch.FloatTensor([[0, 1, 0],
                                [1, 0, 0],
                                [0, 0, -1]])

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])


def readCamerasFromTransforms(transformsfile):
    img_infos, cam_infos = [], []

    with open(transformsfile) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        fl_x, fl_y = contents["fl_x"], contents["fl_y"]
        cx, cy = contents["cx"], contents["cy"]
        W, H = int(contents["w"]), int(contents["h"])
        for idx, frame in enumerate(frames):
            cam_name = frame["file_path"]

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c_full = np.linalg.inv(c2w)   #(4, 4)

            i = idx
            image_tvec = w2c_full[:3, 3].tolist()
            q_xyzw = Rotation.from_matrix(w2c_full[:3, :3]).as_quat()
            image_qvec = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
            
            short_name = os.path.basename(cam_name)
            block_name = os.path.basename(os.path.dirname(cam_name))

            # Image list with two lines of data per image:
            #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
            #   POINTS2D[] as (X, Y, POINT3D_ID)
            img_info = Image(id=i, camera_id=i, 
                  qvec=image_qvec, tvec=image_tvec, 
                  name=block_name + '_' + short_name, 
                  xys=[], point3D_ids=[])
      
            # 2 PINHOLE 3072 2304 2560.56 2560.56 1536 1152
            camera_info = Camera(id=i, model='PINHOLE', 
                    width=W, height=H, params=[fl_x, fl_y, cx, cy])
        
            img_infos.append(img_info)
            cam_infos.append(camera_info)

    return img_infos, cam_infos

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]])


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


def write_cameras_text(cameras:list, path:str):
    lines = []
    for camera in cameras:
        # assert isinstance(camera, Camera)
        camera_id = camera.id
        model = camera.model
        width = camera.width
        height = camera.height
        params = camera.params
        # CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
        elems = [camera_id, model, width, height, *params]
        elems = [str(e) for e in elems]
        lines.append(' '.join(elems)+'\n')

    with open(path, 'w') as f:
        f.writelines(lines) 


def write_images_text(images:list, path:str):
    # write images with empty mapped points3d
    # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME follwed by two \n
    lines = []
    for image in images:
        assert isinstance(image, Image)
        image_id = image.id
        qvec = image.qvec
        tvec = image.tvec
        camera_id = image.camera_id
        name = image.name
        elems = [image_id, *qvec, *tvec, camera_id, name]
        elems = [str(e) for e in elems]
        lines.append(' '.join(elems)+'\n\n')

    with open(path, 'w') as f:
        f.writelines(lines)    
        

def colmap_2_mega(image: Image, camera:Camera, origin_drb, pose_scale_factor):
    w2c = torch.eye(4)
    w2c[:3, :3] = torch.FloatTensor(qvec2rotmat(image.qvec))
    w2c[:3, 3] = torch.FloatTensor(image.tvec)
    c2w = torch.inverse(w2c)

    c2w = torch.hstack((
        RDF_TO_DRB @ c2w[:3, :3] @ torch.inverse(RDF_TO_DRB),
        RDF_TO_DRB @ c2w[:3, 3:]
    ))

    assert camera.model == 'SIMPLE_PINHOLE', camera.model

    camera_matrix = np.array([[camera.params[0], 0, camera.params[1]],
                                [0, camera.params[0], camera.params[2]],
                                [0, 0, 1]])


    camera_in_drb = c2w + 0
    camera_in_drb[:, 3] = (camera_in_drb[:, 3] - origin_drb) / pose_scale_factor

    assert np.logical_and(camera_in_drb >= -1, camera_in_drb <= 1).all()

    c2w_mega =  torch.cat(
                    [camera_in_drb[:, 1:2], -camera_in_drb[:, :1], camera_in_drb[:, 2:4]],
                    -1)  
    intrinsics_mega = torch.FloatTensor(
                    [camera_matrix[0][0], camera_matrix[1][1], camera_matrix[0][2], camera_matrix[1][2]])
    return c2w_mega, intrinsics_mega


if __name__ == "__main__":
    parser = ArgumentParser(description="build points3D.txt cameras.txt images.txt in colmap format")
    parser.add_argument('--json', type=str)
    parser.add_argument('--output_path', type=str)
    args = parser.parse_args(sys.argv[1:])

    # process train_set
    out_path = args.output_path
    out_path_train = out_path
    out_model_path = os.path.join(out_path, 'sparse')
    out_img_path = os.path.join(out_path, 'images')

    os.makedirs(out_path, exist_ok=True)
    os.makedirs(out_model_path, exist_ok=True)
    os.makedirs(out_img_path, exist_ok=True)

    images, cameras = readCamerasFromTransforms(args.json)
    
    write_cameras_text(cameras=cameras, path=os.path.join(out_model_path, 'cameras.txt'))
    write_images_text(images=images, path=os.path.join(out_model_path, 'images.txt'))
    with open(os.path.join(out_model_path, 'points3D.txt'), 'w') as f:
        pass



