# we need to get sfm points with colmap using known camera/image

import argparse
import collections
import os
import struct
from argparse import Namespace
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

class TransData(Dataset):
    def __init__(self, path, out_img_path, downsample_scale, split='train', normalized=False) -> None:
        super().__init__()
        self.path = path
        self.out_img_path = out_img_path
        self.downsample_scale = downsample_scale

        self.coordinates_path = os.path.join(path, 'coordinates.pt')
        self.train_meta_dir = os.path.join(path, split, 'metadata')
        self.train_rgbs_dir = os.path.join(path, split, 'rgbs')
        self.normalized = normalized

        info = torch.load(self.coordinates_path)
        
        if self.normalized:
            self.pose_scale_factor = 1
            self.origin_drb = info['origin_drb'] * 0
        else:
            self.pose_scale_factor = info['pose_scale_factor']
            self.origin_drb = info['origin_drb']

        # millon_19 use .jpg
        jpgs = glob.glob(
            os.path.join(self.train_rgbs_dir, '*.jpg')
        )

        # sci-art use .JPG
        JPGs = glob.glob(
            os.path.join(self.train_rgbs_dir, '*.JPG')
        )

        self.train_images = jpgs + JPGs


    def __len__(self):
        return len(self.train_images)

    def __getitem__(self, index):
        img_path = self.train_images[index]
        img_base_name = os.path.basename(img_path).split('.')[0]
        i = int(img_base_name)

        meta_data = torch.load(
            os.path.join(self.train_meta_dir, '{:06d}.pt'.format(i))
        )
        
        c2w = torch.zeros(3, 4)
        c2w[:, 1:2] = meta_data['c2w'][:, 0:1]
        c2w[:, :1] = -meta_data['c2w'][:, 1:2]
        c2w[:, 2:4] = meta_data['c2w'][:, 2:4]
        assert np.logical_and(c2w >= -1, c2w <= 1).all()
        c2w[:, 3] = c2w[:, 3] * self.pose_scale_factor + self.origin_drb

        c2w_colmap = torch.hstack((
            DRB_TO_RDF @ c2w[:3, :3] @ DRB_TO_RDF.T,
            DRB_TO_RDF @ c2w[:3, 3:]
        ))
        c2w_full = torch.eye(4)
        c2w_full[:3, :] = c2w_colmap
        w2c_full = torch.inverse(c2w_full)
        image_tvec = w2c_full[:3, 3].numpy().tolist()
        q_xyzw = Rotation.from_matrix(w2c_full[:3, :3]).as_quat()
        image_qvec = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]

        H, W = meta_data['H'], meta_data['W']
        camera_matrix = np.zeros((3,3))

        camera_params = np.zeros(4)
        camera_params[3] = meta_data['distortion'][0]

        camera_matrix[0][0] = meta_data['intrinsics'][0]
        camera_matrix[1][1] = meta_data['intrinsics'][1]
        camera_matrix[0][2] = meta_data['intrinsics'][2]
        camera_matrix[1][2] = meta_data['intrinsics'][3]
        
        camera_params[0] = camera_matrix[0][0]
        camera_params[1] = camera_matrix[0][2]
        camera_params[2] = camera_matrix[1][2]

        downsample_scale = self.downsample_scale
        # downsampling:
        if downsample_scale > 0:
            rgbs_img = cv2.imread(img_path)
            img_small = cv2.resize(rgbs_img, None, fx=downsample_scale, fy=downsample_scale, interpolation=cv2.INTER_LINEAR)
            H, W = img_small.shape[0], img_small.shape[1]
            camera_params *= downsample_scale
            cv2.imwrite(
                os.path.join(out_img_path, '{:06d}.jpg'.format(i)),
                img_small
            )

        img_info = Image(id=i, camera_id=i, 
                  qvec=image_qvec, tvec=image_tvec, 
                  name='{:06d}.jpg'.format(i), 
                  xys=[], point3D_ids=[])
        # I think images in /rgbs are already undistorted
        # thus cameras are just SIMPLE_PINHOLE models
      
        camera_info = Camera(id=i, model='PINHOLE', 
                   width=W, height=H, params=[camera_params[0], camera_params[0], camera_params[1], camera_params[2]])
        
        return img_info, camera_info


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
        

def read_mega_nerf_dataset(path, out_img_path, downsample_scale):
    coordinates_path = os.path.join(path, 'coordinates.pt')
    train_meta_dir = os.path.join(path, 'train/metadata')
    train_rgbs_dir = os.path.join(path, 'train/rgbs')

    info = torch.load(coordinates_path)
    origin_drb, pose_scale_factor = info['origin_drb'], info['pose_scale_factor']

    train_images = glob.glob(
        os.path.join(train_rgbs_dir, '*.jpg')
    )

    cameras, images = [], []
    for img_path in tqdm(train_images):
        img_base_name = os.path.basename(img_path).split('.')[0]
        i = int(img_base_name)

        meta_data = torch.load(
            os.path.join(train_meta_dir, '{:06d}.pt'.format(i))
        )
        
        c2w = torch.zeros(3, 4)
        c2w[:, 1:2] = meta_data['c2w'][:, 0:1]
        c2w[:, :1] = -meta_data['c2w'][:, 1:2]
        c2w[:, 2:4] = meta_data['c2w'][:, 2:4]
        assert np.logical_and(c2w >= -1, c2w <= 1).all()
        c2w[:, 3] = c2w[:, 3] * pose_scale_factor + origin_drb

        c2w_colmap = torch.hstack((
            DRB_TO_RDF @ c2w[:3, :3] @ DRB_TO_RDF.T,
            DRB_TO_RDF @ c2w[:3, 3:]
        ))
        c2w_full = torch.eye(4)
        c2w_full[:3, :] = c2w_colmap
        w2c_full = torch.inverse(c2w_full)
        image_tvec = w2c_full[:3, 3]
        q_xyzw = Rotation.from_matrix(w2c_full[:3, :3]).as_quat()
        image_qvec = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]

        H, W = meta_data['H'], meta_data['W']
        camera_matrix = np.zeros((3,3))

        camera_params = np.zeros(4)
        camera_params[3] = meta_data['distortion'][0]

        camera_matrix[0][0] = meta_data['intrinsics'][0]
        camera_matrix[1][1] = meta_data['intrinsics'][1]
        camera_matrix[0][2] = meta_data['intrinsics'][2]
        camera_matrix[1][2] = meta_data['intrinsics'][3]
        
        camera_params[0] = camera_matrix[0][0]
        camera_params[1] = camera_matrix[0][2]
        camera_params[2] = camera_matrix[1][2]

        # downsampling:
        if downsample_scale > 0:
            rgbs_img = cv2.imread(img_path)
            img_small = cv2.resize(rgbs_img, None, fx=downsample_scale, fy=downsample_scale, interpolation=cv2.INTER_LINEAR)
            H, W = img_small.shape[0], img_small.shape[1]
            camera_params *= downsample_scale
            # cv2.imwrite(
            #     os.path.join(out_img_path, '{:06d}.jpg'.format(i)),
            #     img_small
            # )

        images.append(
            Image(id=i, camera_id=i, 
                  qvec=image_qvec, tvec=image_tvec, 
                  name='{:06d}.jpg'.format(i), 
                  xys=[], point3D_ids=[])
        )
        # I think images in /rgbs are already undistorted
        # thus cameras are just SIMPLE_PINHOLE models
        cameras.append(
            Camera(id=i, model='SIMPLE_PINHOLE', 
                   width=W, height=H, params=camera_params[:3])
        )

        c2w_mega, intrinsics_mega = colmap_2_mega(images[-1], cameras[-1], origin_drb, pose_scale_factor)
        pass

    return cameras, images


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
    path = '/jfs/shengyi.chen/HT/Data/Mill_19/OpenDataLab___Mill_19/raw/Mill_19/building-pixsfm'
    downsample_scale = 0.25
    normalized = True

    # process train_set
    out_path = os.path.join('/jfs/shengyi.chen/HT/Data/Mill_19/OpenDataLab___Mill_19/colmap_norm/Mill_19/building-pixsfm')
    out_path_train = out_path
    out_model_path = os.path.join(out_path, 'sparse')
    out_img_path = os.path.join(out_path, 'images')

    os.makedirs(out_path, exist_ok=True)
    os.makedirs(out_model_path, exist_ok=True)
    os.makedirs(out_img_path, exist_ok=True)

    ## dataloader version
    def get_batch(cameras):
        return list(cameras)
    
    dataset = TransData(path, out_img_path, downsample_scale, split='train', normalized=normalized)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=32, 
                            collate_fn=get_batch)
    cameras, images = [], []
    for _i, data in tqdm(enumerate(dataloader)):
        img_info, camera_info = data[0]
        images.append(img_info)
        cameras.append(camera_info)
    
    write_cameras_text(cameras=cameras, path=os.path.join(out_model_path, 'cameras.txt'))
    write_images_text(images=images, path=os.path.join(out_model_path, 'images.txt'))
    with open(os.path.join(out_model_path, 'points3D.txt'), 'w') as f:
        pass

    # process test_set
    out_path = os.path.join(out_path_train, 'test')
    out_model_path = os.path.join(out_path, 'sparse')
    out_img_path = os.path.join(out_path, 'images')

    os.makedirs(out_path, exist_ok=True)
    os.makedirs(out_model_path, exist_ok=True)
    os.makedirs(out_img_path, exist_ok=True)
    
    dataset = TransData(path, out_img_path, downsample_scale, split='val', normalized=normalized)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=32, 
                            collate_fn=get_batch)
    cameras, images = [], []
    for _i, data in tqdm(enumerate(dataloader)):
        img_info, camera_info = data[0]
        images.append(img_info)
        cameras.append(camera_info)
    
    write_cameras_text(cameras=cameras, path=os.path.join(out_model_path, 'cameras.txt'))
    write_images_text(images=images, path=os.path.join(out_model_path, 'images.txt'))
    with open(os.path.join(out_model_path, 'points3D.txt'), 'w') as f:
        pass