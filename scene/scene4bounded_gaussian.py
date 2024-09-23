import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks, readColmapSceneAndEmptyCameraInfo, readColmapOnlyEmptyCameraInfo, readNerfSyntheticAndEmptyCameraInfo, storePly, fetchPly
from scene.dataset_readers import readCustomMill19CameraInfo, readCustomScanNetCameraInfo
from scene.dataset_readers import SceneInfo, CameraInfo
from scene.gaussian_nn_module import GaussianModel2, BoundedGaussianModel, BoundedGaussianModelGroup
from utils.graphics_utils import BasicPointCloud
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON, loadCam, loadEmptyCam
from scene.dataset_readers import CameraInfo
from scene.cameras import Camera
from utils.datasets import CameraListDataset
import numpy as np


class SceneV3:
    gaussians_group : BoundedGaussianModelGroup 
    def __init__(self, args : ModelParams, gaussians_group : BoundedGaussianModelGroup, shuffle=False, load_iteration=None, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians_group = gaussians_group
        try:
            self.padding_width = args.padding_width
        except:
            self.padding_width = 0    

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            if os.path.exists(os.path.join(args.source_path, "train_test_lists.json")):
                print('find train_test_lists.json, assuming ScanNet++ data set!')
                scene_info = readCustomScanNetCameraInfo(args.source_path, pointcloud_sample_rate=args.pointcloud_sample_rate, points3D=args.points3D)  
            elif not os.path.exists(os.path.join(args.source_path, "test")):
                scene_info = readColmapSceneAndEmptyCameraInfo(args.source_path, args.images, args.eval, points3D=args.points3D)
            else:
                print('find test/ folder, assuming Mill_19 data set!')
                scene_info = readCustomMill19CameraInfo(args.source_path, points3D=args.points3D)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = readNerfSyntheticAndEmptyCameraInfo(args.source_path, args.white_background, args.eval, points3D=args.points3D)
        else:
            assert False, "Could not recognize scene type!"

        self.scene_info:SceneInfo = scene_info

        if not self.loaded_iter:
            # with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
            #     dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        self.point_cloud = scene_info.point_cloud
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = CameraListDataset(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = CameraListDataset(scene_info.test_cameras, resolution_scale, args)

        if self.gaussians_group is not None:
            if self.loaded_iter:
                for i in range(self.gaussians_group.group_size):
                    self.gaussians_group.all_gaussians[i].load_ply(os.path.join(self.model_path, 
                                                                                "point_cloud",
                                                                                "iteration_" + str(self.loaded_iter),
                                                                                "point_cloud_{}.ply".format(i)))
            else:
                for i in self.gaussians_group.all_gaussians:
                    _gau = self.gaussians_group.all_gaussians[i]
                    self.loadPointCloud2Gaussians(
                        _gau,
                        range_low=_gau.range_low.cpu().detach().numpy(),
                        range_up=_gau.range_up.cpu().detach().numpy(),
                        padding_width=self.gaussians_group.padding_width
                    )

        # self.save_img_path = os.path.join(self.model_path, 'img')
        self.save_img_path = os.path.join(os.path.dirname(self.model_path), 'img')
        # self.save_depth_path = os.path.join(self.model_path, 'depth')
        # self.save_depth_path = os.path.join(os.path.dirname(self.model_path), 'depth')
        # self.save_gt_path = os.path.join(self.model_path, 'gt')
        self.save_gt_path = os.path.join(os.path.dirname(self.model_path), 'gt')
        os.makedirs(self.save_img_path, exist_ok=True)
        # os.makedirs(self.save_depth_path, exist_ok=True)
        os.makedirs(self.save_gt_path, exist_ok=True)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def loadPointCloud2Gaussians(self, gau: GaussianModel2, range_low, range_up, padding_width=None):
        if padding_width is None:
            padding_width = self.padding_width 

        padding_width = max(padding_width, 0)
        range_low = [e - padding_width for e in range_low]
        range_up = [e + padding_width for e in range_up]

        valid_low = np.logical_and.reduce(self.point_cloud.points >= range_low, axis=-1) 
        valid_up = np.logical_and.reduce(self.point_cloud.points <= range_up, axis=-1)
        valid = np.logical_and(valid_low, valid_up)

        pcd = BasicPointCloud(
            points=self.point_cloud.points[valid],
            colors=self.point_cloud.colors[valid],
            normals=self.point_cloud.normals[valid]
        )
        gau.create_from_pcd(pcd, spatial_lr_scale=self.cameras_extent)
        return valid.sum()

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        for model_id in self.gaussians_group.all_gaussians: 
            self.gaussians_group.all_gaussians[model_id].save_ply(os.path.join(point_cloud_path, "point_cloud_{}.ply".format(model_id)))

    @staticmethod
    def get_batch(cameras:Camera):
        return list(cameras)   
