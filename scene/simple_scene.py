import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks, readColmapSceneAndEmptyCameraInfo, readColmapOnlyEmptyCameraInfo, readNerfSyntheticAndEmptyCameraInfo, storePly, fetchPly
from scene.dataset_readers import readCustomMill19CameraInfo
from scene.dataset_readers import SceneInfo, CameraInfo
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.datasets import CameraListDataset

class SimpleScene:
    def __init__(self, args : ModelParams, load_iteration=None, resolution_scales=[1.0]):
        """
        Compared with Scene, a SimpleScene only collects basic training info
        """
        self.model_path = args.model_path
        self.scale_control_rate = args.scale_control_rate
        self.pointcloud_sample_rate = args.pointcloud_sample_rate
        self.opacity_init = args.opacity_init
        self.loaded_iter = None

        if load_iteration:
            if load_iteration == -1:
                try:
                    self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
                except:
                    self.loaded_iter = None
            else:
                self.loaded_iter = load_iteration
            print("set trained model iteration as {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):

            if not os.path.exists(os.path.join(args.source_path, "test")):
                scene_info = readColmapSceneAndEmptyCameraInfo(args.source_path, args.images, args.eval, pointcloud_sample_rate=args.pointcloud_sample_rate, points3D=args.points3D)
            else:
                print('find test/ folder, assuming Mill_19 data set!')
                scene_info = readCustomMill19CameraInfo(args.source_path, pointcloud_sample_rate=args.pointcloud_sample_rate, points3D=args.points3D)

        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = readNerfSyntheticAndEmptyCameraInfo(args.source_path, args.white_background, args.eval, args.pointcloud_sample_rate)
        else:
            assert False, "Could not recognize scene type!"

        self.scene_info:SceneInfo = scene_info

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
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

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = CameraListDataset(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = CameraListDataset(scene_info.test_cameras, resolution_scale, args)

    def save(self, iteration, gaussians):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def load2gaussians(self, gaussians:GaussianModel):
        if self.loaded_iter:
            gaussians.load_ply_own(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            gaussians.create_from_pcd(self.scene_info.point_cloud, self.cameras_extent, self.scale_control_rate, self.opacity_init)
    
    @staticmethod
    def get_batch(cameras):
        return list(cameras)
    