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

import torch, math
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from gaussian_renderer.render import render4GaussianModel2
from gaussian_renderer.render_half_gs import render4BoundedGaussianModel
from gaussian_renderer.render_metric import render_with_GradNormHelper as render_metric
import gaussian_renderer.pytorch_gs_render.pytorch_render as pytorch_render
from scene.cameras import ViewMessage, Camera, Patch


class GaussianModel2(nn.Module):
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree : int, range_low=[0,0,0], range_up=[0,0,0], device="cuda"):
        super(GaussianModel2, self).__init__()

        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._means2D_meta = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer:torch.optim.Adam = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        self.device = torch.device(device)
        self.range_low = torch.tensor(range_low, dtype=torch.float32).to(self.device)
        self.range_up = torch.tensor(range_up, dtype=torch.float32).to(self.device)

        self._all_features = None 
        self._xyz_home = None

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._means2D_meta,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self._means2D_meta,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_xyz_hom(self):
        if self._xyz_home is not None:
            return self._xyz_home
        else:
            self._xyz_home = torch.cat([self.get_xyz, torch.ones_like(self._opacity)], dim=-1)
            return self._xyz_home 

    @property
    def get_features(self):
        if self._all_features is not None:
            return self._all_features
        else:
            features_dc = self._features_dc
            features_rest = self._features_rest
            self._all_features = torch.cat((features_dc, features_rest), dim=1)
            return self._all_features

    def clean_cached_features(self):
        self._all_features = None 
        self._xyz_home = None   
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().to(self.device)
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().to(self.device))
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().to(self.device)
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().to(self.device)), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device=self.device)
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device=self.device))

        self._xyz = nn.Parameter(fused_point_cloud.contiguous().requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.contiguous().requires_grad_(True))
        self._rotation = nn.Parameter(rots.contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(opacities.contiguous().requires_grad_(True))
        self._means2D_meta = nn.Parameter(torch.zeros_like(opacities).contiguous().requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self.device)

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._means2D_meta], 'lr': 0, "name": "means2D_meta"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.scaling_scheduler_args = get_expon_lr_func(lr_init=training_args.scaling_lr_init,
                                                    lr_final=training_args.scaling_lr_final,
                                                    lr_delay_mult=training_args.scaling_lr_delay_mult,
                                                    max_steps=training_args.scaling_lr_max_steps)

    def update_learning_rate(self, iteration):
        ret = -1.0
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                ret = lr # return lr
                break
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "scaling":
                lr = self.scaling_scheduler_args(iteration)
                param_group['lr'] = lr    
                ret = lr # return lr
                break      

        return ret    
            
    def update_learning_rate_scaling(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "scaling":
                lr = self.scaling_scheduler_args(iteration)
                param_group['lr'] = lr    
                return lr        

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        means2D_meta = None # meaningless to save means2D_meta
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        # elements = np.empty(xyz.shape[0], dtype=dtype_full)
        # elements[:] = list(map(tuple, attributes)) # cost too many memory
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1).conjugate()
        elements = np.frombuffer(attributes.tobytes(), dtype=dtype_full)
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
        # self._means2D_meta *= 0

    def load_ply(self, path, ply_data_in_memory:PlyData=None):
        if ply_data_in_memory is None:
            plydata = PlyData.read(path)
        else:
            plydata = ply_data_in_memory

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        print("Number of points at initialisation : ", xyz.shape[0])
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device=self.device).contiguous().requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device=self.device).transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device=self.device).transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device=self.device).contiguous().requires_grad_(True))
        self._means2D_meta = nn.Parameter(torch.zeros(opacities.shape, dtype=torch.float, device=self.device).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device=self.device).contiguous().requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device=self.device).contiguous().requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self.device)

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"].contiguous()
        self._features_dc = optimizable_tensors["f_dc"].contiguous()
        self._features_rest = optimizable_tensors["f_rest"].contiguous()
        self._opacity = optimizable_tensors["opacity"].contiguous()
        self._means2D_meta = optimizable_tensors["means2D_meta"].contiguous()
        self._scaling = optimizable_tensors["scaling"].contiguous()
        self._rotation = optimizable_tensors["rotation"].contiguous()

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.clean_cached_features()

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        if len(new_xyz) > 0:
            d = {"xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "means2D_meta": torch.zeros_like(new_opacities),
            "scaling" : new_scaling,
            "rotation" : new_rotation}

            optimizable_tensors = self.cat_tensors_to_optimizer(d)
            self._xyz = optimizable_tensors["xyz"].contiguous()
            self._features_dc = optimizable_tensors["f_dc"].contiguous()
            self._features_rest = optimizable_tensors["f_rest"].contiguous()
            self._opacity = optimizable_tensors["opacity"].contiguous()
            self._means2D_meta = optimizable_tensors["means2D_meta"].contiguous()
            self._scaling = optimizable_tensors["scaling"].contiguous()
            self._rotation = optimizable_tensors["rotation"].contiguous()

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self.device)
        self.clean_cached_features()    # discard cached feature 

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device=self.device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = padded_grad >= grad_threshold
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        if selected_pts_mask.long().sum() < 1:
            self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
            self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self.device)
            prune_filter = selected_pts_mask
            self.prune_points(prune_filter)
            return

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cpu")
        samples = torch.normal(mean=means.cpu(), std=stds.cpu()).to(self.device)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.long().sum(), device=self.device, dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = grads.squeeze() >= grad_threshold
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = prune_mask | big_points_vs | big_points_ws
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def __add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.abs(self._means2D_meta.grad[update_filter])
        self.denom[update_filter] += 1    

    def add_densification_stats_ddp(self, viewspace_point_tensor, visibility_filter_total, update_cnt):
        # grad is already average/summed, so we just add it
        # denom should be updated with cnt as one gaussian can be visbiable for multiple views
        update_filter = visibility_filter_total
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += update_cnt[update_filter]

    def clone_from_optimizer_dict(self, tensor_dict:dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            new_group = tensor_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            stored_state["exp_avg"] = new_group['exp_avg']
            stored_state["exp_avg_sq"] = new_group['exp_avg_sq']

            del self.optimizer.state[group['params'][0]]
            group["params"][0] = nn.Parameter(new_group['tensor'].requires_grad_(True))
            self.optimizer.state[group['params'][0]] = stored_state

            optimizable_tensors[group["name"]] = group["params"][0]

        self._xyz = optimizable_tensors["xyz"].contiguous()
        self._features_dc = optimizable_tensors["f_dc"].contiguous()
        self._features_rest = optimizable_tensors["f_rest"].contiguous()
        self._opacity = optimizable_tensors["opacity"].contiguous()
        self._means2D_meta = optimizable_tensors["means2D_meta"].contiguous()
        self._scaling = optimizable_tensors["scaling"].contiguous()
        self._rotation = optimizable_tensors["rotation"].contiguous()

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self.device)

        return None

    def forward(self, viewpoint_cam, pipe, background):
        return render4GaussianModel2(viewpoint_cam, self, pipe, background)            

def create_batch(patches:list):
    # assume all patch has the same complete shape
    batch_data, batch_size = {}, len(patches)
    example_p:Patch = patches[0]
    H, W = example_p.complete_height, example_p.complete_width

    batch_data['packed_views'] = torch.stack([p.pack_up() for p in patches])
    max_tile_num = max([len(p.all_tiles) for p in patches]) 
    batch_data['tile_maps'] = torch.zeros((batch_size, max_tile_num), dtype=torch.int, device='cuda')
    batch_data['tile_nums'] = []
    for i in range(batch_size):
        p:Patch = patches[i]
        num_valid_tile = len(p.all_tiles)
        batch_data['tile_maps'][i, :num_valid_tile] = p.all_tiles
        batch_data['tile_nums'].append(num_valid_tile)
    batch_data['tile_maps_sizes'] = torch.tensor([math.ceil(W/16), math.ceil(H/16)], device='cuda').view(-1, 2).repeat(batch_size, 1)    
    batch_data['tile_maps_sizes_list'] = [math.ceil(W/16), math.ceil(H/16)]
    batch_data['image_size_list'] = [math.ceil(W), math.ceil(H)]
    batch_data['patches_cpu'] = patches  # it's okay to use patch on gpu 
    return batch_data

class BoundedGaussianModel(GaussianModel2):
    def __init__(self, sh_degree: int, range_low=[0, 0, 0], range_up=[0, 0, 0], device="cuda", max_size:int=None):
        super().__init__(sh_degree, range_low, range_up, device)
        self.max_size = max_size

    def forward(self, viewpoint_cam, pipe, background, need_buffer:bool=False):
        
        if pipe.render_version == 'default':
            all_ret = render4BoundedGaussianModel(viewpoint_cam, self, pipe, background)   
            if need_buffer:
                return all_ret
            else:
                return {
                    "render": all_ret["render"],
                    "viewspace_points": all_ret["viewspace_points"],
                    "visibility_filter": all_ret["visibility_filter"],
                    "radii": all_ret["radii"],
                    "depth": all_ret["depth"],
                    "alpha": all_ret["alpha"],
                    "num_rendered": all_ret["num_rendered"],
                }
        elif pipe.render_version == '3d_gs':
            # we need alpha map, thus use render metric 
            ret = render_metric(viewpoint_cam, self, pipe, background)    
            return ret
        elif pipe.render_version == 'pytorch_render':
            # this is not a goot way to call pytorch_render_raster
            # but it is simple and suitable for testing
            assert isinstance(viewpoint_cam, Camera)
            patch = Patch(viewpoint_cam, uid=0, 
                          v_start=0, u_start=0,
                          v_end=viewpoint_cam.image_height, u_end=viewpoint_cam.image_width)
            batch = create_batch([patch])
            return pytorch_render.render(self, batch_data=batch)[0]
        else:
            raise NotImplementedError('not render of version {}'.format(pipe.render_version))
    
    def add_densification_stats(self, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.abs(self._means2D_meta.grad[update_filter])
        self.denom[update_filter] += 1

    def pack_up(self, scale_factor=1):
        with torch.no_grad():
            P = self._xyz.shape[0]
            name2torchPara = {}
            for group in self.optimizer.param_groups: 
                name2torchPara[group['name']] = group["params"][0]

            para = torch.cat(
                [self._xyz.view((P, -1)), 
                 self._features_dc.view((P, -1)), 
                 self._features_rest.view((P, -1)), 
                 self._scaling.view((P, -1)),
                 self._rotation.view((P, -1)),
                 self._opacity.view((P, -1))
                 ], 
                dim=-1
            )
            exp_avg = torch.cat(
                [self.optimizer.state[name2torchPara['xyz']]['exp_avg'].view((P, -1)), 
                 self.optimizer.state[name2torchPara['f_dc']]['exp_avg'].view((P, -1)), 
                 self.optimizer.state[name2torchPara['f_rest']]['exp_avg'].view((P, -1)), 
                 self.optimizer.state[name2torchPara['scaling']]['exp_avg'].view((P, -1)),
                 self.optimizer.state[name2torchPara['rotation']]['exp_avg'].view((P, -1)),
                 self.optimizer.state[name2torchPara['opacity']]['exp_avg'].view((P, -1))
                 ], 
                dim=-1
            )
            exp_avg_sq = torch.cat(
                [self.optimizer.state[name2torchPara['xyz']]['exp_avg_sq'].view((P, -1)), 
                 self.optimizer.state[name2torchPara['f_dc']]['exp_avg_sq'].view((P, -1)), 
                 self.optimizer.state[name2torchPara['f_rest']]['exp_avg_sq'].view((P, -1)), 
                 self.optimizer.state[name2torchPara['scaling']]['exp_avg_sq'].view((P, -1)),
                 self.optimizer.state[name2torchPara['rotation']]['exp_avg_sq'].view((P, -1)),
                 self.optimizer.state[name2torchPara['opacity']]['exp_avg_sq'].view((P, -1))
                 ], 
                dim=-1
            )

            ret = torch.cat([para, exp_avg, exp_avg_sq], dim=-1)

        return ret.to(torch.float).contiguous(), scale_factor*self.get_scaling
    
    def simple_pack_up(self):
        with torch.no_grad():
            P = self._xyz.shape[0]
            rest_channel = (self.max_sh_degree + 1) ** 2 - 1
            all_channel = 14 + 3*rest_channel
            ret = torch.zeros((P, 59), dtype=torch.float, device='cuda')
            ret[:, 0:3] = self._xyz.view((P, -1))
            ret[:, 3:6] = self._features_dc.view((P, -1))
            ret[:, 6:(6 + 3*rest_channel)] = self._features_rest.view((P, -1))
            ret[:, (6 + 3*rest_channel):(6 + 3*rest_channel + 3)] = self._scaling.view((P, -1))
            ret[:, (9 + 3*rest_channel):(13 + 3*rest_channel)] = self._rotation.view((P, -1))
            ret[:, (13 + 3*rest_channel):(14 + 3*rest_channel)] = self._opacity.view((P, -1))

        return ret.to(torch.float).contiguous(), self.get_scaling

    def space_for_pkg(self, size:int):
        rest_channel = (self.max_sh_degree + 1) ** 2 - 1
        all_channel = 14 + 3*rest_channel
        return torch.zeros((size, all_channel*3), dtype=torch.float, device='cuda')

    def un_pack_up(self, pkg:torch.Tensor, spatial_lr_scale:float, iteration:int, step:int, opt):
        self.spatial_lr_scale = spatial_lr_scale 
        pkg = pkg.to('cuda')
        rest_channel = (self.max_sh_degree + 1) ** 2 - 1
        all_channel = 14 + 3*rest_channel
        P = pkg.shape[0]    

        msg = pkg[:, 0:all_channel]
        _xyz = msg[:, 0:3].view((P, 3)).contiguous()
        _features_dc = msg[:, 3:6].view((P, 1, 3)).contiguous()
        _features_rest = msg[:, 6:(6 + 3*rest_channel)].view((P, rest_channel, 3)).contiguous()
        _scaling = msg[:, (6 + 3*rest_channel):(6 + 3*rest_channel + 3)].view((P, 3)).contiguous()
        _rotation = msg[:, (9 + 3*rest_channel):(13 + 3*rest_channel)].view((P, 4)).contiguous()
        _opacity = msg[:, (13 + 3*rest_channel):(14 + 3*rest_channel)].view((P, 1)).contiguous()

        self._xyz = nn.Parameter(_xyz.requires_grad_(True))
        self._features_dc = nn.Parameter(_features_dc.requires_grad_(True))
        self._features_rest = nn.Parameter(_features_rest.requires_grad_(True))
        self._scaling = nn.Parameter(_scaling.requires_grad_(True))
        self._rotation = nn.Parameter(_rotation.requires_grad_(True))
        self._opacity = nn.Parameter(_opacity.requires_grad_(True))
        self._means2D_meta = nn.Parameter(torch.zeros_like(_opacity).contiguous().requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self.device)

        self.training_setup(training_args=opt)
        name2torchPara = {}
        for group in self.optimizer.param_groups: 
            name2torchPara[group['name']] = group["params"][0]

        avg = pkg[:, all_channel:2*all_channel]
        avg_sq = pkg[:, 2*all_channel:3*all_channel]
        names = ['xyz', 'f_dc', 'f_rest', 'scaling', 'rotation', 'opacity']
        idx = [0, 3, 6, (6 + 3*rest_channel), (9 + 3*rest_channel), (13 + 3*rest_channel), (14 + 3*rest_channel)] 
        view_shapes = [(P, 3), (P, 1, 3), (P, rest_channel, 3), (P, 3), (P, 4), (P, 1)]

        for _i in range(len(names)):
            name, idx0, idx1, shape = names[_i], idx[_i], idx[_i+1], view_shapes[_i]
            self.optimizer.state[name2torchPara[name]]['exp_avg'] = avg[:, idx0:idx1].view(shape).contiguous()
            self.optimizer.state[name2torchPara[name]]['exp_avg_sq'] = avg_sq[:, idx0:idx1].view(shape).contiguous()
            self.optimizer.state[name2torchPara[name]]['step'] = torch.tensor(step, dtype=torch.float)
        
        self.update_learning_rate(iteration)
        self.set_SHdegree(iteration//1000)
        
    def simple_un_pack_up(self, pkg:torch.Tensor, spatial_lr_scale:float, iteration:int, step:int, opt):
        self.spatial_lr_scale = spatial_lr_scale 
        pkg = pkg.to('cuda')
        rest_channel = (self.max_sh_degree + 1) ** 2 - 1
        all_channel = 14 + 3*rest_channel
        P = pkg.shape[0]    

        msg = pkg[:, 0:all_channel]
        _xyz = msg[:, 0:3].view((P, 3)).contiguous()
        _features_dc = msg[:, 3:6].view((P, 1, 3)).contiguous()
        _features_rest = msg[:, 6:(6 + 3*rest_channel)].view((P, rest_channel, 3)).contiguous()
        _scaling = msg[:, (6 + 3*rest_channel):(6 + 3*rest_channel + 3)].view((P, 3)).contiguous()
        _rotation = msg[:, (9 + 3*rest_channel):(13 + 3*rest_channel)].view((P, 4)).contiguous()
        _opacity = msg[:, (13 + 3*rest_channel):(14 + 3*rest_channel)].view((P, 1)).contiguous()

        self._xyz = nn.Parameter(_xyz.requires_grad_(True))
        self._features_dc = nn.Parameter(_features_dc.requires_grad_(True))
        self._features_rest = nn.Parameter(_features_rest.requires_grad_(True))
        self._scaling = nn.Parameter(_scaling.requires_grad_(True))
        self._rotation = nn.Parameter(_rotation.requires_grad_(True))
        self._opacity = nn.Parameter(_opacity.requires_grad_(True))
        self._means2D_meta = nn.Parameter(torch.zeros_like(_opacity).contiguous().requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self.device)

        self.training_setup(training_args=opt)    

        name2torchPara = {}
        for group in self.optimizer.param_groups: 
            name2torchPara[group['name']] = group["params"][0]

        # avg = pkg[:, all_channel:2*all_channel]
        # avg_sq = pkg[:, 2*all_channel:3*all_channel]
        # names = ['xyz', 'f_dc', 'f_rest', 'scaling', 'rotation', 'opacity']
        # idx = [0, 3, 6, (6 + 3*rest_channel), (9 + 3*rest_channel), (13 + 3*rest_channel), (14 + 3*rest_channel)] 
        # view_shapes = [(P, 3), (P, 1, 3), (P, rest_channel, 3), (P, 3), (P, 4), (P, 1)]

        # for _i in range(len(names)):
        #     name, idx0, idx1, shape = names[_i], idx[_i], idx[_i+1], view_shapes[_i]
        #     self.optimizer.state[name2torchPara[name]]['exp_avg'] = avg[:, idx0:idx1].view(shape).contiguous()
        #     self.optimizer.state[name2torchPara[name]]['exp_avg_sq'] = avg_sq[:, idx0:idx1].view(shape).contiguous()
        #     self.optimizer.state[name2torchPara[name]]['step'] = torch.tensor(step, dtype=torch.float)

        self.update_learning_rate(iteration)
        self.set_SHdegree(iteration//1000)

    def set_SHdegree(self, value:int):
        v = int(value)
        if v < 0:
            self.active_sh_degree = 0
        elif v >= self.max_sh_degree:  
            self.active_sh_degree = self.max_sh_degree
        else:
            self.active_sh_degree = v

    def get_info(self):
        model_info =  f"BoundedGaussianModel(sh_degree:{self.max_sh_degree}, range_low={self.range_low}, range_up={self.range_up}, max_size={self.max_size})"
        opti_info = ''
        state_info = ''
        if self.optimizer is not None:
            for group in self.optimizer.param_groups:
                opti_info += '(name:lr):({}:{:.5e})'.format(group['name'], group['lr'])
            state_info = 'len state: {}'.format(len(self.optimizer.state))
        return model_info + '\n' + opti_info + '\n' + state_info

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, skip_prune=False, skip_clone=False, skip_split=False):
        if (self.max_size is not None) and self._xyz.shape[0] > self.max_size:
            print('current shape {} is larger than {}, skip densify'.format(self._xyz.shape[0], self.max_size))
        else:
            grads = self.xyz_gradient_accum / self.denom
            grads[grads.isnan()] = 0.0
            if not skip_clone:
                self.densify_and_clone(grads, max_grad, extent)
            if not skip_split:    
                self.densify_and_split(grads, max_grad, extent)
            # add extra postfix in case of skip_clone=skip_split=True
            if skip_split and skip_clone:
                self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
                self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
                self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self.device)   

        if not skip_prune:    
            prune_mask = (self.get_opacity < min_opacity).squeeze()
            if max_screen_size:
                big_points_vs = self.max_radii2D > max_screen_size
                big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
                prune_mask = prune_mask | big_points_vs | big_points_ws
            self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def discard_gs_out_range(self, scale_factor=1):
        scale = scale_factor*self.get_scaling
        max_radii, _idx = torch.max(scale, dim=-1, keepdim=True)
        max_radii = (max_radii*3).clamp(min=0)
        flag1 = (self.get_xyz + max_radii) >= self.range_low
        flag2 = (self.get_xyz - max_radii) <= self.range_up
        _flag = torch.logical_and(flag1, flag2)
        flag = torch.all(_flag, dim=-1, keepdim=False)
        self.prune_points(mask=~flag)


class BoundedGaussianModelGroup(nn.Module):
    def __init__(
            self, 
            sh_degree_list, 
            range_low_list, 
            range_up_list, 
            device_list,
            model_id_list,
            padding_width = 1,
            max_size:int=None,
            ) -> None:
        super().__init__()
        assert len(sh_degree_list) == len(range_low_list) == len(range_up_list) == len(device_list) == len(model_id_list)

        # self.all_gaussians = torch.nn.ModuleList()
        self.all_gaussians = torch.nn.ModuleDict()
        self.range_low_list = range_low_list
        self.range_up_list = range_up_list
        self.group_size = len(device_list)
        self.device_list = device_list
        self.model_id_list = model_id_list
        # self._id_2_model = {id: None for id in model_id_list}
        self.padding_width = padding_width

        self.pack_up_channel = 59 # 59*3
        self.max_size = max_size

        for i in range(len(model_id_list)):
            # self._id_2_model[model_id_list[i]] = i
            model_id:int = model_id_list[i]
            gau = BoundedGaussianModel(sh_degree_list[i], range_low_list[i], range_up_list[i], device_list[i], max_size=max_size)
            self.all_gaussians[str(model_id)] = gau

    def pop_model(self, uid):
        uid = str(uid)
        if uid not in self.all_gaussians:
            return None 
        return self.all_gaussians.pop(uid)
    
    def get_model(self, uid):
        # if id not in self._id_2_model:
        #     return None
        # int_id = self._id_2_model[id]
        # return self.all_gaussians[int_id]   
        uid = str(uid) 
        if uid not in self.all_gaussians:
            return None 
        return self.all_gaussians[uid]   

    def training_setup(self, training_args):    
        for uid in self.all_gaussians:
            gau = self.all_gaussians[uid]
            gau.training_setup(training_args) 

    def update_learning_rate(self, iteration):    
        for uid in self.all_gaussians:
            gau = self.all_gaussians[uid]
            gau.update_learning_rate(iteration) 

    def clean_cached_features(self):    
        for uid in self.all_gaussians:
            gau = self.all_gaussians[uid]
            gau.clean_cached_features()        

    def oneupSHdegree(self):    
        for uid in self.all_gaussians:
            gau = self.all_gaussians[uid]
            gau.oneupSHdegree() 

    def reset_opacity(self):
        for uid in self.all_gaussians:
            gau = self.all_gaussians[uid]
            gau.reset_opacity()

    def set_SHdegree(self, value:int):
        for uid in self.all_gaussians:
            gau = self.all_gaussians[uid]
            gau.set_SHdegree(value) 

    def get_info(self):
        ret = []
        for k in self.all_gaussians:
            ret.append(f"{k}:{self.all_gaussians[k].get_info()}")
        return '\n'.join(ret)

    @staticmethod
    def pack_up_render_ret(render_ret):
        '''
        pack up color, radii, alpha into one tensor
        viewspace_points/visibility_filter/radii is only used in management of gaussian points
        '''    
        with torch.no_grad():
            # (5, H, W) tensor
            pkg = torch.cat(
                [render_ret["render"], render_ret["depth"], render_ret["alpha"]], 
                dim=0
            ).to(dtype=torch.float32) 
        return pkg
    
    @staticmethod
    def unpack_up_render_ret(pkg: torch.Tensor):
        # inversion of pack_up_render_ret
        # Do not forget to set requires_grad, space-allocating function may not do it 
        assert pkg.shape[0] == 5 and len(pkg.shape) == 3
        return {
            "render": pkg[:3].clone().detach().requires_grad_(True),
            "depth": pkg[3:4].clone().detach().requires_grad_(True),
            "alpha": pkg[4:5].clone().detach().requires_grad_(True)
        }

    @staticmethod
    def space_for_task(task:ViewMessage, device='cuda', only_shape=False):
        '''
        task: Camera|ViewMessage
        '''
        H, W = task.image_height, task.image_width
        if only_shape:
            return (5, H, W)
        else:
            return torch.zeros((5, H, W), dtype=torch.float32, device=device, requires_grad=False)
    
    @staticmethod
    def pack_up_grad_of_render_ret(render_ret):
        def fetch_grad(t: torch.Tensor):
            return torch.zeros_like(t, dtype=torch.float32) if t.grad is None else t.grad

        with torch.no_grad():
            pkg = torch.cat(
                [fetch_grad(render_ret["render"]), fetch_grad(render_ret["depth"]), fetch_grad(render_ret["alpha"])], 
                dim=0
            )
        return pkg

    @staticmethod
    def unpack_up_grad_of_render_ret(pkg: torch.Tensor):
        # inversion of pack_up_grad_of_render_ret
        assert pkg.shape[0] == 5 and len(pkg.shape) == 3
        return {
            "render": pkg[:3],
            "depth": pkg[3:4],
            "alpha": pkg[4:5]
        }

    @staticmethod
    def space_for_grad_of_task(task:ViewMessage, device='cuda', only_shape=False):
        '''
        task: Camera|ViewMessage
        '''
        H, W = task.image_height, task.image_width
        if only_shape:
            return (5, H, W)
        else:    
            return torch.zeros((5, H, W), dtype=torch.float32, device=device, requires_grad=False)
    