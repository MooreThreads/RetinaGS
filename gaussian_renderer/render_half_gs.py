import torch
import math
from diff_gaussian_rasterization_half_gaussian import GaussianRasterizationSettings, GaussianRasterizer
from utils.sh_utils import eval_sh
from typing import Any, NamedTuple

class GradNormHelper(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, meta:torch.Tensor) -> Any:
        assert meta.shape[1] == 1
        N = meta.shape[0]
        return torch.zeros((N, 3), dtype=meta.dtype, device=meta.device, requires_grad=True).contiguous()

    @staticmethod
    def backward(ctx: Any, grad_outputs) -> Any:
        normal2 = torch.norm(grad_outputs[:, :2], dim=-1, keepdim=True)  
        return normal2

gradNormHelpFunction = GradNormHelper.apply  

def render4BoundedGaussianModel(viewpoint_camera, pc, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        projmatrix_inv=viewpoint_camera.projection_matrix_inv,
        range_low=pc.range_low,
        range_up=pc.range_up,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = gradNormHelpFunction(pc._means2D_meta)
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    color, radii, depth, alpha = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": color,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth": depth,
            "alpha": alpha,
            "num_rendered": -1,
            }


class RenderInfoFromGS(NamedTuple):
    means3D: torch.Tensor
    means2D: torch.Tensor
    shs: torch.Tensor
    opacity: torch.Tensor
    scales: torch.Tensor
    rotations: torch.Tensor

def render4renderinfo(viewpoint_camera, GS, info:RenderInfoFromGS, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        projmatrix_inv=viewpoint_camera.projection_matrix_inv,
        range_low=GS.range_low,
        range_up=GS.range_up,
        sh_degree=GS.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = info.means3D
    means2D = info.means2D
    opacity = info.opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    # always get cov3D in cuda
    scales = info.scales
    rotations = info.rotations

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    # always get color in cuda
    shs = info.shs
   
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    color, radii, depth, alpha = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": color,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth": depth,
            "alpha": alpha,
            "num_rendered": -1,
            }
