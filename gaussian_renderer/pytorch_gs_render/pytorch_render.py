import torch
import typing
import numpy as np
import math
from .util import spherical_harmonics, cg_torch
from scene.cameras import Camera, Patch

# this raster assume images in batch has the same shape

torch.ops.load_library("submodules/Pytorch-GaussianSplatting/gaussian_splatting/submodules/gaussian_raster/build/libGaussianRaster.so")


class World2NDC(torch.autograd.Function):
    @staticmethod
    def forward(ctx,position,view_project_matrix:torch.Tensor):
        hom_pos=torch.matmul(position,view_project_matrix)
        repc_hom_w=1/(hom_pos[...,3:4]+1e-7)
        ndc_pos=hom_pos*repc_hom_w
        ctx.save_for_backward(view_project_matrix,position,repc_hom_w)
        return ndc_pos
    
    @staticmethod
    def backward(ctx,grad_ndc_pos:torch.Tensor):
        (view_project_matrix,position,repc_hom_w)=ctx.saved_tensors

        #wtf?
        # repc_hom_w=repc_hom_w[...,0]
        # position_grad=torch.zeros_like(position)

        # mul1=(view_project_matrix[...,0,0] * position[...,0] + view_project_matrix[...,1,0] * position[...,1] + view_project_matrix[...,2,0] * position[...,2] + view_project_matrix[...,3,0]) * repc_hom_w * repc_hom_w
        # mul2=(view_project_matrix[...,0,1] * position[...,0] + view_project_matrix[...,1,1] * position[...,1] + view_project_matrix[...,2,1] * position[...,2] + view_project_matrix[...,3,1]) * repc_hom_w * repc_hom_w

        # position_grad[...,0]=(view_project_matrix[...,0,0] * repc_hom_w - view_project_matrix[...,0,3] * mul1) * grad_ndc_pos[...,0] + (view_project_matrix[...,0,1] * repc_hom_w - view_project_matrix[...,0,3] * mul2) * grad_ndc_pos[...,1]

        # position_grad[...,1]=(view_project_matrix[...,1,0] * repc_hom_w - view_project_matrix[...,1,3] * mul1) * grad_ndc_pos[...,0] + (view_project_matrix[...,1,1] * repc_hom_w - view_project_matrix[...,1,3] * mul2) * grad_ndc_pos[...,1]

        # position_grad[...,2]=(view_project_matrix[...,2,0] * repc_hom_w - view_project_matrix[...,2,3] * mul1) * grad_ndc_pos[...,0] + (view_project_matrix[...,2,1] * repc_hom_w - view_project_matrix[...,2,3] * mul2) * grad_ndc_pos[...,1]

        position_grad=torch.ops.RasterBinning.world2ndc_backword(view_project_matrix,position,repc_hom_w,grad_ndc_pos)

        return (position_grad,None)


class CreateTransformMatrix(torch.autograd.Function):
    @staticmethod
    def forward(ctx,quaternion:torch.Tensor,scale:torch.Tensor):
        ctx.save_for_backward(quaternion,scale)
        transform_matrix=torch.ops.RasterBinning.createTransformMatrix_forward(quaternion,scale)
        return transform_matrix
    
    @staticmethod
    def backward(ctx,grad_transform_matrix:torch.Tensor):
        (quaternion,scale)=ctx.saved_tensors
        grad_quaternion,grad_scale=torch.ops.RasterBinning.createTransformMatrix_backward(grad_transform_matrix,quaternion,scale)
        return grad_quaternion,grad_scale


class Transform3dCovAndProjTo2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx,cov3d:torch.Tensor,transforms_matrix:torch.Tensor):
        ctx.save_for_backward(transforms_matrix)
        cov2d=(transforms_matrix@cov3d@transforms_matrix.transpose(-1,-2))[:,:,0:2,0:2].contiguous()
        return cov2d
    
    @staticmethod
    def backward(ctx,cov2d_gradient:torch.Tensor):
        (transforms_matrix,)=ctx.saved_tensors
        N,P=transforms_matrix.shape[0:2]
        # cov3d_gradient=torch.zeros((N,P,3,3),device=transforms_matrix.device)
        # for i in range(0,3):
        #     for j in range(0,3):
        #         cov3d_gradient[:,:,i,j]=\
        #             (transforms_matrix[:,:,0,i]*transforms_matrix[:,:,0,j])*cov2d_gradient[:,:,0,0]\
        #             + (transforms_matrix[:,:,0,i]*transforms_matrix[:,:,1,j])*cov2d_gradient[:,:,0,1]\
        #             + (transforms_matrix[:,:,1,i]*transforms_matrix[:,:,0,j])*cov2d_gradient[:,:,1,0]\
        #             + (transforms_matrix[:,:,1,i]*transforms_matrix[:,:,1,j])*cov2d_gradient[:,:,1,1]
        temp_matrix_A=transforms_matrix[:,:,(0,0,1,1),:].transpose(-1,-2).contiguous()
        temp_matrix_B=(transforms_matrix[:,:,(0,1,0,1),:]*cov2d_gradient.reshape(N,P,-1,1)).contiguous()
        cov3d_gradient=temp_matrix_A@temp_matrix_B

        return cov3d_gradient,None


class CreateCovarianceMatrix(torch.autograd.Function):
    @staticmethod
    def forward(ctx,transforms:torch.Tensor):
        ctx.save_for_backward(transforms)
        cov=transforms.transpose(-1,-2).contiguous()@transforms
        return cov
    
    @staticmethod
    def backward(ctx,CovarianceMatrixGradient:torch.Tensor):
        (transforms,)=ctx.saved_tensors
        return (2*transforms@CovarianceMatrixGradient)


class GaussiansRaster(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        sorted_pointId:torch.Tensor,
        tile_start_index:torch.Tensor,
        mean2d:torch.Tensor,
        cov2d_inv:torch.Tensor,
        color:torch.Tensor,
        opacities:torch.Tensor,
        tiles:torch.Tensor,
        tile_size:int,
        tiles_num_x:int,
        tiles_num_y:int,
        img_h:int,
        img_w:int
    ):
        img,transmitance,lst_contributor=torch.ops.RasterBinning.rasterize_forward(sorted_pointId,tile_start_index,mean2d,cov2d_inv,color,opacities,tiles,
                                                                                   tile_size,tiles_num_x,tiles_num_y,img_h,img_w)
        ctx.save_for_backward(sorted_pointId,tile_start_index,transmitance,lst_contributor,mean2d,cov2d_inv,color,opacities,tiles)
        ctx.arg_tile_size=tile_size
        ctx.tiles_num=(tiles_num_x,tiles_num_y,img_h,img_w)
        return img,transmitance
    
    @staticmethod
    def backward(ctx, grad_out_color:torch.Tensor, grad_out_transmitance):
        sorted_pointId,tile_start_index,transmitance,lst_contributor,mean2d,cov2d_inv,color,opacities,tiles=ctx.saved_tensors
        (tiles_num_x,tiles_num_y,img_h,img_w)=ctx.tiles_num
        tile_size=ctx.arg_tile_size

        grad_mean2d,grad_cov2d_inv,grad_color,grad_opacities=torch.ops.RasterBinning.rasterize_backward(sorted_pointId,tile_start_index,mean2d,cov2d_inv,color,opacities,tiles,
                                                                                                        transmitance,lst_contributor,grad_out_color,
                                                                                                        tile_size,tiles_num_x,tiles_num_y,img_h,img_w)
        grads = (
            None,
            None,
            grad_mean2d,
            grad_cov2d_inv,
            grad_color,
            grad_opacities,
            None,
            None,
            None,
            None,
            None,
            None
        )

        return grads


@torch.no_grad()
def culling_and_sort(ndc_pos,translated_pos,limit_LURD=None):
    '''
    todo implement in cuda
    input: ViewMatrix,ProjMatrix
    output: sorted_visible_points,num_of_points
    '''
    if limit_LURD is None:
        culling_result=torch.any(ndc_pos[...,0:2]<-1.3,dim=2)|torch.any(ndc_pos[...,0:2]>1.3,dim=2)|(translated_pos[...,2]<=0.01)#near plane 0.01
    else:
        culling_result=torch.any(ndc_pos[...,0:2]<limit_LURD.unsqueeze(1)[...,0:2]*1.3,dim=2)|torch.any(ndc_pos[...,0:2]>limit_LURD.unsqueeze(1)[...,2:4]*1.3,dim=2)|(translated_pos[...,2]<=0)|(translated_pos[...,2]<=0.01)

    max_visible_points_num=(~culling_result).sum(1).max()
    threshhold=translated_pos[...,2].max()+1

    masked_depth=translated_pos[...,2]*(~culling_result)+threshhold*culling_result
    sorted_masked_depth,visible_point=torch.sort(masked_depth,1)
    point_index_mask=(sorted_masked_depth<threshhold)[...,:max_visible_points_num]
    points_num=point_index_mask.sum(1)
    visible_point=visible_point[...,:max_visible_points_num]*point_index_mask

    return visible_point,points_num


def transform_to_cov3d(scaling_vec,rotator_vec)->torch.Tensor:
        
    #transform_matrix=GaussianSplattingModel.create_transform_matrix(scaling_vec,rotator_vec)
    transform_matrix=CreateTransformMatrix.apply(rotator_vec,scaling_vec)
    
    #cov3d=torch.matmul(transform_matrix.transpose(-1,-2),transform_matrix)
    cov3d=CreateCovarianceMatrix.apply(transform_matrix)
    
    return cov3d,transform_matrix


def proj_cov3d_to_cov2d(cov3d,point_positions,view_matrix,camera_focal)->torch.Tensor:
        with torch.no_grad():
            t=torch.matmul(point_positions,view_matrix)
            
            # Keep no_grad. Auto gradient will make bad influence of xyz
            # J=torch.zeros_like(cov3d,device='cuda')#view point mat3x3
            # camera_focal=camera_focal.unsqueeze(1)
            # tz_square=t[:,:,2]*t[:,:,2]
            # J[:,:,0,0]=camera_focal[:,:,0]/t[:,:,2]#focal x
            # J[:,:,1,1]=camera_focal[:,:,1]/t[:,:,2]#focal y
            # J[:,:,0,2]=-(camera_focal[:,:,0]*t[:,:,0])/tz_square
            # J[:,:,1,2]=-(camera_focal[:,:,1]*t[:,:,1])/tz_square
            J=torch.ops.RasterBinning.jacobianRayspace(t,camera_focal)

        M=view_matrix.unsqueeze(1)[:,:,0:3,0:3].transpose(-1,-2).contiguous()
        T=J@M

        #T' x cov3d' x T
        #cov2d=(T@cov3d@T.transpose(-1,-2))[:,:,0:2,0:2]
        cov2d=Transform3dCovAndProjTo2d.apply(cov3d,T)#backward improvement

        cov2d[:,:,0,0]+=0.3
        cov2d[:,:,1,1]+=0.3
        return cov2d


@torch.no_grad()
def binning(tilesXY:torch.Tensor, tile_size:int, image_size:torch.Tensor, ndc:torch.Tensor,cov2d:torch.Tensor,valid_points_num:torch.Tensor,b_gather=False):
        tilesX = tilesXY[:, 0:1].int() # (B, 1), LURD must be int32
        tilesY = tilesXY[:, 1:2].int() # (B, 1)
        tiles_num:int = int(tilesX[0, 0] * tilesY[0, 0])
        # image_size (B,2)

        coordX=(ndc[:,:,0]+1.0)*0.5*image_size[:, 0:1]-0.5  # (B,N) * (B,1)
        coordY=(ndc[:,:,1]+1.0)*0.5*image_size[:, 1:2]-0.5  # (B,N) * (B,1)

        det=cov2d[:,:,0,0]*cov2d[:,:,1,1]-cov2d[:,:,0,1]*cov2d[:,:,0,1]
        mid=0.5*(cov2d[:,:,0,0]+cov2d[:,:,1,1])
        temp=(mid*mid-det).clamp_min(0.1).sqrt()
        pixel_radius=(3*(torch.max(mid+temp,mid-temp).sqrt())).ceil()
        
        zeros = torch.zeros_like(tilesX)

        # clamp (B,N) with (B,1)
        L=((coordX-pixel_radius)/tile_size).floor().int().clamp(zeros,tilesX)
        U=((coordY-pixel_radius)/tile_size).floor().int().clamp(zeros,tilesY)
        R=((coordX+pixel_radius+tile_size-1)/tile_size).floor().int().clamp(zeros,tilesX)
        D=((coordY+pixel_radius+tile_size-1)/tile_size).floor().int().clamp(zeros,tilesY)

        #calculate params of allocation
        tiles_touched=(R-L)*(D-U)
        prefix_sum=tiles_touched.cumsum(1)
        total_tiles_num_batch=prefix_sum.gather(1,valid_points_num.unsqueeze(1)-1)
        allocate_size=total_tiles_num_batch.max().cpu()

        
        # allocate table and fill tile_id in it(uint 16)
        my_table=torch.ops.RasterBinning.duplicateWithKeys(L,U,R,D,valid_points_num,prefix_sum,int(allocate_size),int(tilesX[0]))
        tileId_table:torch.Tensor=my_table[0]
        pointId_table:torch.Tensor=my_table[1]


        # sort tile_id with torch.sort
        sorted_tileId,indices=torch.sort(tileId_table,dim=1,stable=True)
        sorted_pointId=pointId_table.gather(dim=1,index=indices)

        #debug:check total_tiles_num_batch
        #cmp_result=(tileId_table!=0).sum(dim=1)==total_tiles_num_batch[:,0]
        #print(cmp_result)
        #cmp_result=(sorted_tileId!=0).sum(dim=1)==total_tiles_num_batch[:,0]
        #print(cmp_result)

        # range
        tile_start_index=torch.ops.RasterBinning.tileRange(sorted_tileId,int(allocate_size),int(tiles_num-1+1))#max_tile_id:tilesnum-1, +1 for offset(tileId 0 is invalid)

        if b_gather:
            sorted_pointId=sorted_pointId.long()
            
        return tile_start_index,sorted_pointId,sorted_tileId,tiles_touched, pixel_radius

def raster(image_size_tensor, tiles_map_size, image_size, tile_size:int, ndc_pos:torch.Tensor,cov2d:torch.Tensor,color:torch.Tensor,opacities:torch.Tensor,tile_start_index:torch.Tensor,sorted_pointId:torch.Tensor,sorted_tileId:torch.Tensor,tiles:torch.Tensor):
        
        # cov2d_inv=torch.linalg.inv(cov2d)#forward backward 1s
        #faster but unstable
        reci_det=1/(torch.det(cov2d)+1e-7)
        cov2d_inv=torch.zeros_like(cov2d)
        cov2d_inv[...,0,1]=-cov2d[...,0,1]*reci_det
        cov2d_inv[...,1,0]=-cov2d[...,1,0]*reci_det
        cov2d_inv[...,0,0]=cov2d[...,1,1]*reci_det
        cov2d_inv[...,1,1]=cov2d[...,0,0]*reci_det

        mean2d=(ndc_pos[:,:,0:2]+1.0)*0.5*image_size_tensor.view(-1, 1, 2)-0.5  # (B, N, 2)*(B,1,2)


        img,transmitance=GaussiansRaster.apply(sorted_pointId,tile_start_index,mean2d,cov2d_inv,color,opacities,tiles,
                                               tile_size, tiles_map_size[0], tiles_map_size[1], image_size[1], image_size[0])

        return img,transmitance

class Means2DGradGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, meta:torch.Tensor, mean2d:torch.Tensor, visible_points_idx:torch.Tensor, visible_points_num:torch.Tensor):
        assert meta.shape[1] == 1
        N = meta.shape[0]
        ctx.save_for_backward(visible_points_idx, visible_points_num)
        ctx.N_meta = N
        return mean2d + 0

    @staticmethod
    def backward(ctx, grad_mean2d:torch.Tensor):
        visible_points_idx, visible_points_num = ctx.saved_tensors # (B, N), (B,)
        normal2 = torch.norm(grad_mean2d[:, :, :2], dim=-1, keepdim=True)  # (B, N, 1)
        sum_norm2 = torch.zeros((ctx.N_meta, 1), device=grad_mean2d.device, dtype=grad_mean2d.dtype)
        for batch_idx in range(len(visible_points_idx)):
            valid_length = visible_points_num[batch_idx]    # tensor(int)
            valid_idx = visible_points_idx[batch_idx][:valid_length]   # (num, )
            sum_norm2[valid_idx] += normal2[batch_idx, :valid_length]  # (valid_length, 1) + (valid_length, 1)

        return sum_norm2, grad_mean2d, None, None

means2DGradGatherFunction = Means2DGradGather.apply


def render(pc, batch_data):
    viewpoint_cameras:torch.Tensor = batch_data['packed_views']
    tile_maps:torch.Tensor = batch_data['tile_maps']
    tile_nums:list = batch_data['tile_nums']    # list of num of valid tiles in map
    tile_maps_sizes:torch.Tensor = batch_data['tile_maps_sizes']  # (B, X_Y) size of complete tile_map
    tile_maps_sizes_list = batch_data['tile_maps_sizes_list'] # [w/16 , h/16]
    image_size_list = batch_data['image_size_list'] # [w, h]
    patches_cpu = batch_data['patches_cpu']

    """
    viewpoint_cameras: packed up Patches of (batch_size, 16, 4) on cuda
    tile_maps: (batch_size, max_num_tiles) tensor padded with 0 on cuda
    tile_nums: list of int 
    """
    batch_size = len(viewpoint_cameras)
    view_matrix = viewpoint_cameras[:, 0:4, :].contiguous()
    view_project_matrix = viewpoint_cameras[:, 4:8, :].contiguous()
    camera_center = viewpoint_cameras[:, 13, 0:3].contiguous()
    camera_wh = torch.tensor(image_size_list, device='cuda').view(1, 2).repeat(batch_size, 1)
    camera_fov_xy = viewpoint_cameras[:, 12, 2:4]
    camera_focal = camera_wh / (torch.tan(camera_fov_xy*0.5)*2)
    NUM_gs = pc._xyz.shape[0]

    with torch.no_grad():
        ndc_pos = cg_torch.world_to_ndc(pc.get_xyz_hom, view_project_matrix)
        translated_pos = cg_torch.world_to_view(pc.get_xyz_hom,view_matrix)
        visible_points_idx, visible_points_num = culling_and_sort(ndc_pos,translated_pos)

    vis_scales = pc.get_scaling[visible_points_idx]
    vis_rotators = pc.get_rotation[visible_points_idx]
    vis_means3D = pc.get_xyz_hom[visible_points_idx]
    vis_opacity = pc.get_opacity[visible_points_idx]
    vis_sh0 = pc._features_dc[visible_points_idx]

    vis_cov3d, transform_matrix = transform_to_cov3d(vis_scales, vis_rotators)
    vis_cov2d = proj_cov3d_to_cov2d(vis_cov3d, vis_means3D, view_matrix, camera_focal)
    
    ### color ###
    visible_color=(vis_sh0*spherical_harmonics.C0+0.5).squeeze(2).clamp_min(0)

    ### mean of 2d-gaussian ###
    #ndc_pos_batch=cg_torch.world_to_ndc(visible_positions,view_project_matrix)
    _ndc_pos_batch=World2NDC.apply(vis_means3D, view_project_matrix)
    ndc_pos_batch = means2DGradGatherFunction(pc._means2D_meta, _ndc_pos_batch, visible_points_idx, visible_points_num)


    #### binning ###
    tile_start_index,sorted_pointId,sorted_tileId,tiles_touched, radii = binning(
        tile_maps_sizes, 16, camera_wh, 
        ndc_pos_batch, vis_cov2d, visible_points_num)
    
    #### raster ###
    tile_img, tile_transmitance = raster(
        camera_wh, tile_maps_sizes_list, image_size_list, 16, 
        ndc_pos_batch, vis_cov2d, visible_color, vis_opacity, 
        tile_start_index, sorted_pointId, sorted_tileId, tile_maps)
    tile_transmitance:torch.Tensor = tile_transmitance.unsqueeze(2)
    # (B, num_tile, 3, 16 ,16), (B, num_tile, 3, 16 ,16)

    #### post-processing
    all_rets = []
    for i in range(batch_size):
        render_ret = {}
        patch:Patch = patches_cpu[i]
        vaild_tile_num = tile_nums[i]
        render_ret['render'] = patch.tiles2patch(tile_img[i][:vaild_tile_num])
        render_ret['alpha'] = 1 - patch.tiles2patch(tile_transmitance[i][:vaild_tile_num])
        render_ret['depth'] = 0 * render_ret['alpha']   # so that 'depth' requires grad and has a grad_fn
        render_ret['num_rendered'] = -1

        render_ret['viewspace_points'] = None
        render_ret['visibility_filter'] = torch.zeros((NUM_gs,), dtype=torch.bool, device='cuda', requires_grad=False)
        render_ret['radii'] = torch.zeros((NUM_gs,), dtype=torch.float, device='cuda', requires_grad=False)

        vis_idx = visible_points_idx[i][:visible_points_num[i]]
        render_ret['visibility_filter'][vis_idx] = True
        render_ret['radii'][vis_idx] = radii[i][:len(vis_idx)]

        all_rets.append(render_ret)

    return all_rets