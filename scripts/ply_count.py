import numpy as np
from plyfile import PlyData, PlyElement
import matplotlib.pyplot as plt

def load_ply_scales(path : str):
                
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
    
    print("Number of points at checkpoint : ", xyz.shape[0])
    
    return scales

def plot_prefix(scales, save_plot_path, save_plot_2_path):
    # 计算体积
    volumes = np.prod(scales, axis=1)
    volumes_sorted = np.sort(volumes)
    print('mean product of activated gs: ', str(volumes.mean().item()))
    # 排序
    scales = np.sort(scales, axis=1)
    scales_sorted = np.sort(scales, axis=0)
    # 计算前缀和
    cumulative_counts = np.arange(1, scales_sorted.shape[0] + 1)

    # 作图    
    indices = np.linspace(0, scales_sorted.shape[0]-1, 1000, dtype=int) 
    # 三轴统计
    plt.figure()
    for i in range(scales.shape[1]):
        # plt.plot(np.squeeze(scales_sorted[indices, i]), np.squeeze(cumulative_counts[indices]), label=f'Row {i+1}')
        plt.semilogx(np.squeeze(scales_sorted[indices, i]), np.squeeze(cumulative_counts[indices]), label=f'axis {i+1}')
        print("mean of axis ", str(i+1), " ", str(scales_sorted[:, i].mean().item()))
        
    plt.legend()
    plt.savefig(save_plot_path)
    print("save at " + save_plot_path)
    # 体积统计
    plt.figure()
    plt.semilogx(np.squeeze(volumes_sorted[indices]), np.squeeze(cumulative_counts[indices]), label=f'volume', color='k')
    plt.legend()
    plt.savefig(save_plot_2_path)
    print("save at " + save_plot_2_path)
    

    

if __name__ == '__main__':
    # ply_path = '/root/Nerf/Predict/indoor/demo_room_luchao/colmap_dense_None-PM_None-xyz_pointcloud_sample_rate-16_huber_loss_r-1_train-with-epoch/point_cloud/iteration_31136/point_cloud_only_activated.ply'
    # ply_path = '/root/Nerf/Predict/indoor/demo_room_luchao/colmap_dense_None-PM_None-xyz_pointcloud_sample_rate-16_huber_loss_r-1_scale-product_train-with-epoch/point_cloud/iteration_31136/point_cloud_only_activated.ply'
    # ply_path = '/root/Nerf/Predict/indoor/demo_room_luchao/colmap_dense_None-PM_None-xyz_pointcloud_sample_rate-16_huber_loss_r-1_scale-sum_train-with-epoch/point_cloud/iteration_31136/point_cloud_only_activated.ply'
    # ply_path = '/root/Nerf/Predict/indoor/demo_room_luchao/colmap_dense_None-PM_None-xyz_pointcloud_sample_rate-16_huber_loss_r-1_scale-square-sum_train-with-epoch/point_cloud/iteration_31136/point_cloud_only_activated.ply'
    ply_path = '/root/Nerf/Predict/indoor/demo_room_luchao/colmap_default_gs_default_r_1/point_cloud/iteration_30000/point_cloud_only_activated.ply'
    save_plot_path = '/root/Nerf/Code/DenseGaussian/backup/demo_room/default.png'
    save_plot_2_path = '/root/Nerf/Code/DenseGaussian/backup/demo_room/default_volume.png'
    scales = load_ply_scales(ply_path)
    scales = np.exp(scales)
    plot_prefix(scales, save_plot_path, save_plot_2_path)
    