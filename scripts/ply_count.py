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

def plot_prefix(scales, save_plot_path):
    # 排序
    scales = np.sort(scales, axis=1)
    scales_sorted = np.sort(scales, axis=0)
    # 计算前缀和
    cumulative_counts = np.arange(1, scales_sorted.shape[0] + 1)

    # 作图
    plt.figure()
    indices = np.linspace(0, scales_sorted.shape[0]-1, 1000, dtype=int) 
    for i in range(scales.shape[1]):
        # plt.plot(np.squeeze(scales_sorted[indices, i]), np.squeeze(cumulative_counts[indices]), label=f'Row {i+1}')
        plt.semilogx(np.squeeze(scales_sorted[indices, i]), np.squeeze(cumulative_counts[indices]), label=f'Row {i+1}')

    plt.legend()
    plt.savefig(save_plot_path)
    print("save at " + save_plot_path)

    

if __name__ == '__main__':
    ply_path = '/root/Nerf/Predict/mipnerf360/room/A100_colmap-default_gaussian-splatting-default_mipnerf360-room/point_cloud/iteration_30000/point_cloud_only_activated.ply'
    save_plot_path = '/root/Nerf/Code/HT/DenseGaussian/backup/prefix_sums.png'
    scales = load_ply_scales(ply_path)
    scales = np.exp(scales)
    plot_prefix(scales, save_plot_path)
    