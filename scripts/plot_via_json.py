import json
import os

def print_matlab(list):
    print(', '.join(map(str, list)))

# 读取文档

benchmark_jsons = []
dataset='mipnerf360'
RESOLUTION=1600
var_list=['bicycle', 'bonsai', 'counter',  'flowers',  'kitchen',  'stump',  'treehill']
for var in var_list:
    benchmark_jsons.append(
    f"/jfs/shengyi.chen/HT/Predict/{dataset}/{var}/default-PM_r-{RESOLUTION}_dgt-0002_iter-60000/results.json"
    )

dataset='MatrixCity'
var_list=[1, 2, 4, 8, 16, 32, 64]
for pointcloud_sample_rate in var_list:
    benchmark_jsons.append(
    f"/jfs/shengyi.chen/HT/Predict/MatrixCity/colmap-dense_None-PM_minimal-xyz_r_1_pointcloud_sample_rate-{pointcloud_sample_rate}_epoch-40_without_sky_ball/results.json"   
    )
# benchmark_jsons = ['/root/Nerf/Code/DenseGaussian/Result_Json/garden_PM.json']

# data_folders = []
# for benchmark_json in benchmark_jsons:
#     with open(benchmark_json, 'r') as f:
#         data = json.load(f)
#     data_folders.extend(data)

# 定义需要值
num_GS = []
train_PSNR = []
train_SSIM = []
train_LPIPS = []
train_GSPI = []
train_GSPP = []
train_MAPP = []
train_MAPP_2 = []
train_AGSR = []
test_PSNR = []
test_SSIM = []
test_LPIPS = []
test_GSPI = []
test_GSPP = []
test_MAPP = []
test_MAPP_2 = []
test_AGSR = []
# 定义列表
val_list = [
    num_GS,
    train_PSNR, train_SSIM, train_LPIPS,
    train_GSPI, train_GSPP, train_MAPP, train_MAPP_2,
    train_AGSR, 
    test_PSNR, test_SSIM, test_LPIPS,
    test_GSPI, test_GSPP, test_MAPP, test_MAPP_2,
    test_AGSR    
    ]
name_list = [
    "num_GS",
    "train_PSNR", "train_SSIM", "train_LPIPS",
    "train_GSPI", "train_GSPP", "train_MAPP", "train_MAPP_2",
    "train_AGSR", 
    "test_PSNR", "test_SSIM", "test_LPIPS",
    "test_GSPI", "test_GSPP", "test_MAPP", "test_MAPP_2",
    "test_AGSR"    
    ]
# 读取需要值
for target_file in benchmark_jsons:    
    # target_file = os.path.join(data_folder, "results.json")
    with open(target_file, "r") as f:
        json_data = json.load(f)
        # num_GS.append(json_data['train']['GS'])
        
        # train_PSNR.append(json_data['train']['Mean of PSNR'])
        # train_SSIM.append(json_data['train']['Mean of SSIM'])
        # train_LPIPS.append(json_data['train']['Mean of LPIPS'])
        # train_GSPI.append(json_data['train']['Mean of GSPI'])
        # train_GSPP.append(json_data['train']['Mean of GSPP'])
        # train_MAPP.append(json_data['train']['Mean of MAPP'])
        # train_MAPP_2.append(json_data['train']['Mean of MAPP_2'])
        # train_AGSR.append(json_data['train']['AGSR'])
        
        test_PSNR.append(json_data['test']['Mean of PSNR'])
        test_SSIM.append(json_data['test']['Mean of SSIM'])
        test_LPIPS.append(json_data['test']['Mean of LPIPS'])
        test_GSPI.append(json_data['test']['Mean of GSPI'])
        test_GSPP.append(json_data['test']['Mean of GSPP'])
        test_MAPP.append(json_data['test']['Mean of MAPP'])
        test_MAPP_2.append(json_data['test']['Mean of MAPP_2'])
        test_AGSR.append(json_data['test']['AGSR'])


        print('{}\npsnr{}\nssim{}\nlpips{}\n'.format(
            target_file, json_data['test']['Mean of PSNR'], json_data['test']['Mean of SSIM'], json_data['test']['Mean of LPIPS']
        ))
        
# 格式化
print(benchmark_jsons)
for i, val in enumerate(val_list):
    print(name_list[i])
    print_matlab(val)