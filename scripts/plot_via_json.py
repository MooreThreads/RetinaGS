import json
import os

def print_matlab(list):
    print(', '.join(map(str, list)))

# 读取文档
benchmark_jsons = ['/root/Nerf/Code/HT/Result_Json/mip_nerf_room_scaling_law_r-1.json']
data_folders = []
for benchmark_json in benchmark_jsons:
    with open(benchmark_json, 'r') as f:
        data = json.load(f)
    data_folders.extend(data)

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
for data_folder in data_folders:    
    target_file = os.path.join(data_folder, "results.json")
    with open(target_file, "r") as f:
        json_data = json.load(f)
        num_GS.append(json_data['train']['GS'])
        
        train_PSNR.append(json_data['train']['Mean of PSNR'])
        train_SSIM.append(json_data['train']['Mean of SSIM'])
        train_LPIPS.append(json_data['train']['Mean of LPIPS'])
        train_GSPI.append(json_data['train']['Mean of GSPI'])
        train_GSPP.append(json_data['train']['Mean of GSPP'])
        train_MAPP.append(json_data['train']['Mean of MAPP'])
        train_MAPP_2.append(json_data['train']['Mean of MAPP_2'])
        train_AGSR.append(json_data['train']['AGSR'])
        
        test_PSNR.append(json_data['test']['Mean of PSNR'])
        test_SSIM.append(json_data['test']['Mean of SSIM'])
        test_LPIPS.append(json_data['test']['Mean of LPIPS'])
        test_GSPI.append(json_data['test']['Mean of GSPI'])
        test_GSPP.append(json_data['test']['Mean of GSPP'])
        test_MAPP.append(json_data['test']['Mean of MAPP'])
        test_MAPP_2.append(json_data['test']['Mean of MAPP_2'])
        test_AGSR.append(json_data['test']['AGSR'])
        
# 格式化
print(benchmark_jsons)
for i, val in enumerate(val_list):
    print(name_list[i])
    print_matlab(val)