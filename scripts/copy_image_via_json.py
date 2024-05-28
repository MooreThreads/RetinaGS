import json
import os
import shutil
from tqdm import tqdm
import argparse

def copy_image_via_json(json_file, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(json_file), "r") as f:
            mate = json.load(f)
        
    for i, frame in tqdm(enumerate(mate['frames']), total=len(mate['frames']), desc='ln -s'):
        image_name = frame['file_path']
        save_name = os.path.join(save_path, os.path.basename(os.path.dirname(image_name)) + '_' + os.path.basename(image_name))
        shutil.copyfile(image_name, save_name)    
        # os.system('ln -s ' + image_name + ' ' + save_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', type=str, help='Path of json')    
    parser.add_argument('--save_path', type=str, help='Path of data')
    args = parser.parse_args()
import json
import os
import shutil
from tqdm import tqdm

# json_file = '/root/Nerf/Data/MatrixCity/bdaibdai___MatrixCity/small_city/street/pose/Block_all_unit-1m_choice-100/transforms_train.json'
json_file = '/jfs/shengyi.chen/HT/Data/MatrixCity/bdaibdai___MatrixCity/small_city/street/pose/block_A_unit-1m_dense_sub-1_val/transforms_all.json'
base_dir = '/jfs/shengyi.chen/HT/Data/MatrixCity/bdaibdai___MatrixCity/small_city/street/pose/block_A_unit-1m_dense_sub-1_val'
save_path = '/jfs/burning/code_bases/DenseGaussian/backup/matrix_city_dataset/block_A_unit-1m_dense_sub-1_val/images'

if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)

with open(os.path.join(json_file), "r") as f:
        mate = json.load(f)
    
for i, frame in tqdm(enumerate(mate['frames']), total=len(mate['frames']), desc='ln -s'):
    image_name = os.path.join(base_dir, frame['file_path']) 
    save_name = os.path.join(save_path, os.path.basename(image_name))
    shutil.copyfile(image_name, save_name)    
    # os.system('ln -s ' + image_name + ' ' + save_name)
        

    copy_image_via_json(args.json_file, args.save_path)