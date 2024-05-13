import json
import os
import shutil
from tqdm import tqdm

json_file = '/root/Nerf/Data/MatrixCity/bdaibdai___MatrixCity/small_city/aerial/pose/block_A_unit-1m/transforms_train.json'
save_path = '/root/Nerf/Data/MatrixCity/block_A_unit-1m_all/images'

if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)

with open(os.path.join(json_file), "r") as f:
        mate = json.load(f)
    
for i, frame in tqdm(enumerate(mate['frames']), total=len(mate['frames']), desc='ln -s'):
    image_name = frame['file_path']
    save_name = os.path.join(save_path, os.path.basename(os.path.dirname(image_name)) + '_' + os.path.basename(image_name))
    shutil.copyfile(image_name, save_name)    
    # os.system('ln -s ' + image_name + ' ' + save_name)
    