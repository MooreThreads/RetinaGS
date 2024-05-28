import os
from PIL import Image
from tqdm import tqdm

# 指定源文件夹和目标文件夹
src_folder = '/root/Nerf/Data/mipnerf360/garden/images'
dst_folder = '/root/Nerf/Data/mipnerf360/garden/resized_3_times/images'
TIMES = 3

# 确保目标文件夹存在
os.makedirs(dst_folder, exist_ok=True)

# 遍历源文件夹中的所有.JPG文件
for idx, filename in enumerate(tqdm(os.listdir(src_folder), desc="resized progress")):
    if filename.endswith('.JPG'):
        # 打开图像
        img = Image.open(os.path.join(src_folder, filename))
        
        # 计算新的尺寸
        new_size = (img.width // TIMES, img.height // TIMES)

        # 调整大小
        img_resized = img.resize(new_size)

        # 保存到目标文件夹
        img_resized.save(os.path.join(dst_folder, filename), quality=100)