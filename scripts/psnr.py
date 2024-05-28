from utils.image_utils import psnr
from PIL import Image
from utils.general_utils import PILtoTorch
import os

render_path = "/root/Nerf/Predict/mipnerf360/garden/colmap_dense_None-PM_train-with-epoch/test/ours_64401/gt"
gts_path = "/root/Nerf/Predict/mipnerf360/garden/resized-3_colmap_default_gs_default_r_1600/test/ours_30000/renders"

for filename in os.listdir(render_path):
    gt = PILtoTorch(Image.open(os.path.join(gts_path, filename)), (1600, 1036))
    render = PILtoTorch(Image.open(os.path.join(render_path, filename)), ((1600, 1036)))
    pass