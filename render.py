
import torch
from scene import SimpleScene
from torch.utils.data import DataLoader
import os, glob
from tqdm import tqdm
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
from utils.image_utils import psnr
import torchvision
from lpipsPyTorch import lpips

# model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
model = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.DEFAULT)
model.cuda()
for param in model.parameters():
    param.requires_grad = False

img = torch.rand((3, 128, 128), device='cuda', requires_grad=True)

# ret = model(img)

loss = lpips(img, img+1)


loss.backward()

print('end')