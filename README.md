# RetinaGS: Scalable Training for Dense Scene Rendering with Billion-Scale 3D Gaussians

<img src="./assets/teaser.png">

We introduce RetinaGS, which explores the possibility of training high-parameter 3D Gaussian splatting (3DGS) models on large-scale, high-resolution datasets. This codebase maintain a model parallel traning framework for native 3DGS which uses a proper rendering equation and can be applied to any scene and arbitrary distribution of Gaussian primitives. 

<img src="./assets/pipeline.png">


[[Project Page]](https://ai-reality.github.io/RetinaGS/)
[[Paper]](https://arxiv.org/pdf/2406.11836)

## Cloning the Repository

The repository contains submodules, thus please check it out with

```
# SSH
git clone git@github.com:mthreads/DenseGaussian.git --recursive
```

or

```
# HTTPS
git clone https://github.com/mthreads/DenseGaussian.git --recursive
```

## Setup

### 3DGS Setup
Our implement is based on 3DGS. First, set up a environment following the guidance in https://github.com/graphdeco-inria/gaussian-splatting

```shell
SET DISTUTILS_USE_SDK=1 # Windows only
conda env create --file environment.yml
conda activate gaussian_splatting
```

### RetinaGS Setup
Install additional packages for RetinaGS.

```
conda install numba opencv-python scipy
```

Install new raster-kernel in rasterization_kernels/diff-gaussian-rasterization-half-gaussian.

```
cd rasterization_kernels/diff-gaussian-rasterization-half-gaussian
python -m setup.py install
```

Please note that we only test RetinaGS on Linux System.

## Usage

### Data 
The data could be orgnised as follows (Colmap formulation):
```
data/
├── scene1/
│   ├── images
│   │   ├── IMG_0.jpg
│   │   ├── IMG_1.jpg
│   │   ├── ...
│   ├── sparse/
│       └──0/
├── scene2/
│   ├── images
│   │   ├── IMG_0.jpg
│   │   ├── IMG_1.jpg
│   │   ├── ...
│   ├── sparse/
│       └──0/
```

也支持其他数据格式（如blender和MegaNeRF），见scene/simple_scene.py的Line 37~53.

### Model 

支持两种模式，split和whole，分别格式为

```
model/
├── scene1_split_model/
│   ├── rank_0
│   │   ├── point_cloud
│   │   │   ├── iteration_xxx
│   │   │   │   ├── point_cloud_xx.ply
│   │   │   │   ├── point_cloud_yy.ply
│   │   │   │   ├── ...
│   │   ├── tree_0.txt
│   │   ├── trainset_relation.pt
│   │   ├── testset_relation.pt
│   │   ├── cfg_args
│   ├── rank_1
│   │   ├── point_cloud
│   │   │   ├── iteration_xxx
│   │   │   │   ├── point_cloud_zz.ply
│   │   │   │   ├── point_cloud_vv.ply
│   │   │   │   ├── ...
│   ├── rank_2
│   │   ├── point_cloud
│   │   │   ├── iteration_xxx
│   │   │   │   ├── point_cloud_mm.ply
│   │   │   │   ├── point_cloud_nn.ply
│   │   │   │   ├── ...
│   |── ...
├── scene2_split_model/
│   ├── ...
```

```
model/
├── scene1_whole_model/
│   ├── point_cloud
│   │   ├── iteration_xxx
│   ├── cfg_args
├── scene2_whole_model/
│   ├── ...
```

### Evaluation
Get data and pretrained models ([[Garden-1.6k]](https://ai-reality.github.io/RetinaGS/))

```
CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --nproc_per_node=2 \
    --master_addr=127.0.0.1 --master_port=7356 \
    train_MP_tree.py -s path/2/data \
        -m path/2/model --bvh_depth 2 \
        --eval --EVAL_ONLY --SAVE_EVAL_IMAGE --SAVE_EVAL_SUB_IMAGE \
        --max_batch_size 4  --max_load 8  
```

### Model Zoo

| Datasets                                                      | PSNR | #GS   | PSNR of 3DGS | #GS of 3DGS |
|:-----------------:                                            |:----:|:-----:|:------------:|:-----------:|
| [[Garden-1.6k]](https://ai-reality.github.io/RetinaGS/)       |27.74 |62.94M |   27.58      |   6.92M     |
| [[Garden-full]](https://ai-reality.github.io/RetinaGS/)       |27.06 |62.94M |   26.67      |   7.39M     |
| [[ScanNet++]](https://ai-reality.github.io/RetinaGS/)         |29.71 |47.59M |   28.95      |   2.65M     |
| [[MatrixCity-M]](https://ai-reality.github.io/RetinaGS/)      |31.12 |62.18M |   27.81      |   1.53M     |
| [[Mega-NeRF]](https://ai-reality.github.io/RetinaGS/)         |25.09 |27.9M  |   24.47      |   4.7M      |
| [[MatrixCity-Aerial]](https://ai-reality.github.io/RetinaGS/) |27.70 |217.3M |   24.47      |   25.06M    |

M means Million. See Appendix in [[Paper]](https://arxiv.org/pdf/2406.11836) for complete results.

### Training 
For single node, an example of using default densification strategy and Colmap Initialization  command is:
```
CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --nproc_per_node=2 \
    --master_addr=127.0.0.1 --master_port=7356 \
    train_MP_tree.py -s path/2/data \
        -m path/2/model --bvh_depth 2 \
        --eval \
        --epochs 187 \
        --max_batch_size 4  --max_load 8  \
        --points3D points3D
```
Here, you would create 2**(bvh_depth) submodels for 2 GPUs, namely 2 submodels for each GPU. The max_batch_size and max_load are arguments for controlling memory cost, a render task for a submodel weight 1 load, thus "--max_batch_size 4  --max_load 8" just set every batch as size of 4 in this case. 

For multiple nodes, start command on each node with corresponding parameters, and example shell scripts for launching/stopping multiple nodes training can be found in multi_node_cmds/

从MVS Initialization出发，关闭点管理
```
CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --nproc_per_node=2 \
    --master_addr=127.0.0.1 --master_port=7356 \
    train_MP_tree.py -s path/2/data \
        -m path/2/model --bvh_depth 2 \
        --eval \
        --epochs 187 \
        --max_batch_size 4  --max_load 8  \
        --position_lr_init 0.0000016 \
        --position_lr_final 0.000000016 \
        --densify_until_epoch 0 \
        --points3D MVS_points3D --pointcloud_sample_rate 1
```

加一个flag可以实现存单个ply，读单个ply分配到多个GPU

## Citation
Please cite the following paper if you use this repository in your reseach or work.
```
@article{li2024retinags,
  title={RetinaGS: Scalable Training for Dense Scene Rendering with Billion-Scale 3D Gaussians},
  author={Li, Bingling and Chen, Shengyi and Wang, Luchao and He, Kaimin and Yan, Sijie and Xiong, Yuanjun},
  journal={arXiv preprint arXiv:2406.11836},
  year={2024}
}
```
## License
Copyright @2023-2024 Moore Threads Technology Co., Ltd("Moore Threads"). All rights reserved. This software may contains part codes from gaussian-splatting，gaussian-splatting is licensed under the Gaussian-Splatting License. Some files of gaussian-splatting may have been modified by Moore Threads Technology Co., Ltd.  Certain derivative work developed by Moore Threads Technology Co., Ltd are subject to the Gaussian-Splatting License.

## Contact
```
Bingling Li    :   lblhust903@gmail.com
Shengyi Chen   :   pythonchanner@gmail.com
```


