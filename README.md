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

Our default, provided install method is based on Conda package and environment management:
```shell
conda env create --file environment.yml
conda activate gaussian_splatting
```
Please note that this process assumes that you have CUDA SDK **11** installed, not **12**. For modifications, see below.

If you can afford the disk space, we recommend using our environment files for setting up a training environment identical to ours. If you want to make modifications, please note that major version changes might affect the results of our method. However, our (limited) experiments suggest that the codebase works just fine inside a more up-to-date environment (Python 3.8, PyTorch 2.0.0, CUDA 12). Make sure to create an environment where PyTorch and its CUDA runtime version match and the installed CUDA SDK has no major version difference with PyTorch's CUDA version.

### RetinaGS Setup
Install additional packages for RetinaGS.

```
pip install numba opencv-python scipy
```

Install new raster-kernel in rasterization_kernels/diff-gaussian-rasterization-half-gaussian.

```
cd rasterization_kernels/diff-gaussian-rasterization-half-gaussian
python -m setup.py install
```

Please note that we only test RetinaGS on Linux System.

## Usage

### Get pretrained models
[[Garden]](https://ai-reality.github.io/RetinaGS/)
[[ScanNet++]](https://ai-reality.github.io/RetinaGS/)
... 

### Data 
The data should be orgnised as follows:
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

### Evaluation
... demo ...

### Model Zoo
Model, Splited Model, PSNR, #GS。。。。
Garden
Scan
MatrixCity Aerial
Full MatrixCity
...



### Training 
For single node, an example command is:
```
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    --master_addr=127.0.0.1 --master_port=7356 \
    train_MP_tree.py -s path/2/data \
        -m path/2/model --bvh_depth 2 \
        --eval \
        --epochs 187 \
        --max_batch_size 4  --max_load 8  
```
Here, you would create 2**(bvh_depth) submodels for 2 processes, namely 2 submodels for each process. The max_batch_size and max_load are arguments for controlling memory cost, a render task for a submodel weight 1 load, thus "--max_batch_size 4  --max_load 8" just set every batch as size of 4 in this case. 

For multiple nodes, start command on each node with corresponding parameters, and example shell scripts for launching/stopping multiple nodes training can be found in multi_node_cmds/: 
```
torchrun --nnodes=$NNODES --node_rank=$NODE_RANK --nproc_per_node=$NUM_GPU_PER_NODE \
    --master_addr=$MASTER_ADDR --master_port=39527  \
    train_MP_tree.py \
        -s path/2/data \
        -m path/2/model --bvh_depth 6 \
        --epochs 40  --eval \
        --bvh_depth 4 \
        --max_batch_size 4  --max_load 8
``` 

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


