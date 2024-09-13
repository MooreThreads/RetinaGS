# RetinaGS: Scalable Training for Dense Scene Rendering with Billion-Scale 3D Gaussians

<img src="./assets/teaser.png">

We introduce RetinaGS, which explores the possibility of training high-parameter 3D Gaussian splatting (3DGS) models on large-scale, high-resolution datasets. This codebase maintain a model parallel traning framework for native 3DGS which uses a proper rendering equation and can be applied to any scene and arbitrary distribution of Gaussian primitives. 

<img src="./assets/pipeline.png">


[[Project Page]](https://ai-reality.github.io/RetinaGS/)
[[Paper]](https://arxiv.org/pdf/2406.11836)

## Prerequisites

1. Clone this repository:
```
git clone https://github.com/mthreads/DenseGaussian.git --recursive
cd DenseGaussian
```


2. Installation:

```shell
conda env create --file environment.yml
conda activate retina_gs
```

Please note that we only test RetinaGS on Ubuntu 20.04.1 LTS.

## Usage

### Evaluation
Get data and pretrained models ([[Garden]](https://ai-reality.github.io/RetinaGS/)). 把data_Garden放到data/下，model_Garden放到model/下.

```
CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=7356 \
    main_MP_tree.py -s data/data_Garden -m model/model_Garden \
        --bvh_depth 2 --WHOLE_MODEL \
        --max_batch_size 4  --max_load 8  \
        -r 1 --eval --EVAL_ONLY --SAVE_EVAL_IMAGE --SAVE_EVAL_SUB_IMAGE
```

Our implement is based on 3DGS (https://github.com/graphdeco-inria/gaussian-splatting). 使用原始仓库训练的模型可以直接跑（替换-s和-m即可）. 
<details>
<summary><span style="font-weight: bold;">Command Line Arguments for main_MP_tree.py under Evaluation</span></summary>
Arguments of 3DGS我们大部分保留. 

  #### CUDA_VISIBLE_DEVICES=0,1
  指定编号为CUDA_0和CUDA_1的GPU参与Evaluation.
  #### --nnodes=1 --nproc_per_node=2
  机器数量为1，GPU数量为2.
  #### --master_addr=127.0.0.1 --master_port=7356
  the host and port of torchrun. 注意同一台机器上不同训练任务间的--master_port需要不同.
  #### --source_path / -s
  Path to the source directory containing a COLMAP or Synthetic NeRF data set.
  #### --model_path / -m 
  Path where the trained model is stored. 
  #### --resolution / -r
  Specifies resolution of the loaded images before training. If provided 1, 2, 4 or 8, uses original, 1/2, 1/4 or 1/8 resolution, respectively. For all other values, rescales the width to the given number while maintaining image aspect. If not set and input image width exceeds 1.6K pixels, inputs are automatically rescaled to this target.
  #### --eval
  Add this flag to use a MipNeRF360-style training/test split for evaluation.
  #### --bvh_depth
  Argument for controlling the number of submodels. Here, you would create 2**(bvh_depth) submodels for 2 GPUs, namely 2 submodels for each GPU. 
  #### --WHOLE_MODEL
  仅读入单个ply
  #### --max_batch_size --max_load 
  Arguments for controlling memory cost, a render task for a submodel weight 1 load, thus "--max_batch_size 4  --max_load 8" just set every batch as size of 4 in this case.
  #### --EVAL_ONLY --SAVE_EVAL_IMAGE --SAVE_EVAL_SUB_IMAGE
  仅进行Evaluation，且保存图像和每个submodel输出的子图像。

</details>
<br>



### Training 
For single machine, an example of using default densification strategy and Colmap Initialization  command is:
```
CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --nproc_per_node=2 \
    --master_addr=127.0.0.1 --master_port=7356 \
    train_MP_tree.py -s path/2/data \
        -m path/2/model --bvh_depth 2 \
        --eval \
        --epochs 187 \
        --max_batch_size 4  --max_load 8
```


从MVS Initialization出发，关闭点管理 (RetinaGS paper中的训练方式)
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

For multiple nodes, start command on each node with corresponding parameters, and example shell scripts for launching/stopping multiple nodes training can be found in multi_node_cmds/

### Model Zoo

| Datasets                                                      | PSNR | #GS   |
|:-----------------:                                            |:----:|:-----:|
| [[Garden-1.6k]](https://ai-reality.github.io/RetinaGS/)       |27.74 |62.94M |
| [[Garden-full]](https://ai-reality.github.io/RetinaGS/)       |27.06 |62.94M |
| [[ScanNet++]](https://ai-reality.github.io/RetinaGS/)         |29.71 |47.59M |
| [[MatrixCity-M]](https://ai-reality.github.io/RetinaGS/)      |31.12 |62.18M |
| [[Mega-NeRF]](https://ai-reality.github.io/RetinaGS/)         |25.09 |27.9M  |
| [[MatrixCity-Aerial]](https://ai-reality.github.io/RetinaGS/) |27.70 |217.3M |

M means Million. See Appendix in [[Paper]](https://arxiv.org/pdf/2406.11836) for complete results.

## To Do
- Output as one whole model  
- 优化读入单独ply（send recv形式）
- Model Zoo（引导用户下载官方数据，再把包括MVS结果在内的colmap放到里面）
- Colmap MVS脚本

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


