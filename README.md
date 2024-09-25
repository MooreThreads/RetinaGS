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

## Quick Start

1. Download the testing scence and the corresponded pretrained model from [GoogleDrive]() and uncompress them under the root path.

2. Evaluate the model using the command below:
```
CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=5356 \
    main.py -s data/data_Garden -m model/model_Garden \
        --bvh_depth 2 --MAX_BATCH_SIZE 2  --MAX_LOAD 2 \
        --eval --EVAL_ONLY --SAVE_EVAL_IMAGE --SAVE_EVAL_SUB_IMAGE
```
3. The final render results, as well as the intermediate outputs of each submodule, can be found in `xxxxxx`.

### Model Zoo

The pre-trained models on several public datasets are available for download on [GoogleDrive]().

| Data and Model                                                | PSNR | #GS   |Resolution|
|:-----------------:                                            |:----:|:-----:|:-----:   |
| [[Room-1.6k]](https://ai-reality.github.io/RetinaGS/)         |32.86 |22.41M |1600×1036 |
| [[Bicycle-full]](https://ai-reality.github.io/RetinaGS/)      |24.86 |31.67M |4944×3284 |
| [[MatrixCity-Aerial]](https://ai-reality.github.io/RetinaGS/) |27.70 |217.3M |1920×1080 |

<!-- M means Million. Add -r 1600 flag while evaluate Room-1.6k. -->


## Usage 

<!-- ### Data
The data structure should be organised as follows:
```
``` -->


### Evaluation

```
CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=5356 \
    main.py -s data/data_Garden -m model/model_Garden \
        --bvh_depth 2 --MAX_BATCH_SIZE 2  --MAX_LOAD 2 \
        --eval --EVAL_ONLY --SAVE_EVAL_IMAGE --SAVE_EVAL_SUB_IMAGE
```

Our implement is based on [3DGS](https://github.com/graphdeco-inria/gaussian-splatting). Models trained using the 3DGS repository can directly run multi-GPU evaluation by simply replacing the -s and -m parameters.

<details>
<summary><span style="font-weight: bold;">More Configuration options</span></summary>

  #### CUDA_VISIBLE_DEVICES=0,1
  Designate GPUs numbered CUDA_0 and CUDA_1 for Evaluation.
  #### --nnodes=1 --nproc_per_node=2
  The number of machine is 1，the number of GPU is 2.
  #### --master_addr=127.0.0.1 --master_port=7356
  the host and port of torchrun. Ensure that the --master_port is different for different training tasks on the same machine.
  #### --source_path / -s
  Path to the source directory containing a COLMAP or Synthetic NeRF data set.
  #### --model_path / -m 
  Path where the trained model is stored. 
  #### --bvh_depth
  Argument for controlling the number of submodels. Here, you would create 2<sup>bvh_depth</sup> submodels. In this example, bvh_depth=2, namely total 4 submodels (2 submodels for each GPU). 
  #### --MAX_BATCH_SIZE --MAX_LOAD 
  Arguments for controlling memory cost, a render task for a submodel weight 1 load, thus "--MAX_BATCH_SIZE 4  --MAX_LOAD 8" just set every batch as size of 4 in this case. If there is insufficient GPU memory, consider reducing these values.
  #### --eval
  Add this flag to use a MipNeRF360-style training/test split for evaluation.
  #### --EVAL_ONLY --SAVE_EVAL_IMAGE --SAVE_EVAL_SUB_IMAGE
  Perform evaluation only, and save both the rendered images and the sub-images output by each submodel involved in the rendering.

</details>
<br>


### Training

Start training via: 
```
CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=7356 \
    main.py -s data/data_Garden -m model/model_Garden_default_densification \
        --bvh_depth 2 --MAX_BATCH_SIZE 2  --MAX_LOAD 2 \
        -r 1 --eval
```

<details>
<summary><span style="font-weight: bold;">More Configuration options</span></summary>


  #### --resolution / -r
  Specifies resolution of the loaded images before training. If provided 1, 2, 4 or 8, uses original, 1/2, 1/4 or 1/8 resolution, respectively. For all other values, rescales the width to the given number while maintaining image aspect. If not set and input image width exceeds 1.6K pixels, inputs are automatically rescaled to this target.
  #### --interations
  Number of total iterations to train for, 30_000 by default.
  #### --epochs
  Number of total epochs to train for. Effective only when --iterations is not specified.

</details>
<br>


### Training with MVS Initialization

In our paper, we use MVS initialization to control the number of Gaussian splats. You can obtain MVS results via Colmap using this script: `scripts/colmap_MVS.sh`. We recommend stopping the growth and pruning of splats during the training process. Additionally, adjusting a few hyperparameters can help stabilize the training. The following is a sample command:

```
CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=8356 \
    main.py -s data/data_Garden -m model/model_Garden_MVS \
        --bvh_depth 2 --MAX_BATCH_SIZE 2  --MAX_LOAD 2 \
        -r 1 --eval \
        --position_lr_init 0.0000016 --position_lr_final 0.000000016 --densify_until_iter 0 \
        --points3D MVS_points3D --pointcloud_sample_rate 1        
```

<details>
<summary><span style="font-weight: bold;">More Configuration options</span></summary>

  #### --position_lr_init --position_lr_final
  Initial and Final 3D position learning rate, 1.6 × 10<sup>-4</sup> to 1.6 × 10<sup>-6</sup> by default. Since the primitives are initialized with relatively accurate position parameters from MVS, we reduce the learning rate for the position parameters in all primitives from 1.6 × 10<sup>-6</sup> to 1.6 × 10<sup>-8</sup> with a exponential decay function

  #### --densify_until_iter
  Iteration where densification stops, ```15000``` by default and ```0``` for abandon.

  #### --points3D
  Specify the point cloud file used for initialization.

  #### --pointcloud_sample_rate
  Specify the downsampling rate at initialization; if N is provided, use 1/N of the point cloud. Consider increasing the downsampling ratio when using MVS initialization if there is not enough GPU memory.

  #### --SPLIT_MODEL
  Output individual ply files for each submodel plus interface information; consider adding this flag to improve read and write overhead when there are too many GS.

  #### --NOT_SHRAE_GS_INFO
  By dafult, we transmit interface GS via communication, achieving the equivalent of single-GPU training results in formulation together with alpha-blending splitting.
  When the --SPLIT_MODEL flag is enabled, consider adding the --NOT_SHARE_GS_INFO flag to slightly speed up training and reduce GPU memory usage.

</details>
<br>

### Multi-node Training

We provide shell scripts to start or stop multi-node training, which can be found in the `multi_node_cmds/` directory.
<br>



## To Do
- [ ] Model Zoo的准备和描述(说明MatrixCity-Aerial的下载和推理)
- [ ] 更多训练参数描述
- [ ] 说明paper呈现结果是用的另一个分支（本分支主要优化结构，使其更易读易改）
- [ ] 放到新仓，修改网址, 修改所有densegaussian为RetinaGS
- [x] 清理多余文件
- [x] 默认打开--SHRAE_GS_INFO
- [x] 翻译并polish
- [x] 加上指定iteration的训练
- [x] 1.6k输出时多余提示
- [x] 支持Evaluation输出LPIPS和SSIM
- [x] 统一evaluation输出到外层
- [x] 使用--SPLIT_MODEL代替--WHOLE_MODEL
- [x] 新读入单独ply形式（通信使用send recv形式，shared GS形式-无交集，可达到无损，强制Guide用户使用）
- [x] 测试Output as one whole model + --SHRAE_GS_INFO（证明和单卡训练结果接近，可近乎无损合并+重新分割）
- [x] data_Garden_MVS（降采样4倍Graden，作为示例）
- [x] Output as one whole model（不加shared GS，边界面会出问题）
- [x] Colmap MVS脚本 + 测试 + 说明
- [x] 读入单独ply（无shared GS）

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
Shengyi Chen   :   chenshengyi@std.uestc.edu.cn
```


