# RetinaGS: Scalable Training for Dense Scene Rendering with Billion-Scale 3D Gaussians

<img src="./assets/teaser.png">

We introduce RetinaGS, which explores the possibility of training high-parameter 3D Gaussian splatting (3DGS) models on large-scale, high-resolution datasets. This codebase maintain a model parallel traning framework for native 3DGS which uses a proper rendering equation and can be applied to any scene and arbitrary distribution of Gaussian primitives. 

<img src="./assets/pipeline.png">


[[Project Page]](https://ai-reality.github.io/RetinaGS/)
[[Paper]](https://arxiv.org/pdf/2406.11836)

## Prerequisites

1. Clone this repository:

```
git clone https://github.com/MooreThreads/RetinaGS.git --recursive
cd RetinaGS
```

2. Installation:

```shell
conda env create --file environment.yml
conda activate retina_gs
```

Please note that we only test RetinaGS on Ubuntu 20.04.1 LTS.

<details>
<summary><span style="font-weight: bold;">More installation detail</span></summary>
  
  If you already have the environment set up from the [original 3DGS repository](https://github.com/graphdeco-inria/gaussian-splatting), you can quickly get started by running the following command:

  ```shell
  pip install rasterization_kernels/diff-gaussian-rasterization-half-gaussian/ numba scipy
  ```

</details>
<br>

## Quick Start

1. Download the testing scence and the corresponded pretrained model from [GoogleDrive]() and uncompress them under the root directory.

2. Evaluate the model with the following command:
```
CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=5356 \
    main.py -s data/Garden-1.6k -m model/Garden-1.6k_5M \
        --bvh_depth 2 --MAX_BATCH_SIZE 2  --MAX_LOAD 2 \
        -r 1 --eval --EVAL_ONLY --SAVE_EVAL_IMAGE --SAVE_EVAL_SUB_IMAGE
```

You can also use models trained with the [[original 3DGS repository]](https://github.com/graphdeco-inria/gaussian-splatting) by specifying the -s (source path) and -m (model path) parameters.

3. The final render results, as well as the intermediate outputs of each submodule, can be found in model/Garden-1.6k_5M/img.

### Model Zoo

The pre-trained models and corresponding data are available for download on [GoogleDrive]().

| data                                                          | model                                                         | PSNR | #GS   |Resolution|
|:-----------------:                                            |:-----------------:                                            |:----:|:-----:|:-----:   |
| [[Garden-1.6k]]()                                             | [[Garden-1.6k_5M]]()                                          |27.33 |5.6M   |1600×1036 |
| [[Garden-1.6k]]()                                             | [[Garden-1.6k_62M]]()                                         |27.63 |62.9M  |1600×1036 |
| [[Garden-full]]()                                             | [[Garden-full_62M]]()                                         |26.95 |62.9M  |5185×3359 |
| [[MatrixCity-Aerial]]()                                       | [[MatrixCity-Aerial_217M]]()                                  |27.77 |217.3M |1920×1080 |

<!-- M means Million. Add -r 1600 flag while evaluate Room-1.6k. -->


## Usage 

### Evaluation

To evaluate a model, use the following command:

```
CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=5356 \
    main.py -s data/Garden-1.6k -m model/Garden-1.6k_62M \
        --bvh_depth 2 --MAX_BATCH_SIZE 2  --MAX_LOAD 4 \
        -r 1 --eval --EVAL_ONLY --SAVE_EVAL_IMAGE --SAVE_EVAL_SUB_IMAGE
```

<details>
<summary><span style="font-weight: bold;">More Configuration options</span></summary>

  #### CUDA_VISIBLE_DEVICES=0,1
  Assigns GPUs numbered CUDA_0 and CUDA_1 for evaluation.
  #### --nnodes=1 --nproc_per_node=2
  Specifies the use of 1 machine and 2 GPUs.
  #### --master_addr=127.0.0.1 --master_port=7356
  Sets the host and port for torchrun. Ensure that the --master_port is unique for different tasks on the same machine to avoid conflicts.
  #### --source_path / -s
  The path to the source directory containing a COLMAP or Synthetic NeRF data set.
  #### --model_path / -m 
  The path where the trained model is stored. 
  #### --bvh_depth
  Controls the number of submodels generated, creating 2<sup>bvh_depth</sup> submodels. For example, bvh_depth=2 results in a total of 4 submodels.
  #### --MAX_BATCH_SIZE --MAX_LOAD 
  These parameters manage memory usage, a render task for a submodel weight 1 load, thus "--MAX_BATCH_SIZE 2  --MAX_LOAD 4" just set every batch as size of 2 in this case. Reduce these values if GPU memory is insufficient.
  #### --eval
  Add this flag to use a MipNeRF360-style training/test split for evaluation.
  #### --EVAL_ONLY --SAVE_EVAL_IMAGE --SAVE_EVAL_SUB_IMAGE
  Limits the operation to evaluation only, saving both the final rendered images and the sub-images from each submodel.

</details>
<br>


### Training

Start training via: 
```
CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=5551 \
    main.py -s data/Garden-1.6k -m model/Garden-1.6k_5M \
        --bvh_depth 1 --MAX_BATCH_SIZE 1  --MAX_LOAD 2 \
        -r 1 --eval
```

<details>
<summary><span style="font-weight: bold;">More Configuration options</span></summary>


  #### --resolution / -r
  Specifies resolution of the loaded images before training. If provided 1, 2, 4 or 8, uses original, 1/2, 1/4 or 1/8 resolution, respectively. For all other values, rescales the width to the given number while maintaining image aspect. If not set and input image width exceeds 1.6K pixels, inputs are automatically rescaled to this target.
  #### --iterations
  The total number of training iterations, defaulting to 30_000.
  #### --epochs
  The total number of training epochs. This is only effective if --iterations is not specified.

</details>
<br>


### Training with MVS Initialization

In our paper, we use MVS initialization to control the number of Gaussian splats. You can obtain MVS results via Colmap using this script: `scripts/colmap_MVS.sh`. We recommend stopping the growth and pruning of splats during the training process. Additionally, adjusting a few hyperparameters can help stabilize the training.

The following is a sample command:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=5551 \
    main.py -s data/Garden-1.6k -m model/Garden-1.6k_62M \
        --bvh_depth 2 --MAX_BATCH_SIZE 1  --MAX_LOAD 2 \
        -r 1 --eval \
        --position_lr_init 0.0000016 --position_lr_final 0.000000016 --densify_until_iter 0 \
        --points3D MVS_points3D --pointcloud_sample_rate 1 \
        --EVAL_INTERVAL_EPOCH 50 --SAVE_INTERVAL_EPOCH 50      
```

<details>
<summary><span style="font-weight: bold;">More Configuration options</span></summary>

  #### --position_lr_init --position_lr_final
  Initial and Final 3D position learning rate, 1.6 × 10<sup>-4</sup> to 1.6 × 10<sup>-6</sup> by default. Since the primitives are initialized with relatively accurate position parameters from MVS, we reduce the learning rate for the position parameters in all primitives from 1.6 × 10<sup>-6</sup> to 1.6 × 10<sup>-8</sup> with a exponential decay function

  #### --densify_until_iter
  Specifies the iteration at which densification stops, defaulting to 15,000 and set to 0 to disable.

  #### --points3D
  The point cloud file used for initialization.

  #### --pointcloud_sample_rate
  Sets the downsampling rate at initialization; for instance, providing N uses 1/N of the point cloud. Increase the downsampling ratio when using MVS initialization if GPU memory is insufficient.

  ### --EVAL_INTERVAL_EPOCH --SAVE_INTERVAL_EPOCH
  Specifies the interval epoch of saving model and evaluation.

  #### --SPLIT_MODEL
  Enables reading individual ply files for each submodel plus interface information, which can reduce read and write overhead with numerous GS.

</details>
<br>

### Multi-node Training

Shell scripts for starting or stopping multi-node training are available in the multi_node_cmds/ directory.
<br>



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


