# Dependence

Install original diff-gaussian-rasterization and knn

```
cd submodules/diff-gaussian-rasterization
python -m setup.py install
cd submodules/simple-knn
python -m setup.py install
```

Install revised diff-gaussian-rasterization (if necessary)

GRID Training

```
cd rasterization_kernels/diff-gaussian-rasterization-half-gaussian
python -m setup.py install
```
New Metric like GSPP and MAGS.
```
cd rasterization_kernels/diff-gaussian-rasterization-metric
python -m setup.py install
```

# Train
default train
```
python train.py  -s dataset/garden -m output/garden_train --eval 
```
use dataset to save GPU memory 
```
python train_with_dataset.py  -s dataset/garden -m output/garden_train_with_dataset --eval
```
train with GRID 
```
CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    --master_addr=127.0.0.1 --master_port=7356 \
    train_MP_tree_partition.py -s /root/Nerf/Code/HT/DenseGaussian/dataset/garden\
        -m /root/Nerf/Predict/mipnerf360/garden/eval_mp --bvh_depth 2 \
        --eval \
        --epochs 187 \
        --scaling_lr_init 0.005 \
        --scaling_lr_final 0.005 \
        --densification_interval 100 \
        --densify_until_iter 15000 \
        --opacity_reset_interval 3000 \
        --max_batch_size 4  --max_load 8  
```

# Precision Validation

Shold be done after new implementation of diff-gaussian-rasterization and py file of train.

On scene garden, eval (161 for train, 24 for test), default hyper-parameters (only -m + -s + -eval + about 3w iters):

train.py: 

[ITER 30000] Evaluating all of test: PSNR 27.28

[ITER 30000] Evaluating all of train: PSNR 30.00

train_with_dataset.py: 

[ITER 30000] Evaluating all of test: PSNR 27.26

[ITER 30000] Evaluating all of train: PSNR 29.92

train_MP_tree_partition.py:

[ITER 30107] Evaluating test: PSNR 26.67

[ITER 30107] Evaluating train: PSNR 28.23

default parameters get worse result with bigger batchsize, if you just set batchsize=1(this also leads to low efficiency) num_gpu=2, num_model=4:

[ITER 29946] Evaluating test: L1 0.028083428197229903 PSNR 27.158511241277058

[ITER 29946] Evaluating train: L1 0.02086768550798297 PSNR 29.720123386383058

# Organizational Reformations

1. gaussian_renderer/__init__.py and scene/__init__.py have no additional implementation functionality. New implementation class should be a separate py file. 

2. Reformat the scene class, decouple functions that can be broken down, the role of the scene itself is changed to just mount configuration information.

3. The implementation of the new feature of diff-gaussian-rasterization, directly mounted to the rasterization_kernels folder.

# Custom Dataset
## Mill_19
as Mill_19 provide train/val set, we just follow its original division
do remember to config the path in scripts/mega_nerf_to_colmap.py to process data and organize files
so that SimpleScene would recongnize them
```
example structure:
--path/to/model
    |--images/ (train images)
    |--sparse/0/ (colmap model)
    |--test
        |--images (test images)
        |--sparse/cameras.txt and images.txt
```