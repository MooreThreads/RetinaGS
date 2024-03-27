# Dependence

Install original diff-gaussian-rasterization and knn

```
cd submodules/diff-gaussian-rasterization
python -m setup.py install
cd submodules/simple-knn
python -m setup.py install
```

Install revised diff-gaussian-rasterization (if necessary)

MP Training

'''
cd rasterization_kernels/diff-gaussian-rasterization-half-gaussian
python -m setup.py install

New Metric like GSPP and MAGS.

'''
cd rasterization_kernels/diff-gaussian-rasterization-metric
python -m setup.py install
'''

# Train

'''
python train.py  -s dataset/garden -m output/garden_train --eval 
python train_with_dataset.py  -s dataset/garden -m output/garden_train_with_dataset --eval
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
'''

# Precision Validation

Shold be done after new implementation of diff-gaussian-rasterization and py file of train.

On scene garden, eval (161 for train, 24 for test), default hyper-parameters (only -m + -s + -eval + about 3w iters):

1. train.py: 

47min31s

[ITER 30000] Evaluating all of test: PSNR 27.28

[ITER 30000] Evaluating all of train: PSNR 29.87

2. train_with_dataset.py: 

60min51s

[ITER 30000] Evaluating all of test: PSNR 27.26

[ITER 30000] Evaluating all of train: PSNR 29.92

3. train_MP_tree_partition.py:

双卡，104min08s

[ITER 30107] Evaluating test: PSNR 26.67

[ITER 30107] Evaluating train: PSNR 28.23

# Organizational Reformations

1. gaussian_renderer/__init__.py and scene/__init__.py have no additional implementation functionality. New implementation class should be a separate py file. 

2. Reformat the scene class, decouple functions that can be broken down, the role of the scene itself is changed to just mount configuration information.

3. The implementation of the new feature of diff-gaussian-rasterization, directly mounted to the rasterization_kernels folder.

