An example for 4-times downsampled graden (for the best consistentance with original 3D-GS train.py, it sets MAX_BATCH_SIZE as 1 which actually lowers the training speed):
```
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    --master_addr=127.0.0.1 --master_port=7356 \
    train_mp_tree.py -s dataset/garden \
        -m backup/mp_garden_shared_gs --bvh_depth 2 \
        --eval --log_level 20 --SHRAE_GS_INFO \
        --epochs 187 \
        --scaling_lr_init 0.005 \
        --scaling_lr_final 0.005 \
        --densification_interval 600 \
        --densify_until_iter 15000 \
        --opacity_reset_interval 3000 \
        --MAX_BATCH_SIZE 1  --MAX_LOAD 8 \
        --EVAL_INTERVAL_EPOCH 10 --SAVE_INTERVAL_EPOCH 30 \
        --ENABLE_REPARTITION --REPARTITION_INTERVAL_EPOCH 50  --REPARTITION_START_EPOCH 1 --REPARTITION_END_EPOCH 200
```
Here is a screenshot of tensorboard (The 2.337 hours is not very representive, there were other training processes on the same server.):
![image](logs/tb_screenshot.png)

