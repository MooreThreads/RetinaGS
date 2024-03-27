CUDA_VISIBLE_DEVICES=0,3 torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    --master_addr=127.0.0.1 --master_port=7356 \
    train_MP_tree_partition.py -s /jfs/shengyi.chen/HT/Data/mipnerf360/garden/A100_colmap-default_gaussian-splatting-default \
        -m backup/debug_mp_garden --bvh_depth 2 \
        --epochs 100 \
        --white_background \
        --scaling_lr_init 0.005 \
        --scaling_lr_final 0.00005 \
        --densification_interval 100 \
        --densify_until_iter 5000 \
        --opacity_reset_interval 3000 \
        --eval  --max_batch_size 4  --max_load 8


CUDA_VISIBLE_DEVICES=0,3 torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    --master_addr=127.0.0.1 --master_port=7356 \
    render_MP_tree_partition.py -s /jfs/shengyi.chen/HT/Data/mipnerf360/garden/A100_colmap-default_gaussian-splatting-default \
        -m backup/debug_mp_garden --bvh_depth 2 \
        --epochs 100 \
        --white_background \
        --scaling_lr_init 0.005 \
        --scaling_lr_final 0.00005 \
        --densification_interval 100 \
        --densify_until_iter 5000 \
        --opacity_reset_interval 3000 \
        --eval  --max_batch_size 4  --max_load 8        