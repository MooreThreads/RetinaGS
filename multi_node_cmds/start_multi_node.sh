

/bin/bash
cd /path/2/RetinaGS/


# pip install tqdm plyfile tensorboard numba
ps aux|grep torchrun|awk 'NR==1'|awk '{print $2}'|xargs kill -9
# kill existing MP process
ps aux|grep train_MP_|awk '{print $2}'|xargs kill -9
ps aux|grep render_MP_|awk '{print $2}'|xargs kill -9


source /root/anaconda3/bin/activate
conda init
source ~/.bashrc
# conda activate dense
# pip install tensorboad
conda activate gaussian_splatting

env

export GLOO_SOCKET_TIMEOUT_MS=6000000
export NCCL_SOCKET_TIMEOUT=6000

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nnodes=$NNODES --node_rank=$NODE_RANK --nproc_per_node=8 \
    --master_addr=$MASTER_ADDR --master_port=39527  \
    train_MP_tree_partition.py \
        -s path/2/data \
        -m path/2/model \
        -r 1 --epochs 40  --eval --points3D $NAME_OF_PLY_FILE \
        --scaling_lr_init 0.0005 --scaling_lr_final 0.000005 --position_lr_init 0.00000016 --position_lr_final 0.0000000016 \
        --densify_until_iter 0 --SKIP_SPLIT --SKIP_CLONE --opacity_reset_interval 3000 --densification_interval 100 \
        --bvh_depth 4 \
        --max_batch_size 4  --max_load 4 \
        --EVAL_INTERVAL_EPOCH 1 --SAVE_INTERVAL_EPOCH 1 --SAVE_INTERVAL_ITER 10000 \
        --CKPT_MAX_NUM 2
