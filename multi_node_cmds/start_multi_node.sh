

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
conda activate retina_gs

env

export GLOO_SOCKET_TIMEOUT_MS=6000000
export NCCL_SOCKET_TIMEOUT=6000

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nnodes=$NNODES --node_rank=$NODE_RANK --nproc_per_node=8 --master_addr=$MASTER_ADDR --master_port=39527  \
    main.py -s data/Garden-1.6k -m model/Garden-1.6k_62M \
        --bvh_depth 4 --MAX_BATCH_SIZE 4  --MAX_LOAD 8 \
        -r 1 --eval \
        --position_lr_init 0.0000016 --position_lr_final 0.000000016 --densify_until_iter 0 \
        --points3D MVS_points3D --pointcloud_sample_rate 1 \
        --iterations 60000
