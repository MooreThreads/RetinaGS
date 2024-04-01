CUDA_VISIBLE_DEVICES=1 python train_with_epoch.py \
    -s /jfs/shengyi.chen/HT/Data/Mill_19/OpenDataLab___Mill_19/colmap/Mill_19/building-pixsfm \
    -m backup/mill19_building_epoch \
    --iterations 300000 \
    --densify_from_iter 1000 --densify_until_iter 30000 --densification_interval 200 \
    --epochs 300 \
    --densify_until_epoch 0 \
    --eval_epoch_interval 1

CUDA_VISIBLE_DEVICES=5 python train_with_dataset.py \
    -s /jfs/shengyi.chen/HT/Data/Mill_19/OpenDataLab___Mill_19/colmap/Mill_19/building-pixsfm \
    -m backup/mill19_building_exp2 \
    --densify_from_iter 1000 --densify_until_iter 150000 --densification_interval 400 \
    --iterations 300000 --test_iterations 20000 80000 160000 200000 

# CUDA_VISIBLE_DEVICES=1 python render_metric.py \
#     -s /jfs/shengyi.chen/HT/Data/Mill_19/OpenDataLab___Mill_19/colmap/Mill_19/building-pixsfm \
#     -m backup/mill19_building \
#     --iterations 60000 \
#     --densify_from_iter 1000 --densify_until_iter 30000 --densification_interval 200 


CUDA_VISIBLE_DEVICES=2 python train_tmp.py \
    -s /jfs/shengyi.chen/HT/Data/mipnerf360/garden/A100_colmap-default_gaussian-splatting-default \
    -m backup/garden_tsize_4 --eval  

CUDA_VISIBLE_DEVICES=3 python train_tmp.py \
    -s /jfs/shengyi.chen/HT/Data/mipnerf360/garden/A100_colmap-default_gaussian-splatting-default \
    -m /jfs/shengyi.chen/HT/Data/mipnerf360/garden/train_with_dataset_garden_default-PM_dgt_00005 \
    --eval  --densify_grad_threshold 0.00005 --max_gaussians 35_000_000 --iterations 622001 --densify_until_iter 0

CUDA_VISIBLE_DEVICES=0 python train_tmp.py \
    -s /jfs/shengyi.chen/HT/Data/mipnerf360/garden/A100_colmap-default_gaussian-splatting-default \
    -m backup/garden_tsize_4 --eval  --densify_grad_threshold 0.00005 --max_gaussians 35_000_000



CUDA_VISIBLE_DEVICES=7 python train_with_epoch.py \
    -s "/jfs/shengyi.chen/HT/Data/MatrixCity/Block_all_unit-1m_choice-100/A100_colmap-default_gaussian-splatting-default" \
    -m back/city_t4 \
    --eval \
    --max_gaussians 35_000_000 \
    --epochs 730 \
    --densify_until_epoch 0 \
    --eval_epoch_interval 1

CUDA_VISIBLE_DEVICES=0 python train_with_epoch.py \
    -s "/jfs/shengyi.chen/HT/Data/MatrixCity/Block_all_unit-1m_choice-100/A100_colmap-default_gaussian-splatting-default" \
    -m "/jfs/shengyi.chen/HT/Predict/MatrixCity/Block_all_unit-1m_choice-100_default-PM_with_dataset_dgt_0010" \
    --eval \
    --max_gaussians 35_000_000 \
    --epochs 730 \
    --densify_until_epoch 0 \
    --eval_epoch_interval 1