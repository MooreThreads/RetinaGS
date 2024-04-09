# 单独提取图片
# python /root/Nerf/Code/MatrixCity-main/scripts/copy_image_via_json.py

DATA_PATH=/jfs/shengyi.chen/HT/Data/Mill_19/OpenDataLab___Mill_19/colmap/Mill_19/building-pixsfm

# mkdir $DATA_PATH/gt_pose
# colmap feature_extractor \
#     --ImageReader.camera_model PINHOLE \
#     --database_path "$DATA_PATH/gt_pose/database.db" \
#     --image_path "$DATA_PATH/images" \
#     --ImageReader.single_camera 1 \
#     --SiftExtraction.use_gpu 1

# 得到Colmap格式的GT Pose
# python /root/Nerf/Code/HT/gaussian-splatting_depth/nerf_to_colmap.py

# 手动创建空的points3D.txt
# cp /root/Nerf/Data/MatrixCity/block_D_unit-1m_center-64/gt_pose/sparse/real/points3D.txt $DATA_PATH/gt_pose/sparse/real

# 改内参
# python /root/Nerf/Code/HT/gaussian-splatting_depth/colmap_intrinsics.py

# 改image ID，对应到db
# python /root/Nerf/Code/HT/gaussian-splatting_depth/change_image_id.py


# colmap exhaustive_matcher \
#     --database_path "$DATA_PATH/gt_pose/database.db" \
#     --SiftMatching.use_gpu 1 \
#      --SiftMatching.gpu_index 2

# mkdir -p $DATA_PATH/gt_pose/triangulated/sparse

# colmap point_triangulator \
#     --database_path "$DATA_PATH/gt_pose/database.db" \
#     --image_path "$DATA_PATH/images" \
#     --input_path "$DATA_PATH/gt_pose/sparse/real" \
#     --output_path "$DATA_PATH/gt_pose/triangulated/sparse" \
#     --Mapper.ba_refine_focal_length 0 \
#     --Mapper.ba_refine_extra_params 0

mkdir -p $DATA_PATH/gt_pose/dense

colmap image_undistorter \
    --image_path "$DATA_PATH/images" \
    --input_path "$DATA_PATH/gt_pose/triangulated/sparse" \
    --output_path "$DATA_PATH/gt_pose/dense"

colmap patch_match_stereo \
    --workspace_path "$DATA_PATH/gt_pose/dense" \
    --PatchMatchStereo.gpu_index 7

colmap stereo_fusion \
    --workspace_path "$DATA_PATH/gt_pose/dense" \
    --output_path "$DATA_PATH/gt_pose/dense/fused.ply"
