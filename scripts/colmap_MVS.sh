DATA_PATH=/root/Nerf/Code/DenseGaussian/data/data_xxx

mkdir -p $DATA_PATH/dense

colmap image_undistorter \
    --image_path "$DATA_PATH/images" \
    --input_path "$DATA_PATH/sparse/0" \
    --output_path "$DATA_PATH/dense"

colmap patch_match_stereo \
    --workspace_path "$DATA_PATH/dense" \
    --PatchMatchStereo.gpu_index 0

colmap stereo_fusion \
    --workspace_path "$DATA_PATH/dense" \
    --output_path "$DATA_PATH/dense/MVS_points3D.ply"

mv $DATA_PATH/dense/MVS_points3D.ply $DATA_PATH/sparse/0
